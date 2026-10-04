"""``prism.hadamard.*`` metadata — the rotated-basis runtime contract.

The T9 packer writes only ``general.*`` keys, but a folded-basis artifact is
unrunnable (and silently wrong) without the rotation block that the fork reads
in ``src/llama-model.cpp`` (``llama_model::load``) and that
``bonsai_forensics.prism_loader`` validates.  This module builds that block and
adds it to a GGUF writer with the exact element types the C++ side expects
(``sign_widths`` / ``sign_values`` are INT32 arrays; the loader reads them into
``std::vector<int32_t>``).

Contract (fork ``prism.hadamard.version == 1``):

- ``block_size``  positive power of two; divides every rotated width
- ``transform``   ``normalized-sylvester-walsh-hadamard``
- ``axis``        ``input-last-dimension``
- ``sign_mode``   ``identity`` (plain block WHT) or ``explicit`` (±1 table)
- ``weight_names`` tensors whose input axis is folded (non-empty)
- ``sign_widths``/``sign_values``  explicit mode only; each width is a full
  rotated dimension, its signs concatenated per block
- ``inverse_weight_names``  row tables needing the inverse after lookup
  (the fork accepts only ``token_embd.weight``)
- ``gdn_v_grouped``  optional Gated-DeltaNet V-head reorder flag
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

VERSION = 1
TRANSFORM = "normalized-sylvester-walsh-hadamard"
AXIS = "input-last-dimension"
SIGN_MODES = ("identity", "explicit")


def sign_table(
    rotations: Mapping[int, Sequence[np.ndarray]],
) -> tuple[list[int], list[int]]:
    """Flatten ``{width: [per-block ±1 vectors]}`` to ``(widths, values)``.

    Every width must equal the concatenated block length and be a multiple of
    the runtime block size (checked by :func:`prism_hadamard_metadata`).
    """
    widths: list[int] = []
    values: list[int] = []
    for width, blocks in rotations.items():
        width = int(width)
        if width <= 0:
            raise ValueError(f"sign width must be positive, got {width}")
        if not blocks:
            raise ValueError(f"sign width {width} has no blocks")
        flat = np.concatenate([np.asarray(b, dtype=np.int64).reshape(-1) for b in blocks])
        if flat.size != width:
            raise ValueError(
                f"sign width {width}: concatenated signs have {flat.size} entries")
        if not np.all(np.isin(flat, (-1, 1))):
            raise ValueError(f"sign width {width}: signs must be ±1")
        widths.append(width)
        values.extend(int(v) for v in flat)
    return widths, values


def prism_hadamard_metadata(
    *,
    block_size: int,
    weight_names: Sequence[str],
    sign_mode: str = "identity",
    rotations: Mapping[int, Sequence[np.ndarray]] | None = None,
    axis: str = AXIS,
    transform: str = TRANSFORM,
    inverse_weight_names: Sequence[str] = (),
    gdn_v_grouped: bool | None = None,
) -> dict:
    """Build the metadata mapping (validated; ready for ``add_prism_metadata``)."""
    block_size = int(block_size)
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError(f"block_size must be a positive power of two, got {block_size}")
    names = [str(n) for n in weight_names]
    if not names:
        raise ValueError("weight_names is empty")
    if len(set(names)) != len(names):
        raise ValueError("weight_names contains duplicates")
    if sign_mode not in SIGN_MODES:
        raise ValueError(f"sign_mode must be one of {SIGN_MODES}, got {sign_mode!r}")
    if transform != TRANSFORM:
        raise ValueError(f"unsupported transform {transform!r}")
    if axis != AXIS:
        raise ValueError(f"unsupported axis {axis!r}")

    meta: dict = {
        "prism.hadamard.version": VERSION,
        "prism.hadamard.block_size": block_size,
        "prism.hadamard.transform": transform,
        "prism.hadamard.axis": axis,
        "prism.hadamard.sign_mode": sign_mode,
        "prism.hadamard.weight_names": names,
    }
    if sign_mode == "explicit":
        if not rotations:
            raise ValueError("sign_mode 'explicit' needs a rotations table")
        widths, values = sign_table(rotations)
        for width in widths:
            if width % block_size:
                raise ValueError(
                    f"sign width {width} is not a multiple of block {block_size}")
        meta["prism.hadamard.sign_widths"] = widths
        meta["prism.hadamard.sign_values"] = values
    elif rotations is not None:
        raise ValueError("sign_mode 'identity' takes no rotations table")

    inverse = [str(n) for n in inverse_weight_names]
    if inverse:
        meta["prism.hadamard.inverse_weight_names"] = inverse
    if gdn_v_grouped is not None:
        meta["prism.hadamard.gdn_v_grouped"] = bool(gdn_v_grouped)
    return meta


def add_prism_metadata(writer, metadata: Mapping) -> None:
    """Add the block to a ``pack_gguf.GGUFWriter`` with runtime-exact types."""
    from bonsai_forensics import pack_gguf

    int32_arrays = {
        "prism.hadamard.sign_widths",
        "prism.hadamard.sign_values",
    }
    for key, value in metadata.items():
        if key in int32_arrays:
            writer.add_metadata(key, [int(v) for v in value],
                                element_type=pack_gguf.GGUF_TYPE_INT32)
        else:
            writer.add_metadata(key, value)
