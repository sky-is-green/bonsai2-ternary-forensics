"""HF <-> ggml (Prism) tensor names and layout for the qwen3_5 export.

``materialize.py`` maps a Prism GGUF *to* HF.  The export bridge needs the
inverse: HF names -> ggml names, and HF layout -> Prism layout (the GDN V-head
reorder, ``ssm_a = -exp(A_log)``, and the ``ssm_conv1d`` axis order).  The name
maps are derived from ``materialize.GLOBAL_MAP`` / ``materialize.LAYER_MAP`` so
they cannot drift; the layout functions are exact inverses of
``materialize.reorder`` / ``materialize.materialize_tensor``.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from bonsai_forensics import materialize

# HF name -> ggml name (inverse of materialize's maps).
GLOBAL_HF_TO_GGML = {hf: ggml for ggml, hf in materialize.GLOBAL_MAP.items()}
# HF layer suffix (e.g. "linear_attn.in_proj_qkv.weight") -> ggml stem.
LAYER_HF_TO_GGML = {hf: stem for stem, hf in materialize.LAYER_MAP.items()}
_VPERM_STEMS = materialize._VPERM_STEMS


def hf_to_ggml(name: str) -> str:
    """Map a primal HF tensor name to its Prism ggml name."""
    if name in GLOBAL_HF_TO_GGML:
        return GLOBAL_HF_TO_GGML[name]
    parts = name.split(".")
    if len(parts) >= 4 and parts[0] == "model" and parts[1] == "layers":
        suffix = ".".join(parts[3:])
        stem = LAYER_HF_TO_GGML.get(suffix)
        if stem is None:
            raise ValueError(f"unmapped HF tensor {name!r}")
        return f"blk.{parts[2]}.{stem}"
    raise ValueError(f"unmapped HF tensor {name!r}")


def ggml_to_hf(name: str) -> str:
    """Map a Prism ggml tensor name to its primal HF name."""
    if name in materialize.GLOBAL_MAP:
        return materialize.GLOBAL_MAP[name]
    if name.startswith("blk."):
        _, layer, stem = name.split(".", 2)
        if stem not in materialize.LAYER_MAP:
            raise ValueError(f"unmapped ggml tensor {name!r}")
        return f"model.layers.{layer}.{materialize.LAYER_MAP[stem]}"
    raise ValueError(f"unmapped ggml tensor {name!r}")


def _inverse_permutation(permutation: np.ndarray) -> np.ndarray:
    return np.argsort(permutation)


def reorder_to_prism(array: np.ndarray, stem: str,
                     geometry: materialize.GdnGeometry) -> np.ndarray:
    """Inverse of ``materialize.reorder`` (HF layout -> Prism layout)."""
    nv, nk, hd, hk = geometry.nv, geometry.nk, geometry.hd, geometry.hk
    if nv == nk:
        return array
    qk = 2 * nk * hk
    if stem in ("attn_qkv.weight", "ssm_conv1d.weight"):
        inverse = _inverse_permutation(geometry.vperm(hd))
        return np.concatenate([array[:qk], array[qk:][inverse]], axis=0)
    if stem == "attn_gate.weight":
        return array[_inverse_permutation(geometry.vperm(hd))]
    if stem in _VPERM_STEMS:
        return array[_inverse_permutation(geometry.vperm(1))]
    return array


def to_prism_layout(hf_name: str, array: np.ndarray,
                    geometry: materialize.GdnGeometry) -> np.ndarray:
    """Transform one HF tensor into Prism's stored layout (inverse of materialize).

    Order matters and mirrors ``materialize.materialize_tensor``: it squeezes the
    ``conv1d`` axis, reorders, then applies the ``ssm_a`` ``-exp`` (materialize
    applies ``log(-x)`` *before* its reorder).
    """
    parts = hf_name.split(".")
    suffix = ".".join(parts[3:]) if len(parts) >= 4 and parts[:2] == ["model", "layers"] else ""
    if suffix == "linear_attn.conv1d.weight":
        array = array[:, 0, :]  # HF (out, 1, kernel) -> Prism (out, kernel)
    stem = LAYER_HF_TO_GGML.get(suffix)
    if stem is not None:
        array = reorder_to_prism(array, stem, geometry)
    if suffix == "linear_attn.A_log":
        array = -np.exp(array)  # HF A_log -> Prism -exp(A_log)
    return array


__all__ = [
    "GLOBAL_HF_TO_GGML",
    "LAYER_HF_TO_GGML",
    "ggml_to_hf",
    "hf_to_ggml",
    "reorder_to_prism",
    "to_prism_layout",
]
