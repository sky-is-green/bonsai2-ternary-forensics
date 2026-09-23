"""ROCm loader for Prism's PQ2_0 ternary GGUF: self-describing basis + unrotation.

Bonsai 2's GGUF embeds its own Hadamard basis. The `prism.hadamard.*` metadata
carries the block size, transform, axis, the explicit full-width ±1 sign
vectors, the 401 forward-rotated tensor names, and the single inverse-rotated
tensor (`token_embd.weight`). The embedded `sign_values` were verified
byte-identical to the published `hadamard-signs-*.npy` manifest, so the loader
needs neither basis recovery nor an external manifest — the file describes
itself.

Prism's bundled MLX runtime (`runtime/runtime.py`, Apache-2.0) fixes the
semantics we have to reproduce:

- `fwht(x, forward)` = signs first, then block-Hadamard = `R x`;
- `fwht(x, inverse)` = block-Hadamard, then signs = `Rᵀ x`;
- a folded linear stores `D ≈ W_hf Rᵀ` and computes `fwht(x) @ Dᵀ = x (D R)ᵀ`,
  so its effective unrotated weight is `W_hf = D R`;
- `token_embd` applies the inverse transform to its *output* (there is no input
  activation to rotate), which also yields `W_hf = D R`.

So all 402 ternary tensors are recovered by one operation, `W_hf = D @ R`, with
`R = (1/√n) H diag(S)` applied per 1024-block along the ggml `ne0` (input)
axis. In this module that is `rotation.apply_rotation(D, signs, transpose=True)`.

This module is CPU-only and read-only: it streams the GGUF header, validates the
embedded basis against `spec.md` (tbr-1.2), and can dequantize + unrotate any
tensor. Serving the recovered weights on ROCm is the caller's job.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from bonsai_forensics import pq2_0, rotation
from bonsai_forensics.gate1_forensics import (
    GGML_TYPE_PQ2_0,
    GGUFHeader,
    TensorInfo,
    parse_gguf_header,
    read_pq2_0,
)

SPEC_SHA256 = "0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b"

PRISM_PREFIX = "prism.hadamard."
TRANSFORM = "normalized-sylvester-walsh-hadamard"
AXIS = "input-last-dimension"
SIGN_MODE = "explicit"
SUPPORTED_BLOCKS = (512, 1024, 2048, 4096)
EXPECTED_BLOCK = rotation.TBR_N  # 1024 (spec §1.1)
EXPECTED_INVERSE = ("token_embd.weight",)

_REQUIRED_KEYS = (
    "prism.hadamard.block_size",
    "prism.hadamard.transform",
    "prism.hadamard.axis",
    "prism.hadamard.sign_mode",
    "prism.hadamard.sign_widths",
    "prism.hadamard.sign_values",
    "prism.hadamard.weight_names",
)


@dataclass
class PrismBasis:
    """The `prism.hadamard.*` block of a Bonsai-2 GGUF, validated."""

    version: int
    block_size: int
    transform: str
    axis: str
    sign_mode: str
    gdn_v_grouped: bool
    weight_names: tuple[str, ...]
    inverse_weight_names: tuple[str, ...]
    sign_widths: tuple[int, ...]
    sign_values: np.ndarray
    _cache: dict[int, list[np.ndarray]] = field(default_factory=dict, repr=False)

    @property
    def rotated_names(self) -> tuple[str, ...]:
        """The 401 forward-rotated tensors (Prism's `weight_names`)."""
        return self.weight_names

    @property
    def all_names(self) -> tuple[str, ...]:
        return self.weight_names + self.inverse_weight_names

    def signs(self, width: int) -> list[np.ndarray]:
        """Per-block ±1 signs for one width: `width / block_size` vectors."""
        width = int(width)
        if width not in self._cache:
            if width not in self.sign_widths:
                raise KeyError(f"width {width} is not in sign_widths {self.sign_widths}")
            offsets = np.cumsum((0,) + self.sign_widths)
            start = int(offsets[list(self.sign_widths).index(width)])
            flat = self.sign_values[start : start + width].astype(np.float64)
            g = self.block_size
            self._cache[width] = [
                flat[k * g : (k + 1) * g].copy() for k in range(width // g)
            ]
        return self._cache[width]

    @property
    def sign_sets(self) -> dict[str, list[np.ndarray]]:
        """`{str(width): per-block signs}` — the shape `rotation.resolve_rotations`
        and `load_sign_manifest` consume, built from the GGUF alone."""
        return {str(w): self.signs(w) for w in self.sign_widths}

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "block_size": self.block_size,
            "transform": self.transform,
            "axis": self.axis,
            "sign_mode": self.sign_mode,
            "gdn_v_grouped": self.gdn_v_grouped,
            "sign_widths": list(self.sign_widths),
            "forward_tensors": len(self.weight_names),
            "inverse_tensors": list(self.inverse_weight_names),
            "sign_values": int(self.sign_values.size),
        }


def _require_int(metadata: dict, key: str) -> int:
    if key not in metadata:
        raise ValueError(f"GGUF is missing {key!r}; not a self-describing Prism artifact")
    value = metadata[key]
    if isinstance(value, (list, tuple)):
        raise ValueError(f"{key} must be a scalar, got {type(value).__name__}")
    return int(value)


def parse_prism_basis(
    metadata: dict, *, strict: bool = True, expected_block: int = EXPECTED_BLOCK
) -> PrismBasis:
    """Validate a GGUF metadata dict against the frozen `tbr-1.2` contract.

    `strict=True` additionally requires the block size to equal the spec's
    `TBR_N` (1024). Prism's runtime accepts `{512,1024,2048,4096}`; ours is the
    frozen 1024 subset, so a non-1024 block is a different basis and is rejected.
    """
    missing = [key for key in _REQUIRED_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"GGUF lacks prism.hadamard metadata: {missing}")

    version = _require_int(metadata, "prism.hadamard.version") if "prism.hadamard.version" in metadata else 0
    block = _require_int(metadata, "prism.hadamard.block_size")
    transform = str(metadata["prism.hadamard.transform"])
    axis = str(metadata["prism.hadamard.axis"])
    sign_mode = str(metadata["prism.hadamard.sign_mode"])
    gdn_v_grouped = bool(metadata.get("prism.hadamard.gdn_v_grouped", False))

    if block not in SUPPORTED_BLOCKS:
        raise ValueError(f"unvalidated Hadamard block size {block}")
    if strict and block != expected_block:
        raise ValueError(
            f"block size {block} != frozen spec block {expected_block} (tbr-1.2); "
            "a different basis is a spec change, not a loader option"
        )
    if transform != TRANSFORM:
        raise ValueError(f"unsupported transform {transform!r}, expected {TRANSFORM!r}")
    if axis != AXIS:
        raise ValueError(f"unsupported axis {axis!r}, expected {AXIS!r}")
    if sign_mode != SIGN_MODE:
        raise ValueError(f"unsupported sign_mode {sign_mode!r}, expected {SIGN_MODE!r}")

    widths = tuple(int(w) for w in metadata["prism.hadamard.sign_widths"])
    if not widths:
        raise ValueError("sign_widths is empty")
    for width in widths:
        if width <= 0 or width % block:
            raise ValueError(f"sign width {width} is not a positive multiple of block {block}")

    values = np.asarray(metadata["prism.hadamard.sign_values"])
    if values.ndim != 1:
        raise ValueError(f"sign_values must be 1-D, got shape {values.shape}")
    if values.size != sum(widths):
        raise ValueError(
            f"sign_values has {values.size} entries, sign_widths sum to {sum(widths)}"
        )
    if not np.all(np.isin(values, (-1, 1))):
        raise ValueError("sign_values must be ±1")

    forward = tuple(str(name) for name in metadata["prism.hadamard.weight_names"])
    inverse = tuple(
        str(name) for name in metadata.get("prism.hadamard.inverse_weight_names", [])
    )
    if not forward:
        raise ValueError("weight_names is empty")
    overlap = set(forward) & set(inverse)
    if overlap:
        raise ValueError(f"forward and inverse manifests overlap: {sorted(overlap)}")

    return PrismBasis(
        version=version,
        block_size=block,
        transform=transform,
        axis=axis,
        sign_mode=sign_mode,
        gdn_v_grouped=gdn_v_grouped,
        weight_names=forward,
        inverse_weight_names=inverse,
        sign_widths=widths,
        sign_values=values.astype(np.int64),
    )


@dataclass
class PrismGGUF:
    """A parsed Bonsai-2 GGUF plus its validated basis and derived tensor widths."""

    header: GGUFHeader
    basis: PrismBasis
    name_widths: dict[str, int]
    inverse_names: frozenset[str]

    def width_for(self, name: str) -> int:
        if name not in self.name_widths:
            raise KeyError(f"{name!r} is not in the transform manifest")
        return self.name_widths[name]

    def is_inverse(self, name: str) -> bool:
        return name in self.inverse_names

    def dequant(self, name: str) -> np.ndarray:
        """Stored PQ2_0 tensor as float32 in `(out, in)` order (ggml reversed)."""
        info = self.header.tensors[name]
        if info.ggml_type != GGML_TYPE_PQ2_0:
            raise ValueError(f"{name} is ggml type {info.ggml_type}, not PQ2_0")
        codes, scales = read_pq2_0(self.header, name)
        return pq2_0.dequantize_pq2_0(codes, scales)

    def recover(self, name: str, dtype=np.float32) -> np.ndarray:
        """Unrotate one tensor to the original basis: `W_hf = D @ R`."""
        info = self.header.tensors[name]
        width = self.width_for(name)
        if info.shape[0] != width:
            raise ValueError(
                f"{name} input axis {info.shape[0]} != manifest width {width}"
            )
        rots = self.basis.signs(width)
        weights = rotation.apply_rotation(self.dequant(name), rots, transpose=True)
        return weights.astype(dtype)

    def iter_recover(self, dtype=np.float32) -> Iterator[tuple[str, np.ndarray]]:
        for name in self.basis.all_names:
            yield name, self.recover(name, dtype=dtype)

    def census(self) -> dict:
        from collections import Counter

        types = Counter(info.ggml_type for info in self.header.tensors.values())
        return {
            "spec_hash": SPEC_SHA256,
            "gguf": str(self.header.path),
            "tensor_count": len(self.header.tensors),
            "ggml_type_census": {str(k): v for k, v in sorted(types.items())},
            "basis": self.basis.as_dict(),
            "transformed_tensors": len(self.name_widths),
            "width_to_tensors": {
                str(width): sum(1 for value in self.name_widths.values() if value == width)
                for width in self.basis.sign_widths
            },
        }


def _validate_against_header(header: GGUFHeader, basis: PrismBasis) -> PrismGGUF:
    name_widths: dict[str, int] = {}
    for name in basis.all_names:
        info = header.tensors.get(name)
        if info is None:
            raise ValueError(f"transform manifest references missing tensor {name!r}")
        if info.ggml_type != GGML_TYPE_PQ2_0:
            raise ValueError(f"transformed tensor {name!r} is ggml type {info.ggml_type}, not PQ2_0")
        if info.shape[0] not in basis.sign_widths:
            raise ValueError(
                f"tensor {name!r} input axis {info.shape[0]} is not a declared sign width"
            )
        name_widths[name] = int(info.shape[0])
    return PrismGGUF(
        header=header,
        basis=basis,
        name_widths=name_widths,
        inverse_names=frozenset(basis.inverse_weight_names),
    )


def load_prism_gguf(path: str | Path, *, strict: bool = True) -> PrismGGUF:
    """Parse a Bonsai-2 PQ2_0 GGUF and validate its embedded basis."""
    header = parse_gguf_header(path)
    basis = parse_prism_basis(header.metadata, strict=strict)
    return _validate_against_header(header, basis)


def verify_against_manifest(
    basis: PrismBasis, manifest_path: str | Path
) -> dict[str, dict]:
    """Cross-check the embedded signs against a Prism `.npy` sign manifest.

    Returns `{str(width): {"match": bool, "n": int, "first_mismatch": int|None}}`
    plus an `ok` roll-up. A mismatch is reported, never silently accepted.
    """
    manifest = rotation.load_sign_manifest(manifest_path)
    report: dict[str, dict] = {}
    ok = True
    for width in basis.sign_widths:
        reference = np.concatenate(manifest[str(width)]) if str(width) in manifest else None
        embedded = np.concatenate(basis.signs(width))
        if reference is None or reference.shape != embedded.shape:
            report[str(width)] = {"match": False, "n": int(embedded.size), "first_mismatch": None}
            ok = False
            continue
        equal = np.array_equal(reference, embedded)
        first = None if equal else int(np.argmax(reference != embedded))
        report[str(width)] = {"match": bool(equal), "n": int(embedded.size), "first_mismatch": first}
        ok = ok and bool(equal)
    report["ok"] = {"match": ok, "n": int(basis.sign_values.size)}
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True, help="Bonsai-2 PQ2_0 GGUF")
    parser.add_argument("--verify-manifest", default=None, help="Prism hadamard-manifest.json")
    parser.add_argument("--recover", default=None, help="recover one tensor to <out>.npy")
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-strict", action="store_true", help="allow a non-1024 block")
    args = parser.parse_args(argv)

    model = load_prism_gguf(args.gguf, strict=not args.no_strict)
    if args.recover:
        weights = model.recover(args.recover)
        out = Path(args.out) if args.out else Path(f"{args.recover.replace('.', '_')}.npy")
        np.save(out, weights)
        print(json.dumps({"recovered": args.recover, "shape": list(weights.shape), "out": str(out)}))
        return 0

    report = model.census()
    if args.verify_manifest:
        report["manifest_check"] = verify_against_manifest(model.basis, args.verify_manifest)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
