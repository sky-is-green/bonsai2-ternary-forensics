"""T7 — oracle: read Prism's released ternary weights and measure what they did.

We have both sides of their quantizer for the 1.7B: the F16 reference GGUF and
the ternary GGUF (PQ2_0, ggml type 142 / Q2_0 g64, type 42). Comparing the two
at group level answers the question benchmark numbers cannot: *is a public
algorithm (RTN, GPTQ) reproducing their trit assignment, or is their method
something else?*

This module is deliberately self-contained:
- a minimal GGUF table parser (metadata + tensor infos + raw data offsets) so
  type 142 needs no fork;
- the PQ2_0 decoder from R2 (`d` fp16 at offset 0, `qs[32]` at 2..33;
  code = `(qs[i//4] >> (2*(i%4))) & 3`, value = `(code-1)*d`);
- agreement metrics against the F16 base tensor.

Agreement is measured on absmean-RTN trits (`round(w / mean|w|)`) per g128
group -- the simplest public assigner. High agreement -> a public algorithm can
reproduce their weights; low agreement -> the gap is the algorithm (R6 finding).

Usage::

    python -m bonsai_forensics.oracle --base F16.gguf --ternary PQ2_0.gguf \
        --tensor blk.0.attn_q.weight
    python -m bonsai_forensics.oracle --base F16.gguf --ternary PQ2_0.gguf --scan 10
"""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPEC_SHA256 = "0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b"

GGML_F32 = 0
GGML_F16 = 1
GGML_Q2_0_G64 = 42
GGML_PQ2_0 = 142
GROUP = 128

# ggml scalar/array KV type ids
_KV_UINT8, _KV_INT8, _KV_UINT16, _KV_INT16 = 0, 1, 2, 3
_KV_UINT32, _KV_INT32, _KV_FLOAT32, _KV_BOOL = 4, 5, 6, 7
_KV_STRING, _KV_ARRAY, _KV_UINT64, _KV_INT64, _KV_FLOAT64 = 8, 9, 10, 11, 12


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    offset: int


@dataclass
class GGUFTable:
    path: Path
    metadata: dict
    tensors: dict[str, TensorInfo]
    data_offset: int


class _Cursor:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def u32(self) -> int:
        value = struct.unpack_from("<I", self.data, self.pos)[0]
        self.pos += 4
        return value

    def u64(self) -> int:
        value = struct.unpack_from("<Q", self.data, self.pos)[0]
        self.pos += 8
        return value

    def string(self) -> str:
        length = self.u64()
        value = self.data[self.pos : self.pos + length].decode("utf-8", errors="replace")
        self.pos += length
        return value

    def scalar(self, value_type: int):
        fmt = {
            _KV_UINT8: "<B", _KV_INT8: "<b", _KV_UINT16: "<H", _KV_INT16: "<h",
            _KV_UINT32: "<I", _KV_INT32: "<i", _KV_FLOAT32: "<f", _KV_BOOL: "<?",
            _KV_UINT64: "<Q", _KV_INT64: "<q", _KV_FLOAT64: "<d",
        }[value_type]
        size = struct.calcsize(fmt)
        value = struct.unpack_from(fmt, self.data, self.pos)[0]
        self.pos += size
        return value

    def value(self, value_type: int):
        if value_type == _KV_STRING:
            return self.string()
        if value_type == _KV_ARRAY:
            element_type = self.u32()
            count = self.u64()
            return [self.value(element_type) for _ in range(count)]
        return self.scalar(value_type)


def parse_gguf_table(path: str | Path) -> GGUFTable:
    """Minimal GGUF table parser: metadata, tensor infos, data offset."""
    path = Path(path)
    raw = path.read_bytes()
    cursor = _Cursor(raw)
    magic = raw[:4]
    if magic != b"GGUF":
        raise ValueError(f"{path} is not a GGUF file")
    cursor.pos = 4
    cursor.u32()  # version
    tensor_count = cursor.u64()
    kv_count = cursor.u64()
    metadata: dict = {}
    for _ in range(kv_count):
        key = cursor.string()
        value_type = cursor.u32()
        metadata[key] = cursor.value(value_type)
    tensors: dict[str, TensorInfo] = {}
    for _ in range(tensor_count):
        name = cursor.string()
        n_dims = cursor.u32()
        shape = tuple(cursor.u64() for _ in range(n_dims))
        ggml_type = cursor.u32()
        offset = cursor.u64()
        tensors[name] = TensorInfo(name=name, shape=shape, ggml_type=ggml_type, offset=offset)
    alignment = int(metadata.get("general.alignment", 32))
    data_offset = (cursor.pos + alignment - 1) // alignment * alignment
    return GGUFTable(path=path, metadata=metadata, tensors=tensors, data_offset=data_offset)


# ---------------------------------------------------------------------------
# Tensor decoding
# ---------------------------------------------------------------------------
def _row_size(ggml_type: int, shape: tuple[int, ...]) -> int:
    ne0 = shape[0]
    if ggml_type == GGML_F32:
        return ne0 * 4
    if ggml_type == GGML_F16:
        return ne0 * 2
    if ggml_type == GGML_PQ2_0:
        return (ne0 // GROUP) * 34
    if ggml_type == GGML_Q2_0_G64:
        return (ne0 // 64) * 18  # 64 values: 16 qs bytes + fp16 d
    raise ValueError(f"unsupported ggml type {ggml_type}")


def read_tensor(table: GGUFTable, name: str) -> np.ndarray:
    """Decode a 2-D tensor; returns float32 in ggml row-major (ne0 = last axis)."""
    info = table.tensors[name]
    if len(info.shape) != 2:
        raise ValueError(f"expected 2-D tensor, got shape {info.shape}")
    ne0, ne1 = info.shape
    row_bytes = _row_size(info.ggml_type, info.shape)
    raw = table.path.read_bytes()
    start = table.data_offset + info.offset
    block = raw[start : start + row_bytes * ne1]
    if len(block) != row_bytes * ne1:
        raise ValueError(f"truncated tensor {name}")
    if info.ggml_type == GGML_F16:
        return np.frombuffer(block, dtype="<f2").reshape(ne1, ne0).astype(np.float32)
    if info.ggml_type == GGML_F32:
        return np.frombuffer(block, dtype="<f4").reshape(ne1, ne0).astype(np.float32)
    if info.ggml_type == GGML_PQ2_0:
        return decode_pq2_0(block, ne0, ne1)
    if info.ggml_type == GGML_Q2_0_G64:
        return decode_q2_0_g64(block, ne0, ne1)
    raise ValueError(f"unsupported type {info.ggml_type}")


def decode_pq2_0(block: bytes, ne0: int, ne1: int) -> np.ndarray:
    """R2 layout: 34-byte g128 block, fp16 d at 0, qs[32] at 2..33."""
    blocks = np.frombuffer(block, dtype=np.uint8).reshape(ne1, ne0 // GROUP, 34)
    scale = blocks[:, :, :2].copy().view("<f2").astype(np.float32).reshape(ne1, ne0 // GROUP)
    qs = blocks[:, :, 2:34]
    out = np.empty((ne1, ne0 // GROUP, GROUP), dtype=np.float32)
    for i in range(GROUP):
        code = ((qs[:, :, i // 4] >> (2 * (i % 4))) & 3).astype(np.int16) - 1
        out[:, :, i] = code * scale
    return out.reshape(ne1, ne0)


def decode_q2_0_g64(block: bytes, ne0: int, ne1: int) -> np.ndarray:
    """Mainline Q2_0: 18-byte g64 block, fp16 d at 16..17, qs[16] at 0..15."""
    blocks = np.frombuffer(block, dtype=np.uint8).reshape(ne1, ne0 // 64, 18)
    scale = blocks[:, :, 16:18].copy().view("<f2").astype(np.float32).reshape(ne1, ne0 // 64)
    qs = blocks[:, :, :16]
    out = np.empty((ne1, ne0 // 64, 64), dtype=np.float32)
    for i in range(64):
        code = ((qs[:, :, i // 4] >> (2 * (i % 4))) & 3).astype(np.int16) - 1
        out[:, :, i] = code * scale
    return out.reshape(ne1, ne0)


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------
def absmean_trits(w: np.ndarray, group: int = GROUP) -> np.ndarray:
    """`round(w / mean|w|)` clipped to {-1,0,1} per group (last axis)."""
    out = np.empty_like(w, dtype=np.int8)
    for start in range(0, w.shape[-1], group):
        block = w[:, start : start + group]
        scale = np.abs(block).mean(axis=-1, keepdims=True)
        scale[scale == 0] = 1.0
        out[:, start : start + group] = np.clip(np.round(block / scale), -1, 1).astype(np.int8)
    return out


def group_scales(w: np.ndarray, codes: np.ndarray, group: int = GROUP) -> np.ndarray:
    """Least-squares scale per group for given codes: (codes·w)/(codes·codes)."""
    rows = w.shape[0]
    n_groups = w.shape[1] // group
    out = np.empty((rows, n_groups), dtype=np.float32)
    for g in range(n_groups):
        block = w[:, g * group : (g + 1) * group]
        c = codes[:, g * group : (g + 1) * group].astype(np.float32)
        denom = (c * c).sum(axis=-1)
        num = (c * block).sum(axis=-1)
        out[:, g] = np.where(denom > 0, num / np.maximum(denom, 1e-12), 0.0)
    return out


def compare(base: np.ndarray, released: np.ndarray, group: int = GROUP) -> dict:
    """Agreement between our public-assigner trits and their released weights."""
    group = min(group, base.shape[-1])
    reference = absmean_trits(base, group)
    # infer their trits from the released dequantized weights and their group scales
    released_group = released.reshape(base.shape[0], -1, group)
    scale = np.abs(released_group).max(axis=-1, keepdims=True)
    scale[scale == 0] = 1.0
    theirs = np.clip(np.round(released_group / scale), -1, 1).astype(np.int8).reshape(base.shape)
    weights = reference.reshape(base.shape[0], -1, group)
    match = (theirs.reshape(base.shape[0], -1, group) == weights).mean().item()
    base_norm = float(np.linalg.norm(base))
    released_norm = float(np.linalg.norm(released))
    diff = base - released
    return {
        "shape": list(base.shape),
        "sign_agreement_absmean": round(match, 4),
        "relative_weight_error": round(float(np.linalg.norm(diff)) / max(base_norm, 1e-12), 4),
        "base_norm": round(base_norm, 4),
        "released_norm": round(released_norm, 4),
        "released_over_base_norm": round(released_norm / max(base_norm, 1e-12), 4),
    }


def scan(base_path: str | Path, ternary_path: str | Path, limit: int | None = None) -> dict:
    base_table = parse_gguf_table(base_path)
    ternary_table = parse_gguf_table(ternary_path)
    shared = [n for n in ternary_table.tensors if n in base_table.tensors]
    results: dict[str, dict] = {}
    for name in shared[: limit or len(shared)]:
        base = read_tensor(base_table, name)
        released = read_tensor(ternary_table, name)
        if base.shape != released.shape:
            continue
        try:
            results[name] = compare(base, released)
        except Exception as error:  # pragma: no cover - forensic resilience
            results[name] = {"error": str(error)}
    agreements = [v["sign_agreement_absmean"] for v in results.values() if "sign_agreement_absmean" in v]
    return {
        "base": str(base_path),
        "ternary": str(ternary_path),
        "tensors_compared": len(results),
        "mean_sign_agreement": round(float(np.mean(agreements)), 4) if agreements else None,
        "per_tensor": results,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--ternary", required=True)
    parser.add_argument("--tensor", default="")
    parser.add_argument("--scan", type=int, default=0, help="compare the first N tensors")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    if args.tensor:
        base_table = parse_gguf_table(args.base)
        ternary_table = parse_gguf_table(args.ternary)
        base = read_tensor(base_table, args.tensor)
        released = read_tensor(ternary_table, args.tensor)
        result = {"tensor": args.tensor, **compare(base, released)}
    else:
        result = scan(args.base, args.ternary, args.scan or None)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
