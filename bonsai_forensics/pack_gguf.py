"""T6 — GGUF v3 writer/reader and byte-exact `TQ2_0` blocks (spec §3.1).

Shapes follow ggml convention: `shape[0]` is `ne[0]`, the fastest-varying
(row) axis. A torch weight `(out, in)` therefore has `shape = (in, out)` and
its 2-D codes array is `(out, in)` — the two orders meet in `pack_tq2_0`.
`dequantize_*` returns numpy arrays in torch order (`tuple(reversed(shape))`).

TQ2_0 (ggml type 35): 256 values per block, 66 bytes = `qs[64] || fp16 d`,
little-endian. Packing order is frozen by spec §3.1 and cross-checked in tests
against llama.cpp's `quantize_row_tq2_0_ref` / `dequantize_row_tq2_0` loaded
from `libggml-base.so` via ctypes, plus gguf-py round-trips.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

SPEC_SHA256 = "0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b"

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_TQ1_0 = 34
GGML_TYPE_TQ2_0 = 35
GGML_TYPE_NAMES = {
    GGML_TYPE_F32: "F32",
    GGML_TYPE_F16: "F16",
    GGML_TYPE_TQ1_0: "TQ1_0",
    GGML_TYPE_TQ2_0: "TQ2_0",
}
GGML_TYPE_SIZES = {GGML_TYPE_F32: 4, GGML_TYPE_F16: 2}
GGML_TQ_BLOCK = {GGML_TYPE_TQ1_0: (256, 54), GGML_TYPE_TQ2_0: (256, 66)}

TQ2_0_BLOCK = 256
TQ2_0_BLOCK_BYTES = 66
TQ2_0_QS_BYTES = 64

# GGUF metadata value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12


def tq2_0_row_size(ne0: int) -> int:
    if ne0 <= 0 or ne0 % TQ2_0_BLOCK:
        raise ValueError(f"TQ2_0 row length must be a positive multiple of 256, got {ne0}")
    return TQ2_0_BLOCK_BYTES * (ne0 // TQ2_0_BLOCK)


def tensor_nbytes(ggml_type: int, shape: Sequence[int]) -> int:
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError(f"invalid shape {tuple(shape)}")
    numel = int(np.prod(shape, dtype=np.int64))
    if ggml_type in GGML_TYPE_SIZES:
        return numel * GGML_TYPE_SIZES[ggml_type]
    if ggml_type in GGML_TQ_BLOCK:
        block, block_bytes = GGML_TQ_BLOCK[ggml_type]
        if shape[0] % block:
            raise ValueError(f"{GGML_TYPE_NAMES[ggml_type]} row length must be a multiple of {block}, got {shape[0]}")
        return block_bytes * numel // block
    raise ValueError(f"unsupported ggml type {ggml_type}")


def pack_tq2_0(codes: np.ndarray, scales: np.ndarray) -> bytes:
    """Pack ternary codes/scales into `TQ2_0` blocks (spec §3.1).

    `codes`: int8 `(..., n)` with `n % 256 == 0`; `scales`: `(..., n // 256)`.
    """
    codes = np.asarray(codes)
    scales = np.asarray(scales)
    if codes.dtype != np.int8:
        raise ValueError(f"codes must be int8, got {codes.dtype}")
    if codes.ndim < 1 or codes.shape[-1] % TQ2_0_BLOCK:
        raise ValueError(f"codes last axis must be a multiple of 256, got shape {codes.shape}")
    if scales.shape != codes.shape[:-1] + (codes.shape[-1] // TQ2_0_BLOCK,):
        raise ValueError(f"scales shape {scales.shape} does not match codes shape {codes.shape}")
    if np.any(codes < -1) or np.any(codes > 1):
        raise ValueError("ternary codes must be in {-1, 0, +1}")

    n_blocks = codes.shape[-1] // TQ2_0_BLOCK
    blocks = codes.reshape(codes.shape[:-1] + (n_blocks, TQ2_0_BLOCK)).astype(np.int16)
    quantized = (blocks + 1) & 3
    quantized = quantized.reshape(codes.shape[:-1] + (n_blocks, 2, 4, 32))
    qs = np.zeros(codes.shape[:-1] + (n_blocks, 2, 32), dtype=np.uint8)
    for n in range(4):
        qs |= quantized[..., :, n, :].astype(np.uint8) << (2 * n)

    d = np.asarray(scales, dtype="<f2").reshape(codes.shape[:-1] + (n_blocks, 1))
    block_bytes = np.concatenate(
        [qs.reshape(codes.shape[:-1] + (n_blocks, TQ2_0_QS_BYTES)), d.view(np.uint8)], axis=-1
    )
    return np.ascontiguousarray(block_bytes).tobytes()


def unpack_tq2_0(data: bytes, shape: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `pack_tq2_0` for a ggml-order `shape` (ne0 first)."""
    shape = tuple(int(s) for s in shape)
    expected = tensor_nbytes(GGML_TYPE_TQ2_0, shape)
    if len(data) != expected:
        raise ValueError(f"TQ2_0 payload is {len(data)} bytes, expected {expected}")
    rows = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
    n_blocks = shape[0] // TQ2_0_BLOCK
    blocks = np.frombuffer(data, dtype=np.uint8).reshape(rows, n_blocks, TQ2_0_BLOCK_BYTES)
    qs = blocks[:, :, :TQ2_0_QS_BYTES].reshape(rows, n_blocks, 2, 32)
    scales = blocks[:, :, TQ2_0_QS_BYTES:].copy().view("<f2").reshape(rows, n_blocks).astype(np.float32)
    quantized = np.empty((rows, n_blocks, 2, 4, 32), dtype=np.int8)
    for n in range(4):
        quantized[..., :, n, :] = ((qs >> (2 * n)) & 3).astype(np.int8) - 1
    codes = quantized.reshape(rows, n_blocks * TQ2_0_BLOCK)
    return codes, scales


def dequantize_tq2_0(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    expanded = np.repeat(scales, TQ2_0_BLOCK, axis=-1)
    return codes.astype(np.float32) * expanded[..., : codes.shape[-1]]


_SCALAR_FORMATS = {
    GGUF_TYPE_UINT8: "<B",
    GGUF_TYPE_INT8: "<b",
    GGUF_TYPE_UINT16: "<H",
    GGUF_TYPE_INT16: "<h",
    GGUF_TYPE_UINT32: "<I",
    GGUF_TYPE_INT32: "<i",
    GGUF_TYPE_FLOAT32: "<f",
    GGUF_TYPE_BOOL: "<B",
    GGUF_TYPE_UINT64: "<Q",
    GGUF_TYPE_INT64: "<q",
    GGUF_TYPE_FLOAT64: "<d",
}


def _pack_scalar(value, value_type: int) -> bytes:
    fmt = _SCALAR_FORMATS[value_type]
    if value_type == GGUF_TYPE_BOOL:
        return struct.pack(fmt, 1 if value else 0)
    if value_type in (GGUF_TYPE_FLOAT32, GGUF_TYPE_FLOAT64):
        return struct.pack(fmt, float(value))
    return struct.pack(fmt, int(value))


def _infer_type(value) -> int:
    if isinstance(value, bool):
        return GGUF_TYPE_BOOL
    if isinstance(value, (int, np.integer)):
        value = int(value)
        if value >= 0:
            return GGUF_TYPE_UINT32 if value <= 0xFFFFFFFF else GGUF_TYPE_UINT64
        return GGUF_TYPE_INT32 if value >= -(2**31) else GGUF_TYPE_INT64
    if isinstance(value, (float, np.floating)):
        return GGUF_TYPE_FLOAT32
    if isinstance(value, str):
        return GGUF_TYPE_STRING
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return GGUF_TYPE_ARRAY
    raise TypeError(f"unsupported GGUF metadata value: {type(value)!r}")


def _encode_value(value, value_type: int | None = None, element_type: int | None = None) -> tuple[int, bytes]:
    if value_type is None:
        value_type = _infer_type(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if value_type == GGUF_TYPE_STRING:
        raw = value.encode("utf-8")
        return value_type, struct.pack("<Q", len(raw)) + raw
    if value_type == GGUF_TYPE_ARRAY:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"array metadata expects a list, got {type(value)!r}")
        if not value:
            return value_type, struct.pack("<IQ", GGUF_TYPE_UINT8, 0)
        if element_type is None:
            element_type = _infer_type(value[0])
        body = b"".join(_encode_value(v, element_type)[1] for v in value)
        return value_type, struct.pack("<IQ", element_type, len(value)) + body
    if value_type not in _SCALAR_FORMATS:
        raise TypeError(f"unsupported GGUF metadata type {value_type}")
    return value_type, _pack_scalar(value, value_type)


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    offset: int
    nbytes: int

    @property
    def type_name(self) -> str:
        return GGML_TYPE_NAMES.get(self.ggml_type, str(self.ggml_type))


class GGUFWriter:
    def __init__(self, alignment: int = GGUF_DEFAULT_ALIGNMENT) -> None:
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("alignment must be a positive power of two")
        self.alignment = alignment
        self.metadata: dict[str, object] = {}
        self._metadata_types: dict[str, tuple[int | None, int | None]] = {}
        self._tensors: list[tuple[str, tuple[int, ...], int, bytes]] = []

    def add_metadata(self, key: str, value, value_type: int | None = None, element_type: int | None = None) -> None:
        _encode_value(value, value_type, element_type)
        self.metadata[key] = value
        self._metadata_types[key] = (value_type, element_type)

    def add_tensor(self, name: str, shape: Sequence[int], ggml_type: int, data: bytes) -> None:
        shape = tuple(int(s) for s in shape)
        if any(existing[0] == name for existing in self._tensors):
            raise ValueError(f"duplicate tensor {name!r}")
        expected = tensor_nbytes(ggml_type, shape)
        data = bytes(data)
        if len(data) != expected:
            raise ValueError(f"tensor {name!r}: {len(data)} bytes, expected {expected}")
        self._tensors.append((name, shape, int(ggml_type), data))

    @property
    def tensor_names(self) -> list[str]:
        return [t[0] for t in self._tensors]

    def to_bytes(self) -> bytes:
        metadata = dict(self.metadata)
        if self.alignment != GGUF_DEFAULT_ALIGNMENT:
            metadata.setdefault("general.alignment", self.alignment)
        out = io.BytesIO()
        out.write(GGUF_MAGIC)
        out.write(struct.pack("<IQQ", GGUF_VERSION, len(self._tensors), len(metadata)))
        for key, value in metadata.items():
            out.write(_encode_value(key)[1])
            override, element_type = self._metadata_types.get(key, (None, None))
            value_type, payload = _encode_value(value, override, element_type)
            out.write(struct.pack("<I", value_type))
            out.write(payload)

        offsets: list[int] = []
        position = 0
        for _, shape, ggml_type, data in self._tensors:
            offsets.append(position)
            position = _align(position + len(data), self.alignment)
        for (name, shape, ggml_type, _), offset in zip(self._tensors, offsets):
            out.write(_encode_value(name)[1])
            out.write(struct.pack("<I", len(shape)))
            for dim in shape:
                out.write(struct.pack("<Q", dim))
            out.write(struct.pack("<IQ", ggml_type, offset))
        out.write(b"\x00" * (_align(out.tell(), self.alignment) - out.tell()))
        data_start = out.tell()
        for _, _, _, data in self._tensors:
            out.write(data)
            out.write(b"\x00" * (_align(out.tell() - data_start, self.alignment) - (out.tell() - data_start)))
        return out.getvalue()

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_bytes(self.to_bytes())
        return path


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class GGUFReader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._blob = self.path.read_bytes()
        self.metadata: dict[str, object] = {}
        self.tensors: dict[str, TensorInfo] = {}
        self._parse()

    def _parse(self) -> None:
        blob = self._blob
        if len(blob) < 24 or blob[:4] != GGUF_MAGIC:
            raise ValueError("not a GGUF file")
        version, n_tensors, n_kv = struct.unpack_from("<IQQ", blob, 4)
        if version != GGUF_VERSION:
            raise ValueError(f"unsupported GGUF version {version}")
        stream = io.BytesIO(blob)
        stream.seek(24)
        for _ in range(n_kv):
            key = self._read_string(stream)
            self.metadata[key] = self._read_value(stream)
        infos: list[TensorInfo] = []
        for _ in range(n_tensors):
            name = self._read_string(stream)
            (n_dims,) = struct.unpack("<I", stream.read(4))
            shape = tuple(struct.unpack("<Q", stream.read(8))[0] for _ in range(n_dims))
            ggml_type, offset = struct.unpack("<IQ", stream.read(12))
            infos.append(TensorInfo(name, shape, ggml_type, offset, tensor_nbytes(ggml_type, shape)))
        alignment = int(self.metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT))
        self.tensor_data_offset = _align(stream.tell(), alignment)
        for info in infos:
            if info.offset % alignment:
                raise ValueError(f"tensor {info.name!r} offset violates alignment")
            self.tensors[info.name] = info

    @staticmethod
    def _read_string(stream: io.BytesIO) -> str:
        (length,) = struct.unpack("<Q", stream.read(8))
        return stream.read(length).decode("utf-8")

    @classmethod
    def _read_value(cls, stream: io.BytesIO, value_type: int | None = None):
        if value_type is None:
            (value_type,) = struct.unpack("<I", stream.read(4))
        if value_type == GGUF_TYPE_ARRAY:
            element_type, count = struct.unpack("<IQ", stream.read(12))
            return [cls._read_value(stream, element_type) for _ in range(count)]
        if value_type == GGUF_TYPE_STRING:
            return cls._read_string(stream)
        fmt, size = {
            GGUF_TYPE_UINT8: ("<B", 1),
            GGUF_TYPE_INT8: ("<b", 1),
            GGUF_TYPE_UINT16: ("<H", 2),
            GGUF_TYPE_INT16: ("<h", 2),
            GGUF_TYPE_UINT32: ("<I", 4),
            GGUF_TYPE_INT32: ("<i", 4),
            GGUF_TYPE_FLOAT32: ("<f", 4),
            GGUF_TYPE_BOOL: ("<B", 1),
            GGUF_TYPE_UINT64: ("<Q", 8),
            GGUF_TYPE_INT64: ("<q", 8),
            GGUF_TYPE_FLOAT64: ("<d", 8),
        }[value_type]
        raw = stream.read(size)
        if len(raw) != size:
            raise ValueError("truncated GGUF file")
        value = struct.unpack(fmt, raw)[0]
        if value_type == GGUF_TYPE_BOOL:
            return bool(value)
        return value

    def tensor_bytes(self, name: str) -> bytes:
        info = self.tensors[name]
        start = self.tensor_data_offset + info.offset
        return self._blob[start : start + info.nbytes]

    def dequantize(self, name: str) -> np.ndarray:
        info = self.tensors[name]
        raw = self.tensor_bytes(name)
        if info.ggml_type == GGML_TYPE_F32:
            flat = np.frombuffer(raw, dtype="<f4")
        elif info.ggml_type == GGML_TYPE_F16:
            flat = np.frombuffer(raw, dtype="<f2").astype(np.float32)
        elif info.ggml_type == GGML_TYPE_TQ2_0:
            rows = int(np.prod(info.shape[1:], dtype=np.int64)) if len(info.shape) > 1 else 1
            n_blocks = info.shape[0] // TQ2_0_BLOCK
            codes, scales = unpack_tq2_0(raw, info.shape)
            dequantized = dequantize_tq2_0(codes, scales).reshape(rows, n_blocks * TQ2_0_BLOCK)
            return dequantized.reshape(tuple(reversed(info.shape)))
        else:
            raise ValueError(f"unsupported ggml type {info.ggml_type} for {name!r}")
        return flat.reshape(tuple(reversed(info.shape)))


def pack_f16(tensor, shape: Sequence[int] | None = None) -> bytes:
    array = np.asarray(tensor, dtype="<f2")
    if shape is not None and tuple(array.shape) != tuple(reversed(tuple(shape))):
        raise ValueError(f"tensor shape {array.shape} does not match ggml shape {tuple(shape)}")
    return np.ascontiguousarray(array).tobytes()


def pack_f32(tensor, shape: Sequence[int] | None = None) -> bytes:
    array = np.asarray(tensor, dtype="<f4")
    if shape is not None and tuple(array.shape) != tuple(reversed(tuple(shape))):
        raise ValueError(f"tensor shape {array.shape} does not match ggml shape {tuple(shape)}")
    return np.ascontiguousarray(array).tobytes()


def pack_tq2_0_tensor(w: np.ndarray, group_size: int = TQ2_0_BLOCK, refine_iters: int | None = None) -> bytes:
    """Quantize a torch-order `(out, in)` tensor with T3 and pack it."""
    from bonsai_forensics.quant import REFINE_ITERS, quantize

    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError("expected a 2-D weight in torch order (out, in)")
    if group_size != TQ2_0_BLOCK:
        raise ValueError("TQ2_0 requires group_size 256")
    quantized = quantize(w, group_size=group_size, refine_iters=REFINE_ITERS if refine_iters is None else refine_iters)
    return pack_tq2_0(quantized.codes, quantized.scales)


def artifact_name(calib: str, git: str, date: str) -> str:
    """`tbr27b-<calib>-<git>-<date>.gguf` (spec §5)."""
    if calib.lower() not in {"a", "b", "c"}:
        raise ValueError("calib must be one of a/b/c")
    if not git or not date:
        raise ValueError("git and date are required")
    return f"tbr27b-{calib.lower()}-{git}-{date}.gguf"


def sha256_file(path: str | Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
