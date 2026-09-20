"""T25 — PQ2_0 (g128) block codec: reader/packer for Prism's ternary format.

Layout verified in R2 (`research/pq2_0-layout.md`; T7's `oracle.decode_pq2_0`
is byte-exact against Prism's own F16 dequant of the released weights). A
PQ2_0 tensor row is a sequence of 34-byte blocks of 128 values, little-endian:

    bytes 0..1   fp16 scale `d`
    bytes 2..33  packed 2-bit codes `qs[32]`
    value i      = (((qs[i // 4] >> (2 * (i % 4))) & 3) - 1) * d

Code 3 decodes as `+2d` (legal on read, never emitted by the reference RTN
quantizer); `pack_pq2_0` refuses codes outside {-1, 0, +1}. The ggml type id is
142 (`QK_PQ2_0 = 128`); mainline llama.cpp has no kernels for it, so serving a
type-142 artifact is the Prism ROCm fork's job (ADR-2).

Contract note: this is the R2-verified implementation, not a guessed layout.
`bonsai_forensics/spec.md` §3.2 still marks PQ2_0 deferred because its
constants block is hash-frozen (`tbr-1.2`); formalizing it there changes the
canonical hash and re-pins every consumer, which is a QUEEN hotspot change
tracked as a T25 follow-up. No spec file is edited here, so `SPEC_SHA256`
below matches the frozen hash.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from bonsai_forensics import pack_gguf

SPEC_SHA256 = "0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b"

GGML_TYPE_PQ2_0 = 142
BLOCK_SIZE = 128
BLOCK_BYTES = 34
QS_OFFSET = 2
QS_BYTES = 32
BPW = BLOCK_BYTES * 8 / BLOCK_SIZE  # 2.125 bits per weight


def pq2_0_row_size(ne0: int) -> int:
    """Bytes for one row of `ne0` values; rows must be a multiple of 128."""
    if ne0 <= 0 or ne0 % BLOCK_SIZE:
        raise ValueError(f"PQ2_0 row length must be a positive multiple of 128, got {ne0}")
    return BLOCK_BYTES * (ne0 // BLOCK_SIZE)


def pq2_0_nbytes(shape: Sequence[int]) -> int:
    shape = tuple(int(s) for s in shape)
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError(f"invalid shape {shape}")
    rows = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
    return pq2_0_row_size(shape[0]) * rows


def pack_pq2_0(codes: np.ndarray, scales: np.ndarray) -> bytes:
    """Pack ternary codes/scales into PQ2_0 blocks (R2 layout).

    `codes`: int8 `(..., n)` with `n % 128 == 0`; `scales`: `(..., n // 128)`
    fp16-compatible. Byte-exact inverse of `unpack_pq2_0`."""
    codes = np.asarray(codes)
    scales = np.asarray(scales)
    if codes.dtype != np.int8:
        raise ValueError(f"codes must be int8, got {codes.dtype}")
    if codes.ndim < 1 or codes.shape[-1] % BLOCK_SIZE:
        raise ValueError(f"codes last axis must be a multiple of 128, got shape {codes.shape}")
    if scales.shape != codes.shape[:-1] + (codes.shape[-1] // BLOCK_SIZE,):
        raise ValueError(f"scales shape {scales.shape} does not match codes shape {codes.shape}")
    if np.any(codes < -1) or np.any(codes > 1):
        raise ValueError("ternary codes must be in {-1, 0, +1}")

    leading = codes.shape[:-1]
    n_blocks = codes.shape[-1] // BLOCK_SIZE
    values = codes.reshape(leading + (n_blocks, BLOCK_SIZE)).astype(np.int16)
    quantized = ((values + 1) & 3).reshape(leading + (n_blocks, QS_BYTES, 4))
    qs = np.zeros(leading + (n_blocks, QS_BYTES), dtype=np.uint8)
    for j in range(4):
        qs |= quantized[..., j].astype(np.uint8) << (2 * j)

    d = np.asarray(scales, dtype="<f2").reshape(leading + (n_blocks, 1)).view(np.uint8)
    blocks = np.concatenate([d, qs], axis=-1)
    return np.ascontiguousarray(blocks).tobytes()


def unpack_pq2_0(data: bytes, shape: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `pack_pq2_0` for a ggml-order `shape` (ne0 first).

    Returns `(codes, scales)` with `codes` int8 `(rows, ne0)` in {-1..2} (code
    3 survives as `+2`) and `scales` float32 `(rows, ne0 // 128)`."""
    shape = tuple(int(s) for s in shape)
    expected = pq2_0_nbytes(shape)
    if len(data) != expected:
        raise ValueError(f"PQ2_0 payload is {len(data)} bytes, expected {expected}")
    rows = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
    n_blocks = shape[0] // BLOCK_SIZE
    blocks = np.frombuffer(data, dtype=np.uint8).reshape(rows, n_blocks, BLOCK_BYTES)
    scales = blocks[:, :, :QS_OFFSET].copy().view("<f2").astype(np.float32).reshape(rows, n_blocks)
    qs = blocks[:, :, QS_OFFSET:].reshape(rows, n_blocks, QS_BYTES, 1)
    shifts = (2 * np.arange(4, dtype=np.uint8)).reshape(1, 1, 1, 4)
    quantized = ((qs >> shifts) & 3).astype(np.int8).reshape(rows, n_blocks, BLOCK_SIZE)
    codes = (quantized - 1).reshape(rows, n_blocks * BLOCK_SIZE)
    return codes, scales


def dequantize_pq2_0(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    expanded = np.repeat(scales, BLOCK_SIZE, axis=-1)
    return codes.astype(np.float32) * expanded[..., : codes.shape[-1]]


def decode_pq2_0(data: bytes, ne0: int, ne1: int) -> np.ndarray:
    """Decode a full 2-D tensor payload to float32 `(ne1, ne0)`."""
    codes, scales = unpack_pq2_0(data, (ne0, ne1))
    return dequantize_pq2_0(codes, scales)


class PQ2_0GGUFWriter(pack_gguf.GGUFWriter):
    """`GGUFWriter` that also knows ggml type 142 (PQ2_0).

    `pack_gguf.tensor_nbytes` predates the Prism type (mainline build 11030
    ships only TQ1_0/TQ2_0), so the base writer rejects it; this subclass
    supplies the PQ2_0 row math without modifying the hotspot module."""

    def add_tensor(self, name: str, shape: Sequence[int], ggml_type: int, data: bytes) -> None:
        if int(ggml_type) != GGML_TYPE_PQ2_0:
            super().add_tensor(name, shape, ggml_type, data)
            return
        shape = tuple(int(s) for s in shape)
        if any(existing[0] == name for existing in self._tensors):
            raise ValueError(f"duplicate tensor {name!r}")
        expected = pq2_0_nbytes(shape)
        data = bytes(data)
        if len(data) != expected:
            raise ValueError(f"tensor {name!r}: {len(data)} bytes, expected {expected}")
        self._tensors.append((name, shape, GGML_TYPE_PQ2_0, data))
