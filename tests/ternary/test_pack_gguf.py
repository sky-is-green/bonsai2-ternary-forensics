"""T6 — GGUF/TQ2_0 acceptance: byte layout matches ggml, tiny model loads in
llama-server (live-gated), cross-validated against gguf-py and libggml.

No GPU, no downloads: the live server test is skipped unless `TBR_LIVE_GGUF=1`
and the local ROCm build exists (R1 is QUEEN's spike; this test only asserts
the format).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import json
import re
import socket
import struct
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from bonsai_forensics import pack_gguf as pg

SPEC_PATH = Path(__file__).resolve().parents[2] / "bonsai_forensics" / "spec.md"
SPEC_SHA256 = "0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b"

LLAMA_SERVER = Path.home() / ".unsloth" / "llama.cpp" / "build" / "bin" / "llama-server"
GGML_LIB_CANDIDATES = (
    Path.home() / ".unsloth" / "llama.cpp" / "build" / "bin" / "libggml-base.so",
    Path.home() / ".unsloth" / "llama.cpp" / "build" / "bin" / "libggml.so",
)


def _spec_constants() -> dict:
    text = SPEC_PATH.read_text(encoding="utf-8")
    return json.loads(re.findall(r"```json\n(.*?)\n```", text, re.S)[0])


def _load_ggml_quant_functions():
    for candidate in GGML_LIB_CANDIDATES:
        if not candidate.is_file():
            continue
        lib = ctypes.CDLL(str(candidate))
        if not (hasattr(lib, "dequantize_row_tq2_0") and hasattr(lib, "quantize_row_tq2_0_ref")):
            continue
        lib.dequantize_row_tq2_0.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        lib.quantize_row_tq2_0_ref.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        return lib
    return None


GGML = _load_ggml_quant_functions()
requires_ggml = pytest.mark.skipif(GGML is None, reason="libggml TQ2_0 symbols unavailable")


def test_spec_hash_is_pinned() -> None:
    canon = json.dumps(_spec_constants(), sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canon.encode()).hexdigest() == SPEC_SHA256 == pg.SPEC_SHA256


def test_tq2_0_constants_match_spec() -> None:
    spec = _spec_constants()["tq2_0"]
    assert spec["type_id"] == pg.GGML_TYPE_TQ2_0 == 35
    assert spec["block_size"] == pg.TQ2_0_BLOCK == 256
    assert spec["block_bytes"] == pg.TQ2_0_BLOCK_BYTES == 66
    assert spec["qs_bytes"] == pg.TQ2_0_QS_BYTES == 64
    assert spec["bpw"] == 2.0625


def test_row_size_and_nbytes_arithmetic() -> None:
    assert pg.tq2_0_row_size(256) == 66
    assert pg.tq2_0_row_size(512) == 132
    assert pg.tensor_nbytes(pg.GGML_TYPE_TQ2_0, (256, 2)) == 132
    assert pg.tensor_nbytes(pg.GGML_TYPE_F32, (4,)) == 16
    assert pg.tensor_nbytes(pg.GGML_TYPE_F16, (4, 3)) == 24
    for bad in (0, 128, 255):
        with pytest.raises(ValueError):
            pg.tq2_0_row_size(bad)
    with pytest.raises(ValueError):
        pg.tensor_nbytes(pg.GGML_TYPE_TQ2_0, (128,))
    with pytest.raises(ValueError):
        pg.tensor_nbytes(99, (4,))


def test_hand_computed_block_layout() -> None:
    codes = np.full((1, 256), -1, dtype=np.int8)
    codes[0, 0] = codes[0, 32] = codes[0, 64] = codes[0, 96] = 1
    payload = pg.pack_tq2_0(codes, np.array([[1.0]], dtype=np.float32))
    assert len(payload) == 66
    assert payload[0] == 0xAA
    assert payload[1:64] == b"\x00" * 63
    assert payload[64:66] == struct.pack("<e", 1.0)


def test_pack_unpack_round_trip() -> None:
    rng = np.random.default_rng(0)
    codes = rng.integers(-1, 2, size=(3, 512)).astype(np.int8)
    scales = (rng.random((3, 2)) + 0.1).astype(np.float32)
    payload = pg.pack_tq2_0(codes, scales)
    assert len(payload) == 66 * 2 * 3
    got_codes, got_scales = pg.unpack_tq2_0(payload, (512, 3))
    assert np.array_equal(got_codes, codes)
    assert np.array_equal(got_scales, scales.astype(np.float16).astype(np.float32))
    assert np.array_equal(
        pg.dequantize_tq2_0(got_codes, got_scales), codes * np.repeat(got_scales, 256, axis=-1)
    )


def test_pack_rejects_malformed_input() -> None:
    codes = np.zeros((1, 256), dtype=np.int8)
    with pytest.raises(ValueError):
        pg.pack_tq2_0(codes.astype(np.int16), np.array([[1.0]], np.float32))
    with pytest.raises(ValueError):
        pg.pack_tq2_0(codes, np.array([[1.0], [2.0]], np.float32))
    with pytest.raises(ValueError):
        pg.pack_tq2_0(codes[:, :128], np.array([[1.0]], np.float32))
    bad = codes.copy()
    bad[0, 0] = 2
    with pytest.raises(ValueError):
        pg.pack_tq2_0(bad, np.array([[1.0]], np.float32))
    with pytest.raises(ValueError):
        pg.unpack_tq2_0(b"\x00" * 65, (256,))


def _round_trip_file(tmp_path: Path, alignment: int = 32) -> tuple[Path, np.ndarray, bytes, bytes]:
    rng = np.random.default_rng(1)
    codes = rng.integers(-1, 2, size=(4, 256)).astype(np.int8)
    scales = (rng.random((4, 1)) + 0.1).astype(np.float32)
    ternary = pg.pack_tq2_0(codes, scales)
    f32 = pg.pack_f32(rng.standard_normal((2, 3)).astype(np.float32))
    f16 = pg.pack_f16(rng.standard_normal(5).astype(np.float16))
    writer = pg.GGUFWriter(alignment=alignment)
    writer.add_metadata("general.architecture", "llama")
    writer.add_metadata("llama.vocab_size", 256)
    writer.add_metadata("some.flag", True)
    writer.add_metadata("some.ratio", 0.5)
    writer.add_metadata("some.negative", -7)
    writer.add_metadata("some.big", 2**40)
    writer.add_metadata("tokenizer.ggml.tokens", ["a", "b", "c"])
    writer.add_metadata("some.ints", [1, 2, 3])
    writer.add_tensor("blk.0.attn_q.weight", (256, 4), pg.GGML_TYPE_TQ2_0, ternary)
    writer.add_tensor("norm.weight", (3, 2), pg.GGML_TYPE_F32, f32)
    writer.add_tensor("small.weight", (5,), pg.GGML_TYPE_F16, f16)
    path = writer.write(tmp_path / "round_trip.gguf")
    stored_scales = scales.astype(np.float16).astype(np.float32)
    return path, codes * np.repeat(stored_scales, 256, axis=-1), f32, f16


def test_writer_reader_round_trip(tmp_path: Path) -> None:
    path, expected_ternary, f32, f16 = _round_trip_file(tmp_path)
    reader = pg.GGUFReader(path)
    assert reader.metadata["general.architecture"] == "llama"
    assert reader.metadata["some.flag"] is True
    assert reader.metadata["some.negative"] == -7
    assert reader.metadata["some.big"] == 2**40
    assert reader.metadata["some.ratio"] == np.float32(0.5)
    assert reader.metadata["tokenizer.ggml.tokens"] == ["a", "b", "c"]
    assert reader.metadata["some.ints"] == [1, 2, 3]
    assert set(reader.tensors) == {"blk.0.attn_q.weight", "norm.weight", "small.weight"}
    assert reader.tensors["blk.0.attn_q.weight"].type_name == "TQ2_0"
    assert reader.tensors["norm.weight"].shape == (3, 2)
    assert np.array_equal(reader.dequantize("blk.0.attn_q.weight"), expected_ternary)
    assert np.array_equal(reader.dequantize("norm.weight"), np.frombuffer(f32, dtype="<f4").reshape(2, 3))
    assert np.array_equal(reader.dequantize("small.weight"), np.frombuffer(f16, dtype="<f2").astype(np.float32))


def test_alignment_is_honored_and_recorded(tmp_path: Path) -> None:
    path, _, _, _ = _round_trip_file(tmp_path, alignment=64)
    reader = pg.GGUFReader(path)
    assert reader.metadata["general.alignment"] == 64
    offsets = [info.offset for info in reader.tensors.values()]
    assert offsets[0] == 0
    for offset in offsets:
        assert offset % 64 == 0


def test_writer_validation(tmp_path: Path) -> None:
    writer = pg.GGUFWriter()
    with pytest.raises(ValueError):
        pg.GGUFWriter(alignment=0)
    with pytest.raises(ValueError):
        pg.GGUFWriter(alignment=24)
    writer.add_tensor("t", (4,), pg.GGML_TYPE_F32, b"\x00" * 16)
    with pytest.raises(ValueError):
        writer.add_tensor("t", (4,), pg.GGML_TYPE_F32, b"\x00" * 16)
    with pytest.raises(ValueError):
        writer.add_tensor("u", (4,), pg.GGML_TYPE_F32, b"\x00" * 15)
    with pytest.raises(TypeError):
        writer.add_metadata("bad", object())


def test_reader_rejects_malformed(tmp_path: Path) -> None:
    path = tmp_path / "bad.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 2, 0, 0) + b"\x00" * 4)
    with pytest.raises(ValueError):
        pg.GGUFReader(path)
    path.write_bytes(b"NOPE" + b"\x00" * 64)
    with pytest.raises(ValueError):
        pg.GGUFReader(path)
    path.write_bytes(b"GGUF")
    with pytest.raises(ValueError):
        pg.GGUFReader(path)


def test_gguf_py_reads_our_file(tmp_path: Path) -> None:
    gguf = pytest.importorskip("gguf")
    path, expected_ternary, f32, _ = _round_trip_file(tmp_path)
    reader = gguf.GGUFReader(path)
    by_name = {tensor.name: tensor for tensor in reader.tensors}
    assert by_name["blk.0.attn_q.weight"].tensor_type == gguf.GGMLQuantizationType.TQ2_0
    assert tuple(int(d) for d in by_name["blk.0.attn_q.weight"].shape) == (256, 4)
    blob = path.read_bytes()
    info = by_name["blk.0.attn_q.weight"]
    start = int(info.data_offset)
    raw = blob[start : start + len(info.data.tobytes())]
    assert len(raw) == 66 * 4
    assert info.data.tobytes() == raw
    assert np.array_equal(pg.GGUFReader(path).dequantize("blk.0.attn_q.weight"), expected_ternary)
    assert reader.fields["general.architecture"].contents() == "llama"
    assert list(reader.fields["tokenizer.ggml.tokens"].contents()) == ["a", "b", "c"]
    assert by_name["norm.weight"].tensor_type == gguf.GGMLQuantizationType.F32
    assert np.array_equal(
        np.frombuffer(blob[int(by_name["norm.weight"].data_offset) : int(by_name["norm.weight"].data_offset) + 24], dtype="<f4").reshape(2, 3),
        np.frombuffer(f32, dtype="<f4").reshape(2, 3),
    )


def test_our_reader_reads_gguf_py_file(tmp_path: Path) -> None:
    gguf = pytest.importorskip("gguf")
    path = tmp_path / "from_gguf_py.gguf"
    writer = gguf.GGUFWriter(str(path), arch="llama")
    writer.add_uint32("llama.vocab_size", 256)
    writer.add_string("general.name", "tiny")
    writer.add_array("tokenizer.ggml.tokens", ["x", "y", "z"])
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    writer.add_tensor("t.f32", array)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    reader = pg.GGUFReader(path)
    assert reader.metadata["llama.vocab_size"] == 256
    assert reader.metadata["general.name"] == "tiny"
    assert reader.metadata["tokenizer.ggml.tokens"] == ["x", "y", "z"]
    assert reader.tensors["t.f32"].type_name == "F32"
    assert np.array_equal(reader.dequantize("t.f32"), array)


@requires_ggml
@pytest.mark.parametrize("n", [256, 512])
def test_ggml_quantizer_bytes_round_trip(n: int) -> None:
    x = np.random.default_rng(2).standard_normal(n).astype(np.float32)
    out = np.zeros(pg.tensor_nbytes(pg.GGML_TYPE_TQ2_0, (n,)), dtype=np.uint8)
    GGML.quantize_row_tq2_0_ref(
        x.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p), n
    )
    reference = out.tobytes()
    codes, scales = pg.unpack_tq2_0(reference, (n,))
    assert pg.pack_tq2_0(codes, scales) == reference


@requires_ggml
def test_ggml_dequantize_matches_ours() -> None:
    rng = np.random.default_rng(3)
    codes = rng.integers(-1, 2, size=(256,)).astype(np.int8)
    scales = np.array([0.375], dtype=np.float16).astype(np.float32)
    payload = pg.pack_tq2_0(codes.reshape(1, 256), scales.reshape(1, 1))
    buffer = ctypes.create_string_buffer(payload, len(payload))
    out = np.zeros(256, dtype=np.float32)
    GGML.dequantize_row_tq2_0(
        ctypes.cast(buffer, ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p), 256
    )
    ours = pg.dequantize_tq2_0(codes.reshape(1, 256), scales.reshape(1, 1)).ravel()
    assert np.array_equal(out, ours)


def test_pack_tq2_0_tensor_helper(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    weight = rng.standard_normal((8, 512)) * 0.05
    payload = pg.pack_tq2_0_tensor(weight, group_size=256)
    assert len(payload) == 66 * 2 * 8
    codes, scales = pg.unpack_tq2_0(payload, (512, 8))
    err = np.abs(pg.dequantize_tq2_0(codes, scales) - weight).mean()
    assert err < np.abs(weight).mean()
    with pytest.raises(ValueError):
        pg.pack_tq2_0_tensor(rng.standard_normal(256), group_size=256)
    with pytest.raises(ValueError):
        pg.pack_tq2_0_tensor(rng.standard_normal((8, 512)), group_size=128)


def test_artifact_naming() -> None:
    assert pg.artifact_name("A", "abc1234", "20260919") == "tbr27b-a-abc1234-20260919.gguf"
    with pytest.raises(ValueError):
        pg.artifact_name("D", "abc1234", "20260919")
    with pytest.raises(ValueError):
        pg.artifact_name("A", "", "20260919")


def test_metadata_type_coverage(tmp_path: Path) -> None:
    writer = pg.GGUFWriter()
    writer.add_metadata("u8ish", 255)
    writer.add_metadata("u16ish", 70000)
    writer.add_metadata("i32ish", -(2**20))
    writer.add_metadata("i64ish", -(2**40))
    writer.add_metadata("f64ish", 1.5)
    writer.add_metadata("empty.list", [])
    writer.add_metadata("bools", [True, False])
    writer.add_metadata("floats", [0.25, 0.5])
    path = writer.write(tmp_path / "types.gguf")
    metadata = pg.GGUFReader(path).metadata
    assert metadata["u8ish"] == 255
    assert metadata["u16ish"] == 70000
    assert metadata["i32ish"] == -(2**20)
    assert metadata["i64ish"] == -(2**40)
    assert metadata["f64ish"] == 1.5
    assert metadata["empty.list"] == []
    assert metadata["bools"] == [True, False]
    assert metadata["floats"] == [np.float32(0.25), np.float32(0.5)]


def _tiny_model_metadata(vocab: int, hidden: int, ffn: int, layers: int, heads: int, kv_heads: int, head_dim: int) -> dict:
    tokens = ["<unk>", "<s>", "</s>"] + [f"<{i}>" for i in range(3, vocab)]
    token_types = [2, 3, 3] + [1] * (vocab - 3)
    return {
        "general.architecture": "llama",
        "general.name": "tbr-tiny",
        "llama.vocab_size": vocab,
        "llama.context_length": 128,
        "llama.embedding_length": hidden,
        "llama.feed_forward_length": ffn,
        "llama.block_count": layers,
        "llama.attention.head_count": heads,
        "llama.attention.head_count_kv": kv_heads,
        "llama.attention.key_length": head_dim,
        "llama.attention.value_length": head_dim,
        "llama.attention.layer_norm_rms_epsilon": 1e-5,
        "llama.rope.dimension_count": head_dim,
        "llama.rope.freq_base": 10000.0,
        "tokenizer.ggml.model": "llama",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.scores": [0.0] * vocab,
        "tokenizer.ggml.token_type": token_types,
        "tokenizer.ggml.bos_token_id": 1,
        "tokenizer.ggml.eos_token_id": 2,
        "tokenizer.ggml.unknown_token_id": 0,
    }


def write_tiny_model(path: Path, vocab: int = 256, hidden: int = 256, ffn: int = 512, layers: int = 1) -> Path:
    """A ~200 KB llama-architecture GGUF: norms F32, linears TQ2_0."""
    heads, kv_heads, head_dim = 2, 1, hidden // 2
    rng = np.random.default_rng(7)
    writer = pg.GGUFWriter()
    for key, value in _tiny_model_metadata(vocab, hidden, ffn, layers, heads, kv_heads, head_dim).items():
        if key == "tokenizer.ggml.token_type":
            writer.add_metadata(key, value, element_type=pg.GGUF_TYPE_INT32)
        else:
            writer.add_metadata(key, value)

    def add_ternary(name: str, shape: tuple[int, ...]) -> None:
        torch_order = rng.standard_normal(tuple(reversed(shape))) * 0.05
        writer.add_tensor(name, shape, pg.GGML_TYPE_TQ2_0, pg.pack_tq2_0_tensor(torch_order))

    def add_norm(name: str, size: int) -> None:
        writer.add_tensor(name, (size,), pg.GGML_TYPE_F32, pg.pack_f32(np.ones(size, dtype=np.float32)))

    add_ternary("token_embd.weight", (hidden, vocab))
    add_ternary("output.weight", (hidden, vocab))
    add_norm("output_norm.weight", hidden)
    for layer in range(layers):
        prefix = f"blk.{layer}"
        add_norm(f"{prefix}.attn_norm.weight", hidden)
        add_ternary(f"{prefix}.attn_q.weight", (hidden, heads * head_dim))
        add_ternary(f"{prefix}.attn_k.weight", (hidden, kv_heads * head_dim))
        add_ternary(f"{prefix}.attn_v.weight", (hidden, kv_heads * head_dim))
        add_ternary(f"{prefix}.attn_output.weight", (heads * head_dim, hidden))
        add_norm(f"{prefix}.ffn_norm.weight", hidden)
        add_ternary(f"{prefix}.ffn_gate.weight", (hidden, ffn))
        add_ternary(f"{prefix}.ffn_up.weight", (hidden, ffn))
        add_ternary(f"{prefix}.ffn_down.weight", (ffn, hidden))
    return writer.write(path)


def test_tiny_model_is_wellformed(tmp_path: Path) -> None:
    path = write_tiny_model(tmp_path / "tiny.gguf")
    assert path.stat().st_size < 2 * 1024 * 1024
    reader = pg.GGUFReader(path)
    assert reader.metadata["general.architecture"] == "llama"
    assert reader.metadata["llama.block_count"] == 1
    assert len(reader.metadata["tokenizer.ggml.tokens"]) == 256
    assert reader.tensors["token_embd.weight"].type_name == "TQ2_0"
    assert reader.tensors["blk.0.ffn_down.weight"].shape == (512, 256)
    assert reader.tensors["output_norm.weight"].type_name == "F32"
    assert np.allclose(reader.dequantize("output_norm.weight"), 1.0)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.skipif(
    not (
        (Path.home() / ".unsloth" / "llama.cpp" / "build" / "bin" / "llama-server").is_file()
        and os.environ.get("TBR_LIVE_GGUF")
    ),
    reason="live llama-server load is gated by TBR_LIVE_GGUF=1 (local ROCm build required)",
)
def test_tiny_gguf_loads_in_llama_server(tmp_path: Path) -> None:
    path = write_tiny_model(tmp_path / "tiny.gguf")
    port = _free_port()
    process = subprocess.Popen(
        [
            str(Path.home() / ".unsloth" / "llama.cpp" / "build" / "bin" / "llama-server"),
            "-m",
            str(path),
            "-ngl",
            "0",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-c",
            "64",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.time() + 120
        healthy = False
        while time.time() < deadline:
            if process.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                    if response.status == 200:
                        healthy = True
                        break
            except Exception:
                time.sleep(0.5)
        if not healthy:
            log = process.stdout.read() if process.poll() is not None else ""
            pytest.fail(f"llama-server never became healthy (exit={process.poll()}):\n{log[-4000:]}")
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
