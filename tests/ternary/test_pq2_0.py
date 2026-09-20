"""T25 — offline tests for the R2-verified PQ2_0 block codec.

Synthetic blocks only: byte layout, round-trips, validation, decoder parity
with `oracle.decode_pq2_0` (T7's byte-exact reference against Prism's own
dequant), and a type-142 GGUF round trip read back by the independent `oracle`
parser. No GPU, no downloads, no artifact files.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from bonsai_forensics import oracle
from bonsai_forensics import pq2_0 as pq

ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "bonsai_forensics" / "spec.md"
SPEC_SHA256 = "0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b"


def _spec_constants() -> dict:
    text = SPEC_PATH.read_text(encoding="utf-8")
    return json.loads(re.findall(r"```json\n(.*?)\n```", text, re.S)[0])


def test_spec_hash_is_pinned() -> None:
    canon = json.dumps(_spec_constants(), sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canon.encode()).hexdigest() == SPEC_SHA256 == pq.SPEC_SHA256


def test_constants_match_r2() -> None:
    assert pq.GGML_TYPE_PQ2_0 == 142
    assert (pq.BLOCK_SIZE, pq.BLOCK_BYTES, pq.QS_OFFSET, pq.QS_BYTES) == (128, 34, 2, 32)
    assert pq.BPW == pytest.approx(2.125)


def test_row_size_and_nbytes() -> None:
    assert pq.pq2_0_row_size(128) == 34
    assert pq.pq2_0_row_size(5120) == 40 * 34
    assert pq.pq2_0_nbytes((128, 3)) == 3 * 34
    assert pq.pq2_0_nbytes((5120, 7)) == 7 * 40 * 34
    with pytest.raises(ValueError):
        pq.pq2_0_row_size(100)
    with pytest.raises(ValueError):
        pq.pq2_0_nbytes((128, 0))


def test_pack_byte_layout_is_pinned() -> None:
    codes = np.zeros((1, 128), dtype=np.int8)
    codes[0, :4] = [-1, 0, 1, -1]
    scales = np.array([[0.5]], dtype=np.float32)
    block = pq.pack_pq2_0(codes, scales)
    assert len(block) == 34
    assert np.frombuffer(block[:2], dtype="<f2")[0] == pytest.approx(0.5)
    # codes -> 2-bit: -1->0, 0->1, +1->2 at shifts 0, 2, 4, 6 of qs[0]
    assert block[2] == 0x24
    # values 4..7 are all ternary 0 -> code 1 at every shift
    assert block[3] == 0x55
    assert all(b == 0 for b in block[34:])


def test_round_trip_codes_exact_scales_fp16() -> None:
    rng = np.random.default_rng(0)
    codes = rng.integers(-1, 2, size=(5, 256)).astype(np.int8)
    scales = rng.uniform(0.1, 1.5, size=(5, 2)).astype(np.float32)
    data = pq.pack_pq2_0(codes, scales)
    assert len(data) == pq.pq2_0_nbytes((256, 5))
    back_codes, back_scales = pq.unpack_pq2_0(data, (256, 5))
    assert np.array_equal(back_codes, codes)
    assert np.allclose(back_scales, scales, atol=1e-3)


def test_dequantize_matches_repeat_scale() -> None:
    codes = np.array([[-1, 0, 1]], dtype=np.int8)
    scales = np.array([[0.25]], dtype=np.float32)
    expected = np.array([[-0.25, 0.0, 0.25]], dtype=np.float32)
    assert np.allclose(pq.dequantize_pq2_0(codes, scales), expected)


def test_code_three_decodes_as_plus_two_d() -> None:
    block = bytearray(34)
    block[0:2] = np.float16(0.5).tobytes()
    block[2] = 3  # value 0 -> code 3 -> +2d
    codes, scales = pq.unpack_pq2_0(bytes(block), (128, 1))
    assert codes[0, 0] == 2
    assert np.allclose(pq.dequantize_pq2_0(codes, scales)[0, 0], 1.0)


def test_pack_validation() -> None:
    codes = np.zeros((2, 128), dtype=np.int8)
    scales = np.ones((2, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="int8"):
        pq.pack_pq2_0(codes.astype(np.int16), scales)
    with pytest.raises(ValueError, match="multiple of 128"):
        pq.pack_pq2_0(np.zeros((2, 100), dtype=np.int8), np.ones((2, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="scales shape"):
        pq.pack_pq2_0(codes, np.ones((2, 2), dtype=np.float32))
    bad = codes.copy()
    bad[0, 0] = 2
    with pytest.raises(ValueError, match="ternary codes"):
        pq.pack_pq2_0(bad, scales)
    with pytest.raises(ValueError, match="payload"):
        pq.unpack_pq2_0(b"\x00", (128, 1))


def test_pack_is_deterministic() -> None:
    rng = np.random.default_rng(1)
    codes = rng.integers(-1, 2, size=(4, 128)).astype(np.int8)
    scales = rng.uniform(0.5, 1.0, size=(4, 1)).astype(np.float32)
    assert pq.pack_pq2_0(codes, scales) == pq.pack_pq2_0(codes, scales)


def test_decoder_parity_with_oracle_reference() -> None:
    rng = np.random.default_rng(2)
    ne0, ne1 = 256, 3
    codes = rng.integers(-1, 2, size=(ne1, ne0)).astype(np.int8)
    scales = rng.uniform(0.1, 1.0, size=(ne1, ne0 // pq.BLOCK_SIZE)).astype(np.float32)
    data = pq.pack_pq2_0(codes, scales)
    assert np.allclose(pq.decode_pq2_0(data, ne0, ne1),
                       oracle.decode_pq2_0(data, ne0, ne1), atol=1e-4)


def test_type_142_gguf_round_trips_through_oracle_parser(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    codes = rng.integers(-1, 2, size=(3, 128)).astype(np.int8)
    scales = rng.uniform(0.2, 0.8, size=(3, 1)).astype(np.float32)
    writer = pq.PQ2_0GGUFWriter()
    writer.add_metadata("general.architecture", "llama")
    writer.add_metadata("general.name", "pq2-test")
    writer.add_tensor("blk.0.test.weight", (128, 3), pq.GGML_TYPE_PQ2_0,
                      pq.pack_pq2_0(codes, scales))
    path = writer.write(tmp_path / "pq2.gguf")

    table = oracle.parse_gguf_table(path)
    info = table.tensors["blk.0.test.weight"]
    assert info.ggml_type == pq.GGML_TYPE_PQ2_0
    decoded = oracle.read_tensor(table, "blk.0.test.weight")
    assert decoded.shape == (3, 128)
    assert np.allclose(decoded, pq.dequantize_pq2_0(codes, scales), atol=1e-3)


def test_writer_rejects_wrong_payload_size(tmp_path: Path) -> None:
    writer = pq.PQ2_0GGUFWriter()
    with pytest.raises(ValueError, match="expected"):
        writer.add_tensor("t", (128, 1), pq.GGML_TYPE_PQ2_0, b"\x00")


def test_build_artifact_pq2_0_round_trip(tmp_path: Path) -> None:
    from bonsai_forensics import run_quant as rq

    rng = np.random.default_rng(4)
    codes = rng.integers(-1, 2, size=(3, 128)).astype(np.int8)
    scales = rng.uniform(0.1, 0.9, size=(3, 1)).astype(np.float32)
    f16 = rng.standard_normal((2, 4)).astype(np.float16)
    run_dir = tmp_path / "run"
    rq._write_checkpoint(rq._checkpoint_path(run_dir, "blk.0.attn_qkv.weight"),
                         {"kind": rq.CHECKPOINT_KIND_TERNARY, "codes": codes, "scales": scales})
    rq._write_checkpoint(rq._checkpoint_path(run_dir, "blk.0.input_layernorm.weight"),
                         {"kind": rq.CHECKPOINT_KIND_F16, "data": f16})

    path = rq.build_artifact(
        run_dir, ["blk.0.attn_qkv.weight", "blk.0.input_layernorm.weight"],
        tmp_path / "pq2.gguf", ternary_type="pq2_0",
    )
    table = oracle.parse_gguf_table(path)
    info = table.tensors["blk.0.attn_qkv.weight"]
    assert info.ggml_type == pq.GGML_TYPE_PQ2_0
    assert info.shape == (128, 3)
    decoded = oracle.read_tensor(table, "blk.0.attn_qkv.weight")
    assert np.allclose(decoded, pq.dequantize_pq2_0(codes, scales), atol=1e-3)
    assert table.tensors["blk.0.input_layernorm.weight"].ggml_type == 1
    assert table.metadata["general.file_type"] == rq.PQ2_0_FILE_TYPE == 141


def test_build_artifact_default_stays_tq2_0(tmp_path: Path) -> None:
    from bonsai_forensics import run_quant as rq

    rng = np.random.default_rng(5)
    codes = rng.integers(-1, 2, size=(2, 256)).astype(np.int8)
    scales = rng.uniform(0.1, 0.9, size=(2, 1)).astype(np.float32)
    run_dir = tmp_path / "run"
    rq._write_checkpoint(rq._checkpoint_path(run_dir, "blk.0.ffn_down.weight"),
                         {"kind": rq.CHECKPOINT_KIND_TERNARY, "codes": codes, "scales": scales})
    path = rq.build_artifact(run_dir, ["blk.0.ffn_down.weight"], tmp_path / "tq.gguf")
    table = oracle.parse_gguf_table(path)
    assert table.tensors["blk.0.ffn_down.weight"].ggml_type == 35
    assert "general.file_type" not in table.metadata


def test_build_artifact_rejects_unknown_ternary_type(tmp_path: Path) -> None:
    from bonsai_forensics import run_quant as rq

    with pytest.raises(ValueError, match="unknown ternary type"):
        rq.build_artifact(tmp_path, [], tmp_path / "x.gguf", ternary_type="q4_0")
