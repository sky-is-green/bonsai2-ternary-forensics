"""T7 — oracle offline tests: PQ2_0/Q2_0 decoders, GGUF table parser, agreement."""

from __future__ import annotations

import numpy as np

from bonsai_forensics import oracle
from bonsai_forensics import pack_gguf as pg


def _block(trits: list[int], scale: float, size: int) -> bytes:
    assert len(trits) == size
    qs = bytearray(size // 4)
    for i, trit in enumerate(trits):
        qs[i // 4] |= ((trit + 1) & 3) << (2 * (i % 4))
    return bytes(qs) + np.float16(scale).tobytes()


def _pq2_0_block(trits: list[int], scale: float) -> bytes:
    """PQ2_0 puts the fp16 scale first (offset 0), then qs[32]."""
    assert len(trits) == 128
    qs = bytearray(32)
    for i, trit in enumerate(trits):
        qs[i // 4] |= ((trit + 1) & 3) << (2 * (i % 4))
    return np.float16(scale).tobytes() + bytes(qs)


def test_decode_pq2_0_matches_layout() -> None:
    trits = [-1, 0, 1, -1] * 32
    decoded = oracle.decode_pq2_0(_pq2_0_block(trits, 0.5), 128, 1)
    assert decoded.shape == (1, 128)
    assert np.allclose(decoded[0], np.array(trits, dtype=np.float32) * 0.5)


def test_decode_q2_0_g64_matches_layout() -> None:
    trits = ([-1, 0, 1] * 21 + [0])[:64]
    decoded = oracle.decode_q2_0_g64(_block(trits, 0.25, 64), 64, 1)
    assert decoded.shape == (1, 64)
    assert np.allclose(decoded[0], np.array(trits, dtype=np.float32) * 0.25)


def test_gguf_table_parser_round_trips_writer(tmp_path) -> None:
    writer = pg.GGUFWriter()
    writer.add_metadata("general.architecture", "qwen3")
    weights = (np.random.default_rng(0).standard_normal((4, 256)) * 0.01).astype(np.float32)
    writer.add_tensor("blk.0.attn_q.weight", (256, 4), pg.GGML_TYPE_F16, pg.pack_f16(weights, (256, 4)))
    path = writer.write(tmp_path / "mini.gguf")

    table = oracle.parse_gguf_table(path)
    assert table.metadata["general.architecture"] == "qwen3"
    info = table.tensors["blk.0.attn_q.weight"]
    assert info.ggml_type == oracle.GGML_F16
    decoded = oracle.read_tensor(table, "blk.0.attn_q.weight")
    assert decoded.shape == (4, 256)
    assert np.allclose(decoded, weights, atol=1e-3)


def _ternary_model(rows: int, cols: int, seed: int):
    rng = np.random.default_rng(seed)
    trits = rng.integers(-1, 2, size=(rows, cols)).astype(np.float32)
    scales = (0.2 + 0.3 * rng.random((rows, cols // 128, 1))).astype(np.float32)
    released = (trits.reshape(rows, cols // 128, 128) * scales).reshape(rows, cols)
    return trits, released.astype(np.float32)


def test_compare_reports_perfect_agreement_on_matching_ternary() -> None:
    trits, released = _ternary_model(3, 256, seed=2)
    score = oracle.compare(trits, released)
    assert set(score) >= {"sign_agreement_absmean", "relative_weight_error", "released_over_base_norm"}
    assert score["sign_agreement_absmean"] == 1.0


def test_compare_detects_a_shuffled_tensor() -> None:
    trits, released = _ternary_model(4, 128, seed=3)
    rng = np.random.default_rng(4)
    shuffled = released.copy()
    rng.shuffle(shuffled, axis=1)
    assert oracle.compare(trits, shuffled)["sign_agreement_absmean"] < 0.9
