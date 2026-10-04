"""Tests for the ``prism.hadamard.*`` metadata contract (offline, no weights)."""

from __future__ import annotations

import numpy as np
import pytest

from bonsai_forensics import pack_gguf, prism_metadata


def _signs(width: int, block: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.choice(np.array([-1, 1], dtype=np.int64), size=block)
            for _ in range(width // block)]


def test_identity_block_fields():
    meta = prism_metadata.prism_hadamard_metadata(
        block_size=1024, weight_names=["blk.0.attn_q.weight", "blk.0.ssm_out.weight"])
    assert meta["prism.hadamard.version"] == 1
    assert meta["prism.hadamard.block_size"] == 1024
    assert meta["prism.hadamard.transform"] == prism_metadata.TRANSFORM
    assert meta["prism.hadamard.axis"] == prism_metadata.AXIS
    assert meta["prism.hadamard.sign_mode"] == "identity"
    assert meta["prism.hadamard.weight_names"] == [
        "blk.0.attn_q.weight", "blk.0.ssm_out.weight"]
    assert "prism.hadamard.sign_widths" not in meta


def test_explicit_signs_round_trip_through_writer():
    rotations = {1024: _signs(1024, 1024, 1), 2048: _signs(2048, 1024, 2)}
    meta = prism_metadata.prism_hadamard_metadata(
        block_size=1024, weight_names=["blk.0.attn_q.weight"],
        sign_mode="explicit", rotations=rotations)
    assert meta["prism.hadamard.sign_widths"] == [1024, 2048]
    assert len(meta["prism.hadamard.sign_values"]) == 1024 + 2048

    writer = pack_gguf.GGUFWriter()
    prism_metadata.add_prism_metadata(writer, meta)
    # the C++ loader reads these into std::vector<int32_t>; pin the element type
    _, element = writer._metadata_types["prism.hadamard.sign_values"]
    assert element == pack_gguf.GGUF_TYPE_INT32
    _, element = writer._metadata_types["prism.hadamard.sign_widths"]
    assert element == pack_gguf.GGUF_TYPE_INT32

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = writer.write(Path(tmp) / "meta.gguf")
        read = pack_gguf.GGUFReader(path).metadata
        assert read["prism.hadamard.sign_widths"] == [1024, 2048]
        assert read["prism.hadamard.sign_values"] == meta["prism.hadamard.sign_values"]


def test_validation_errors():
    with pytest.raises(ValueError, match="power of two"):
        prism_metadata.prism_hadamard_metadata(block_size=1000, weight_names=["w"])
    with pytest.raises(ValueError, match="weight_names is empty"):
        prism_metadata.prism_hadamard_metadata(block_size=1024, weight_names=[])
    with pytest.raises(ValueError, match="duplicates"):
        prism_metadata.prism_hadamard_metadata(
            block_size=1024, weight_names=["w", "w"])
    with pytest.raises(ValueError, match="needs a rotations table"):
        prism_metadata.prism_hadamard_metadata(
            block_size=1024, weight_names=["w"], sign_mode="explicit")
    with pytest.raises(ValueError, match="takes no rotations"):
        prism_metadata.prism_hadamard_metadata(
            block_size=1024, weight_names=["w"],
            rotations={1024: _signs(1024, 1024)})
    with pytest.raises(ValueError, match="not a multiple of block"):
        prism_metadata.prism_hadamard_metadata(
            block_size=1024, weight_names=["w"], sign_mode="explicit",
            rotations={1536: [np.ones(1536, dtype=np.int64)]})
    with pytest.raises(ValueError, match="must be ±1"):
        prism_metadata.prism_hadamard_metadata(
            block_size=1024, weight_names=["w"], sign_mode="explicit",
            rotations={1024: [np.zeros(1024, dtype=np.int64)]})
    with pytest.raises(ValueError, match="entries"):
        prism_metadata.prism_hadamard_metadata(
            block_size=1024, weight_names=["w"], sign_mode="explicit",
            rotations={1024: [np.ones(512, dtype=np.int64)]})


def test_optional_keys():
    meta = prism_metadata.prism_hadamard_metadata(
        block_size=1024, weight_names=["output.weight"],
        inverse_weight_names=["token_embd.weight"], gdn_v_grouped=True)
    assert meta["prism.hadamard.inverse_weight_names"] == ["token_embd.weight"]
    assert meta["prism.hadamard.gdn_v_grouped"] is True
