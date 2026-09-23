"""Offline tests for the ROCm PQ2_0 GGUF loader (synthetic tensors only)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bonsai_forensics import pack_gguf, pq2_0, rotation
from bonsai_forensics import prism_loader as pl

BLOCK = 1024
FWD = ("blk.0.ffn_gate.weight", "blk.0.ffn_down.weight", "output.weight")
INV = ("token_embd.weight",)
WIDTH = 1024


def _sign_values(widths, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=sum(widths)).astype(np.int64)


def _metadata(widths=(WIDTH,), forward=FWD, inverse=INV, seed=0, **overrides):
    meta = {
        "general.architecture": "qwen35",
        "prism.hadamard.version": 1,
        "prism.hadamard.block_size": 1024,
        "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
        "prism.hadamard.axis": "input-last-dimension",
        "prism.hadamard.sign_mode": "explicit",
        "prism.hadamard.gdn_v_grouped": True,
        "prism.hadamard.sign_widths": list(widths),
        "prism.hadamard.sign_values": _sign_values(widths, seed).tolist(),
        "prism.hadamard.weight_names": list(forward),
        "prism.hadamard.inverse_weight_names": list(inverse),
    }
    meta.update(overrides)
    return meta


def _pack(rows, ne0=WIDTH, seed=1):
    rng = np.random.default_rng(seed)
    codes = rng.integers(-1, 2, size=(rows, ne0)).astype(np.int8)
    scales = rng.uniform(0.01, 0.1, size=(rows, ne0 // pq2_0.BLOCK_SIZE)).astype(np.float32)
    return pq2_0.pack_pq2_0(codes, scales)


def _model_gguf(tmp_path, metadata=None, names=None):
    metadata = metadata or _metadata()
    names = names or [("blk.0.ffn_gate.weight", 6), ("blk.0.ffn_down.weight", 5),
                      ("output.weight", 7), ("token_embd.weight", 8)]
    writer = pq2_0.PQ2_0GGUFWriter()
    for key, value in metadata.items():
        if key == "prism.hadamard.sign_values":
            # ±1 mixes signs, so the writer's first-element inference is wrong
            writer.add_metadata(key, value, element_type=pack_gguf.GGUF_TYPE_INT32)
        else:
            writer.add_metadata(key, value)
    for index, (name, rows) in enumerate(names):
        writer.add_tensor(name, (WIDTH, rows), pq2_0.GGML_TYPE_PQ2_0, _pack(rows, seed=index + 1))
    writer.add_tensor("blk.0.attn_norm.weight", (WIDTH,), 0, pack_gguf.pack_f32(np.ones(WIDTH), (WIDTH,)))
    path = tmp_path / "model.gguf"
    writer.write(path)
    return path


def _dense_rotation(rots):
    g = len(rots[0])
    n = g * len(rots)
    out = np.zeros((n, n))
    for k, signs in enumerate(rots):
        out[k * g : (k + 1) * g, k * g : (k + 1) * g] = rotation.rotation_matrix(signs)
    return out


def test_load_reads_self_describing_basis(tmp_path):
    model = pl.load_prism_gguf(_model_gguf(tmp_path))
    basis = model.basis

    assert basis.block_size == 1024
    assert basis.transform == "normalized-sylvester-walsh-hadamard"
    assert basis.axis == "input-last-dimension"
    assert basis.sign_mode == "explicit"
    assert basis.gdn_v_grouped is True
    assert basis.sign_widths == (WIDTH,)
    assert basis.weight_names == FWD
    assert basis.inverse_weight_names == INV
    assert [s.shape for s in basis.signs(WIDTH)] == [(BLOCK,)]
    assert set(basis.sign_sets) == {str(WIDTH)}
    assert model.width_for("blk.0.ffn_gate.weight") == WIDTH
    assert model.is_inverse("token_embd.weight")
    assert not model.is_inverse("output.weight")
    assert model.census()["transformed_tensors"] == len(FWD) + len(INV)


def test_recover_matches_runtime_semantics(tmp_path):
    """`W_hf = D @ R`, and it cancels the runtime's forward transform."""
    model = pl.load_prism_gguf(_model_gguf(tmp_path))
    name = "blk.0.ffn_gate.weight"
    rots = model.basis.signs(WIDTH)
    dequant = model.dequant(name)
    recovered = model.recover(name)

    dense = _dense_rotation(rots)
    np.testing.assert_allclose(recovered, dequant @ dense, atol=1e-4)

    rng = np.random.default_rng(7)
    x = rng.standard_normal((1, WIDTH)).astype(np.float32)
    lhs = x @ recovered.T
    rhs = rotation.apply_rotation(x, rots, transpose=False) @ dequant.T
    np.testing.assert_allclose(lhs, rhs, atol=1e-4)


def test_recover_applies_to_inverse_embedding_too(tmp_path):
    model = pl.load_prism_gguf(_model_gguf(tmp_path))
    rots = model.basis.signs(WIDTH)
    dequant = model.dequant("token_embd.weight")
    recovered = model.recover("token_embd.weight")
    np.testing.assert_allclose(recovered, dequant @ _dense_rotation(rots), atol=1e-4)


def test_iter_recover_covers_manifest_in_order(tmp_path):
    model = pl.load_prism_gguf(_model_gguf(tmp_path))
    seen = [name for name, _ in model.iter_recover()]
    assert seen == list(FWD) + list(INV)
    for name, weights in model.iter_recover():
        info = model.header.tensors[name]
        assert weights.shape == (info.shape[1], info.shape[0])


def test_verify_against_manifest(tmp_path):
    model = pl.load_prism_gguf(_model_gguf(tmp_path))
    basis = model.basis
    for width in basis.sign_widths:
        np.save(tmp_path / f"signs-{width}.npy", np.concatenate(basis.signs(width)))
    manifest = tmp_path / "hadamard-manifest.json"
    manifest.write_text(json.dumps({
        "block_size": 1024,
        "transform": "normalized-sylvester-walsh-hadamard",
        "sign_widths": list(basis.sign_widths),
        "sign_files": [f"signs-{w}.npy" for w in basis.sign_widths],
    }), encoding="utf-8")

    assert pl.verify_against_manifest(basis, manifest)["ok"]["match"] is True

    corrupted = np.load(tmp_path / f"signs-{basis.sign_widths[0]}.npy")
    corrupted[3] *= -1
    np.save(tmp_path / f"signs-{basis.sign_widths[0]}.npy", corrupted)
    report = pl.verify_against_manifest(basis, manifest)
    assert report["ok"]["match"] is False
    assert report[str(basis.sign_widths[0])]["first_mismatch"] == 3


def test_parse_rejects_bad_metadata():
    with pytest.raises(ValueError, match="block size"):
        pl.parse_prism_basis(_metadata(**{"prism.hadamard.block_size": 512}))
    # non-frozen block is accepted only when strict is off
    assert pl.parse_prism_basis(
        _metadata(**{"prism.hadamard.block_size": 512}), strict=False
    ).block_size == 512

    with pytest.raises(ValueError, match="sign_values"):
        pl.parse_prism_basis(_metadata(**{"prism.hadamard.sign_values": [1, -1]}))
    with pytest.raises(ValueError, match="overlap"):
        pl.parse_prism_basis(
            _metadata(forward=("token_embd.weight",), inverse=("token_embd.weight",))
        )
    with pytest.raises(ValueError, match="transform"):
        pl.parse_prism_basis(_metadata(**{"prism.hadamard.transform": "dct"}))


def test_load_rejects_manifest_tensor_missing_from_gguf(tmp_path):
    metadata = _metadata(forward=FWD + ("blk.9.ffn_up.weight",))
    with pytest.raises(ValueError, match="missing tensor"):
        pl.load_prism_gguf(_model_gguf(tmp_path, metadata=metadata))


def test_load_rejects_transform_manifest_of_unquantized_tensor(tmp_path):
    path = _model_gguf(tmp_path, metadata=_metadata(forward=("blk.0.attn_norm.weight",)))
    with pytest.raises(ValueError, match="not PQ2_0"):
        pl.load_prism_gguf(path)
