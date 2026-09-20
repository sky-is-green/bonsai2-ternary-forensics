"""T31 — offline tests for the Gate-2 RTN artifact patcher (synthetic tensors only)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bonsai_forensics import gate1_forensics as g1
from bonsai_forensics import gate2_rtn_artifact as g2
from bonsai_forensics import pack_gguf, pq2_0


def _sign_manifest(tmp_path):
    np.save(tmp_path / "signs-1024.npy", np.ones(1024, dtype=np.float64))
    manifest = {
        "axis": "input-last-dimension",
        "block_size": 1024,
        "sign_widths": [1024],
        "sign_files": ["signs-1024.npy"],
        "transform": "normalized-sylvester-walsh-hadamard",
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _source_gguf(tmp_path):
    writer = pq2_0.PQ2_0GGUFWriter()
    writer.add_metadata("general.architecture", "qwen3_5")
    mapped = pq2_0.pack_pq2_0(np.zeros((4, 1024), dtype=np.int8), np.full((4, 8), 0.5, dtype=np.float32))
    other = pq2_0.pack_pq2_0(np.ones((2, 1024), dtype=np.int8), np.full((2, 8), 0.25, dtype=np.float32))
    head = pq2_0.pack_pq2_0(np.zeros((4, 1024), dtype=np.int8), np.full((4, 8), 0.75, dtype=np.float32))
    writer.add_tensor("blk.0.ffn_down.weight", (1024, 4), 142, mapped)
    writer.add_tensor("blk.0.ffn_up.weight", (1024, 4), 0, pack_gguf.pack_f32(np.zeros((4, 1024)), (1024, 4)))
    writer.add_tensor("blk.0.ssm_conv1d.weight", (1024, 2), 142, other)
    writer.add_tensor("output.weight", (1024, 4), 142, head)
    path = tmp_path / "source.gguf"
    writer.write(path)
    return path, mapped, other, head


def _loader(weight):
    def load(name, start=None, end=None):
        return weight[start:end] if start is not None else weight

    return load


def _payload(path, header, name):
    info = header.tensors[name]
    with open(path, "rb") as handle:
        handle.seek(header.data_offset + info.offset)
        return handle.read(g1.tensor_nbytes(info))


def test_quantize_tensor_packs_expected_codes(tmp_path):
    manifest = _sign_manifest(tmp_path)
    signs = g1.rotation.load_sign_manifest(manifest)["1024"]
    weight = np.arange(4 * 1024, dtype=np.float32).reshape(4, 1024) / 4096.0
    payload, codes, scales = g2.quantize_tensor(weight, signs)
    read_codes, read_scales = pq2_0.unpack_pq2_0(payload, (1024, 4))
    np.testing.assert_array_equal(read_codes, codes)
    np.testing.assert_allclose(read_scales, scales, rtol=1e-3)


def test_patch_entries_includes_layers_and_top_level(tmp_path):
    source, _, _, _ = _source_gguf(tmp_path)
    header = g1.parse_gguf_header(source)
    names = [entry["gguf"] for entry in g2.patch_entries(header, [0])]
    assert names == ["blk.0.ffn_down.weight", "blk.0.ffn_up.weight", "output.weight"]


def test_patch_artifact_patches_mapped_and_preserves_rest(tmp_path):
    source, _, other, _ = _source_gguf(tmp_path)
    header = g1.parse_gguf_header(source)
    manifest = _sign_manifest(tmp_path)
    weight = np.linspace(-1, 1, 4 * 1024, dtype=np.float32).reshape(4, 1024)
    dest = tmp_path / "dest.gguf"
    report = g2.patch_artifact(source, dest, header, manifest, _loader(weight), layers=[0], keep=())

    assert [item["tensor"] for item in report["patched"]] == ["blk.0.ffn_down.weight", "output.weight"]
    assert report["skipped"] == [{"tensor": "blk.0.ffn_up.weight", "reason": "type 0"}]
    assert report["aggregate"]["patched"] == 2

    signs = g1.rotation.load_sign_manifest(manifest)["1024"]
    expected, _, _ = g2.quantize_tensor(weight, signs)
    assert _payload(dest, header, "blk.0.ffn_down.weight") == expected
    assert _payload(dest, header, "output.weight") == expected
    assert _payload(dest, header, "blk.0.ffn_up.weight") == _payload(source, header, "blk.0.ffn_up.weight")
    assert _payload(dest, header, "blk.0.ssm_conv1d.weight") == other

    reparsed = g1.parse_gguf_header(dest)
    assert set(reparsed.tensors) == set(header.tensors)
    assert reparsed.metadata["general.architecture"] == "qwen3_5"
    verify = g2.verify_patch(source, dest, header, report)
    assert verify["ok"] and verify["checked"] == 2


def test_patch_artifact_chunked_rows(tmp_path):
    source, _, _, _ = _source_gguf(tmp_path)
    header = g1.parse_gguf_header(source)
    manifest = _sign_manifest(tmp_path)
    weight = np.linspace(-1, 1, 4 * 1024, dtype=np.float32).reshape(4, 1024)
    dest = tmp_path / "dest.gguf"
    report = g2.patch_artifact(
        source, dest, header, manifest, _loader(weight), layers=[0], keep=(), chunk_rows=2,
    )
    signs = g1.rotation.load_sign_manifest(manifest)["1024"]
    expected, _, _ = g2.quantize_tensor(weight, signs)
    assert _payload(dest, header, "output.weight") == expected
    assert report["patched"][1]["tensor"] == "output.weight"


def test_patch_artifact_keeps_requested_tensors(tmp_path):
    source, _, _, _ = _source_gguf(tmp_path)
    header = g1.parse_gguf_header(source)
    manifest = _sign_manifest(tmp_path)
    weight = np.zeros((4, 1024), dtype=np.float32)
    dest = tmp_path / "dest.gguf"
    report = g2.patch_artifact(
        source, dest, header, manifest, _loader(weight),
        layers=[0], keep=("blk.0.ffn_down.weight", "output.weight"),
    )
    assert report["patched"] == []
    assert {"tensor": "blk.0.ffn_down.weight", "reason": "kept"} in report["skipped"]
    assert {"tensor": "output.weight", "reason": "kept"} in report["skipped"]
    assert _payload(dest, header, "blk.0.ffn_down.weight") == _payload(source, header, "blk.0.ffn_down.weight")


def test_patch_artifact_rejects_same_path(tmp_path):
    source, _, _, _ = _source_gguf(tmp_path)
    header = g1.parse_gguf_header(source)
    manifest = _sign_manifest(tmp_path)
    with pytest.raises(ValueError):
        g2.patch_artifact(source, source, header, manifest, _loader(None), layers=[0])


def test_build_loader_reads_shard(tmp_path):
    from safetensors.numpy import save_file

    values = np.arange(8, dtype=np.float32).reshape(2, 4)
    save_file({"model.language_model.layers.0.mlp.down_proj.weight": values}, tmp_path / "shard.safetensors")
    index = {"weight_map": {"model.language_model.layers.0.mlp.down_proj.weight": "shard.safetensors"}}
    loader = g2.build_loader(tmp_path, index)
    got = loader("model.language_model.layers.0.mlp.down_proj.weight")
    np.testing.assert_allclose(got, values)
    sliced = loader("model.language_model.layers.0.mlp.down_proj.weight", 0, 1)
    np.testing.assert_allclose(sliced, values[:1])


def test_sha256_hex_stable():
    assert g2.sha256_hex(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
