"""T30 — offline tests for the Gate-1 forensics module (synthetic tensors only)."""

from __future__ import annotations

import numpy as np
import pytest

from bonsai_forensics import gate1_forensics as g1
from bonsai_forensics import pack_gguf, pq2_0


def _write_gguf(tmp_path, tensors, metadata=None):
    writer = pq2_0.PQ2_0GGUFWriter()
    writer.add_metadata("general.architecture", "qwen3_5")
    for key, value in (metadata or {}).items():
        writer.add_metadata(key, value)
    for name, shape, ggml_type, data in tensors:
        writer.add_tensor(name, shape, ggml_type, data)
    path = tmp_path / "tiny.gguf"
    writer.write(path)
    return path


def test_parse_header_and_read_pq2_0_roundtrip(tmp_path):
    codes = np.array([[1, 0, -1, 1] * 32], dtype=np.int8)
    scales = np.array([[0.125]], dtype=np.float32)
    payload = pq2_0.pack_pq2_0(codes, scales)
    path = _write_gguf(tmp_path, [("blk.0.attn_qkv.weight", (128, 1), 142, payload)])

    header = g1.parse_gguf_header(path)
    info = header.tensors["blk.0.attn_qkv.weight"]
    assert info.ggml_type == 142
    assert header.metadata["general.architecture"] == "qwen3_5"
    read_codes, read_scales = g1.read_pq2_0(header, "blk.0.attn_qkv.weight")
    np.testing.assert_array_equal(read_codes, codes)
    np.testing.assert_allclose(read_scales, scales)


def test_parse_header_rejects_non_gguf(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(b"NOPE" + b"\x00" * 16)
    with pytest.raises(ValueError):
        g1.parse_gguf_header(path)


def test_tensor_nbytes_and_payload_bounds(tmp_path):
    codes = np.zeros((1, 128), dtype=np.int8)
    payload = pq2_0.pack_pq2_0(codes, np.ones((1, 1), dtype=np.float32))
    path = _write_gguf(tmp_path, [("blk.0.ffn_down.weight", (128, 1), 142, payload)])
    header = g1.parse_gguf_header(path)
    assert g1.tensor_nbytes(header.tensors["blk.0.ffn_down.weight"]) == 34
    with pytest.raises(ValueError):
        g1.read_scalar_tensor(header, "blk.0.ffn_down.weight")
    info = g1.TensorInfo("x", (3,), 999, 0)
    with pytest.raises(ValueError):
        g1.tensor_nbytes(info)


def test_reorder_v_heads_matches_manual_transpose():
    num_k, num_v, head = 2, 6, 2
    rows = num_v * head
    tensor = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3)
    got = g1.reorder_v_heads(tensor, 0, num_k_heads=num_k, num_v_heads=num_v, head_dim=head)
    manual = (
        tensor.reshape(num_k, num_v // num_k, head, 3)
        .transpose(1, 0, 2, 3)
        .reshape(rows, 3)
    )
    np.testing.assert_array_equal(got, manual)
    assert sorted(got[:, 0].tolist()) == sorted(tensor[:, 0].tolist())


def test_reorder_v_heads_validates_shape():
    with pytest.raises(ValueError):
        g1.reorder_v_heads(np.zeros((10, 2)), 0, num_k_heads=16, num_v_heads=48, head_dim=128)


def test_apply_layout_rows_keep_q_and_reorder_v():
    width = 3
    weight = np.arange(10240 * width, dtype=np.float32).reshape(10240, width)
    out = g1.apply_layout(weight, "qkv_v_rows")
    np.testing.assert_array_equal(out[:4096], weight[:4096])
    np.testing.assert_array_equal(out[4096:], g1.reorder_v_heads(weight[4096:], 0))
    z = np.arange(6144 * width, dtype=np.float32).reshape(6144, width)
    np.testing.assert_array_equal(g1.apply_layout(z, "z_rows"), g1.reorder_v_heads(z, 0))
    np.testing.assert_array_equal(g1.apply_layout(weight, "grouped"), weight)
    with pytest.raises(ValueError):
        g1.apply_layout(weight, "bogus")


def test_layer_map_resolves_linear_and_full_layers():
    tensors = {
        "blk.0.attn_qkv.weight": g1.TensorInfo("blk.0.attn_qkv.weight", (128, 10240), 142, 0),
        "blk.0.attn_gate.weight": g1.TensorInfo("blk.0.attn_gate.weight", (128, 6144), 142, 0),
        "blk.0.ssm_out.weight": g1.TensorInfo("blk.0.ssm_out.weight", (128, 5120), 142, 0),
        "blk.0.ffn_down.weight": g1.TensorInfo("blk.0.ffn_down.weight", (128, 5120), 142, 0),
        "blk.0.ssm_a": g1.TensorInfo("blk.0.ssm_a", (48,), 0, 0),
        "blk.3.attn_q.weight": g1.TensorInfo("blk.3.attn_q.weight", (128, 12288), 142, 0),
        "blk.3.attn_output.weight": g1.TensorInfo("blk.3.attn_output.weight", (128, 5120), 142, 0),
    }
    header = g1.GGUFHeader(path="x", metadata={}, tensors=tensors, data_offset=0)
    linear = {entry["gguf"]: entry for entry in g1.layer_map(header, 0)}
    assert linear["blk.0.attn_qkv.weight"]["hf"].endswith("linear_attn.in_proj_qkv.weight")
    assert linear["blk.0.attn_qkv.weight"]["layout"] == "qkv_v_rows"
    assert linear["blk.0.attn_gate.weight"]["gamma"].endswith("input_layernorm.weight")
    assert linear["blk.0.ssm_out.weight"]["layout"] == "grouped"
    assert linear["blk.0.ffn_down.weight"]["gamma"] is None
    assert "blk.0.ssm_a" not in linear
    full = {entry["gguf"]: entry for entry in g1.layer_map(header, 3)}
    assert full["blk.3.attn_q.weight"]["hf"].endswith("self_attn.q_proj.weight")
    assert full["blk.3.attn_output.weight"]["layout"] == "plain"


def test_hf_tensor_name_resolution():
    assert g1.hf_tensor_name(7, "attn_output") == "model.language_model.layers.7.self_attn.o_proj.weight"
    assert g1.hf_tensor_name(7, "ssm_out") == "model.language_model.layers.7.linear_attn.out_proj.weight"
    assert g1.hf_tensor_name(7, "ssm_norm") is None


def test_absmean_trits_known_vector():
    weight = np.array([[0.9, 0.1, -0.9, 0.3]])
    codes, scales = g1.absmean_trits(weight, group=4)
    np.testing.assert_array_equal(codes, np.array([[1, 0, -1, 1]], dtype=np.int8))
    np.testing.assert_allclose(scales, np.array([[0.55]], dtype=np.float32), rtol=1e-6)


def test_trits_with_scale_uses_supplied_scale():
    weight = np.array([[0.9, 0.1, -0.9, 0.3]])
    codes = g1.trits_with_scale(weight, np.array([[1.0]], dtype=np.float32), group=4)
    np.testing.assert_array_equal(codes, np.array([[1, 0, -1, 0]], dtype=np.int8))


def test_absmean_trits_requires_group_multiple():
    with pytest.raises(ValueError):
        g1.absmean_trits(np.zeros((1, 10)), group=4)


def test_agreement_and_mismatch():
    a = np.array([[1, 0, -1]], dtype=np.int8)
    b = np.array([[1, 1, -1]], dtype=np.int8)
    assert g1.agreement(a, a) == 1.0
    assert g1.agreement(a, b) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        g1.agreement(a, np.zeros((2, 3), dtype=np.int8))


def test_compare_tensor_round_trip_with_synthetic_rotation():
    signs = [np.array([1.0, -1.0, 1.0, -1.0])]
    weight = np.array([[0.4, 0.9, -0.7, 0.2], [0.1, -0.5, 0.8, -0.3]], dtype=np.float32)
    rotated = g1.rotation.absorb_input(weight, signs).astype(np.float32)
    codes, scales = g1.absmean_trits(rotated, group=4)
    result = g1.compare_tensor(weight, codes, scales, signs, quantize_spec=False, group=4)
    assert result["agreement"]["in_rot_absmean"] == 1.0
    assert result["agreement"]["in_rot_their_scale"] == 1.0
    assert result["out_of_range_codes"] == 0
    assert result["scale"]["absmean_over_stored_median"] == pytest.approx(1.0)


def test_compare_tensor_flat_variant_when_gamma_given():
    signs = [np.array([1.0, -1.0, 1.0, -1.0])]
    weight = np.ones((1, 4), dtype=np.float32)
    gamma = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    codes, scales = g1.absmean_trits(weight, group=4)
    result = g1.compare_tensor(weight, codes, scales, signs, gamma=gamma, quantize_spec=False, group=4)
    assert "in_rot_fold_absmean" in result["agreement"]


def test_read_scalar_tensor_f32(tmp_path):
    writer = pack_gguf.GGUFWriter()
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    writer.add_metadata("general.architecture", "qwen3_5")
    writer.add_tensor("blk.0.attn_norm.weight", (4,), 0, pack_gguf.pack_f32(values, (4,)))
    path = tmp_path / "scalar.gguf"
    writer.write(path)
    header = g1.parse_gguf_header(path)
    got = g1.read_scalar_tensor(header, "blk.0.attn_norm.weight")
    np.testing.assert_allclose(got, values)


def test_read_scalar_tensor_bf16(tmp_path):
    class _BF16Writer(pack_gguf.GGUFWriter):
        def add_tensor(self, name, shape, ggml_type, data):
            if int(ggml_type) != 30:
                return super().add_tensor(name, shape, ggml_type, data)
            self._tensors.append((name, tuple(shape), 30, bytes(data)))

    values = np.array([1.5, -2.25, 0.125], dtype=np.float32)
    bf16 = np.asarray(values, dtype=np.float32).view(np.uint32) >> 16
    payload = bf16.astype("<u2").tobytes()
    writer = _BF16Writer()
    writer.add_metadata("general.architecture", "qwen3_5")
    writer.add_tensor("blk.0.ssm_alpha.weight", (3,), 30, payload)
    path = tmp_path / "bf16.gguf"
    writer.write(path)
    header = g1.parse_gguf_header(path)
    got = g1.read_scalar_tensor(header, "blk.0.ssm_alpha.weight")
    np.testing.assert_allclose(got, values)
