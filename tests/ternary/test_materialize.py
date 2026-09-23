"""Offline tests for the PQ2_0 -> transformers/Qwen3_5 materializer."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bonsai_forensics import materialize as mat
from bonsai_forensics import pack_gguf, pq2_0
from bonsai_forensics.prism_loader import load_prism_gguf

BLOCK = 1024
HIDDEN = 1024
INTER = 1024
VOCAB = 256
NV, NK, HK, HD = 8, 4, 128, 128
QFULL = NV * HD  # 1024
KVFULL = NK * HK  # 512
QKV = 2 * NK * HK + NV * HD  # 2048

FWD = (
    "blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_out.weight",
    "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
    "blk.1.attn_q.weight", "blk.1.attn_k.weight", "blk.1.attn_v.weight",
    "blk.1.attn_output.weight", "blk.1.ffn_gate.weight", "blk.1.ffn_up.weight",
    "blk.1.ffn_down.weight", "output.weight",
)
INV = ("token_embd.weight",)


def _pq2(rows, ne0=HIDDEN, seed=0):
    rng = np.random.default_rng(seed)
    codes = rng.integers(-1, 2, size=(rows, ne0)).astype(np.int8)
    scales = rng.uniform(0.01, 0.08, size=(rows, ne0 // pq2_0.BLOCK_SIZE)).astype(np.float32)
    return pq2_0.pack_pq2_0(codes, scales)


def _mini_gguf(tmp_path):
    rng = np.random.default_rng(3)
    writer = pq2_0.PQ2_0GGUFWriter()
    meta = {
        "general.architecture": "qwen35",
        "qwen35.block_count": 2,
        "qwen35.context_length": 2048,
        "qwen35.embedding_length": HIDDEN,
        "qwen35.feed_forward_length": INTER,
        "qwen35.attention.head_count": 8,
        "qwen35.attention.head_count_kv": 4,
        "qwen35.attention.key_length": 128,
        "qwen35.attention.layer_norm_rms_epsilon": 1e-6,
        "qwen35.rope.dimension_count": 64,
        "qwen35.rope.freq_base": 10000.0,
        "qwen35.ssm.conv_kernel": 4,
        "qwen35.ssm.group_count": NK,
        "qwen35.ssm.inner_size": NV * HD,
        "qwen35.ssm.state_size": HK,
        "qwen35.ssm.time_step_rank": NV,
        "qwen35.full_attention_interval": 2,
        "prism.hadamard.version": 1,
        "prism.hadamard.block_size": BLOCK,
        "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
        "prism.hadamard.axis": "input-last-dimension",
        "prism.hadamard.sign_mode": "explicit",
        "prism.hadamard.gdn_v_grouped": True,
        "prism.hadamard.sign_widths": [BLOCK],
        "prism.hadamard.weight_names": list(FWD),
        "prism.hadamard.inverse_weight_names": list(INV),
    }
    for key, value in meta.items():
        writer.add_metadata(key, value)
    writer.add_metadata(
        "prism.hadamard.sign_values",
        rng.choice([-1, 1], size=BLOCK).tolist(),
        element_type=pack_gguf.GGUF_TYPE_INT32,
    )

    def pq(name, shape, seed):
        writer.add_tensor(name, shape, pq2_0.GGML_TYPE_PQ2_0, _pq2(shape[1], shape[0], seed))

    def f32(name, shape, data):
        writer.add_tensor(name, shape, 0, pack_gguf.pack_f32(data, shape))

    def f16(name, shape, data):
        writer.add_tensor(name, shape, 1, pack_gguf.pack_f16(data, shape))

    pq("token_embd.weight", (HIDDEN, VOCAB), 1)
    pq("output.weight", (HIDDEN, VOCAB), 2)
    f32("output_norm.weight", (HIDDEN,), np.ones(HIDDEN))
    pq("blk.0.attn_qkv.weight", (HIDDEN, QKV), 10)
    pq("blk.0.attn_gate.weight", (HIDDEN, QFULL), 11)
    pq("blk.0.ssm_out.weight", (HIDDEN, HIDDEN), 12)
    pq("blk.0.ffn_gate.weight", (HIDDEN, INTER), 13)
    pq("blk.0.ffn_up.weight", (HIDDEN, INTER), 14)
    pq("blk.0.ffn_down.weight", (INTER, HIDDEN), 15)
    f32("blk.0.attn_norm.weight", (HIDDEN,), np.ones(HIDDEN))
    f32("blk.0.post_attention_norm.weight", (HIDDEN,), np.ones(HIDDEN))
    f16("blk.0.ssm_alpha.weight", (HIDDEN, NV), np.full((NV, HIDDEN), 0.5, dtype=np.float32))
    f16("blk.0.ssm_beta.weight", (HIDDEN, NV), np.full((NV, HIDDEN), 0.25, dtype=np.float32))
    f32("blk.0.ssm_a", (NV,), np.full(NV, -1.5, dtype=np.float32))
    f32("blk.0.ssm_dt.bias", (NV,), np.arange(1, NV + 1, dtype=np.float32))
    f32("blk.0.ssm_norm.weight", (HD,), np.ones(HD))
    f32("blk.0.ssm_conv1d.weight", (4, QKV), np.arange(QKV * 4, dtype=np.float32).reshape(QKV, 4) / 1000.0)
    pq("blk.1.attn_q.weight", (HIDDEN, 2 * QFULL), 20)
    pq("blk.1.attn_k.weight", (HIDDEN, KVFULL), 21)
    pq("blk.1.attn_v.weight", (HIDDEN, KVFULL), 22)
    pq("blk.1.attn_output.weight", (QFULL, HIDDEN), 23)
    pq("blk.1.ffn_gate.weight", (HIDDEN, INTER), 24)
    pq("blk.1.ffn_up.weight", (HIDDEN, INTER), 25)
    pq("blk.1.ffn_down.weight", (INTER, HIDDEN), 26)
    f32("blk.1.attn_norm.weight", (HIDDEN,), np.ones(HIDDEN))
    f32("blk.1.post_attention_norm.weight", (HIDDEN,), np.ones(HIDDEN))
    f32("blk.1.attn_q_norm.weight", (128,), np.ones(128))
    f32("blk.1.attn_k_norm.weight", (128,), np.ones(128))
    path = tmp_path / "mini.gguf"
    writer.write(path)
    return path


def test_text_config_matches_gguf(tmp_path):
    from bonsai_forensics.gate1_forensics import parse_gguf_header

    config = mat.text_config_from_gguf(parse_gguf_header(_mini_gguf(tmp_path)))
    assert config["model_type"] == "qwen3_5_text"
    assert config["architectures"] == ["Qwen3_5ForCausalLM"]
    assert config["hidden_size"] == HIDDEN
    assert config["vocab_size"] == VOCAB
    assert config["num_hidden_layers"] == 2
    assert config["linear_num_value_heads"] == NV
    assert config["linear_num_key_heads"] == NK
    assert config["linear_value_head_dim"] == HD
    assert config["linear_key_head_dim"] == HK
    assert config["layer_types"] == ["linear_attention", "full_attention"]
    assert config["rope_parameters"]["partial_rotary_factor"] == 0.5


def test_target_name_mapping():
    assert mat.target_name("token_embd.weight")[0] == "model.embed_tokens.weight"
    assert mat.target_name("output.weight")[0] == "lm_head.weight"
    assert mat.target_name("blk.5.ssm_a")[0] == "model.layers.5.linear_attn.A_log"
    assert mat.target_name("blk.5.attn_q.weight")[0] == "model.layers.5.self_attn.q_proj.weight"
    with pytest.raises(ValueError, match="unmapped"):
        mat.target_name("blk.0.mystery.weight")


def test_conv1d_axis_and_ssm_a_log(tmp_path):
    model = load_prism_gguf(_mini_gguf(tmp_path))
    geometry = mat.GdnGeometry.from_header(model.header)

    _, conv = mat.materialize_tensor(model, "blk.0.ssm_conv1d.weight", geometry)
    assert conv.shape == (2 * NK * HK + NV * HD, 1, 4)

    _, a_log = mat.materialize_tensor(model, "blk.0.ssm_a", geometry)
    assert a_log.shape == (NV,)
    # materialize_tensor defaults to float16, so compare at f16 precision.
    np.testing.assert_allclose(a_log, np.log(1.5), rtol=0, atol=1e-3)


def test_pq2_tensor_is_unrotated_and_row_permuted(tmp_path):
    from bonsai_forensics import rotation

    model = load_prism_gguf(_mini_gguf(tmp_path))
    geometry = mat.GdnGeometry.from_header(model.header)
    name = "blk.0.attn_gate.weight"
    _, got = mat.materialize_tensor(model, name, geometry)

    dequant = model.dequant(name)
    unrotated = rotation.apply_rotation(
        dequant, model.basis.signs(HIDDEN), transpose=True
    )
    expected = unrotated[geometry.vperm(HD)]
    np.testing.assert_allclose(got, expected.astype(np.float16), rtol=0, atol=1e-3)
    np.testing.assert_allclose(np.linalg.norm(got, axis=1), np.linalg.norm(expected, axis=1), rtol=1e-3)


def test_materialize_writes_index_and_shards(tmp_path):
    out = tmp_path / "out"
    report = mat.materialize(_mini_gguf(tmp_path), out, shard_bytes=1)
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] > 0
    assert report["shards"] > 1  # 1-byte budget forces one shard per tensor
    assert (out / "config.json").is_file()
    assert all((out / shard).is_file() for shard in index["weight_map"].values())


def test_materialized_checkpoint_loads_into_transformers(tmp_path):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    out = tmp_path / "out"
    mat.materialize(_mini_gguf(tmp_path), out, dtype="float16")

    model = transformers.AutoModelForCausalLM.from_pretrained(out)
    params = dict(model.named_parameters())

    assert "model.layers.0.linear_attn.in_proj_qkv.weight" in params
    assert "model.layers.0.linear_attn.conv1d.weight" in params
    assert "model.layers.1.self_attn.q_proj.weight" in params
    assert params["model.layers.0.linear_attn.conv1d.weight"].shape == (
        2 * NK * HK + NV * HD, 1, 4,
    )
    assert params["model.embed_tokens.weight"].shape == (VOCAB, HIDDEN)
    assert params["lm_head.weight"].shape == (VOCAB, HIDDEN)
