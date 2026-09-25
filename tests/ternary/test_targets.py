from types import SimpleNamespace

import pytest
import torch

from bonsai_forensics.targets import (
    QWEN3_5,
    get_profile,
    select_target_linears,
    target_coverage,
    target_side,
)


class _Child(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_qkv = torch.nn.Linear(128, 256, bias=False)
        self.in_proj_z = torch.nn.Linear(128, 128, bias=False)
        self.in_proj_a = torch.nn.Linear(128, 8, bias=False)
        self.in_proj_b = torch.nn.Linear(128, 8, bias=False)
        self.out_proj = torch.nn.Linear(128, 128, bias=False)
        self.q_proj = torch.nn.Linear(128, 128, bias=False)
        self.k_proj = torch.nn.Linear(128, 64, bias=False)
        self.v_proj = torch.nn.Linear(128, 64, bias=False)
        self.o_proj = torch.nn.Linear(128, 128, bias=False)
        self.gate_proj = torch.nn.Linear(128, 256, bias=False)
        self.up_proj = torch.nn.Linear(128, 256, bias=False)
        self.down_proj = torch.nn.Linear(256, 128, bias=False)


class _Model(torch.nn.Module):
    def __init__(self, model_type="qwen3_5", tied=False):
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type, tie_word_embeddings=tied)
        self.layers = torch.nn.ModuleList([_Child()])
        self.lm_head = torch.nn.Linear(128, 256, bias=False)
        self.visual = torch.nn.ModuleDict({"fc": torch.nn.Linear(128, 128)})
        self.mtp = torch.nn.ModuleDict({"fc": torch.nn.Linear(128, 128)})


def test_qwen35_profile_selects_hybrid_and_lm_head():
    model = _Model()
    selected = select_target_linears(model, profile="auto")
    names = [name for name, _ in selected]
    assert "layers.0.in_proj_qkv" in names
    assert "layers.0.in_proj_z" in names
    assert "layers.0.out_proj" in names
    assert "lm_head" in names
    assert not any("in_proj_a" in name or "in_proj_b" in name for name in names)
    assert not any("visual" in name or "mtp" in name for name in names)


def test_qwen3_profile_does_not_silently_accept_qwen35_defaults():
    model = _Model()
    with pytest.raises(ValueError, match="does not match detected"):
        select_target_linears(model, profile="qwen3")


def test_target_sides_follow_profile_not_suffix_position():
    assert target_side("layers.0.in_proj_qkv", QWEN3_5) == "input"
    assert target_side("layers.0.out_proj", QWEN3_5) == "output"
    assert target_side("layers.0.o_proj", QWEN3_5) == "output"
    with pytest.raises(ValueError):
        target_side("layers.0.in_proj_a", QWEN3_5)


def test_coverage_reports_widths_and_blocks():
    report = target_coverage(_Model(), profile="qwen3_5")
    assert report["profile"] == "qwen3_5"
    assert report["selected_linear_tensors"] == 11
    assert report["selected_fraction"] > 0.8
    assert report["rotation_widths"] == [128]
    assert report["effective_rotation_blocks"] == {"128": 128}
    assert report["missed_linear_tensors"] == 4  # recurrent controls + excluded towers


def test_gpt_neox_projection_only_excludes_embed_out():
    model = _Model(model_type="gpt_neox", tied=False)
    model.config.num_hidden_layers = 1
    model.layers = torch.nn.ModuleList()
    model.layers.append(torch.nn.ModuleDict({
        "attention": torch.nn.ModuleDict({
            "query_key_value": torch.nn.Linear(128, 384),
            "dense": torch.nn.Linear(128, 128),
        }),
        "mlp": torch.nn.ModuleDict({
            "dense_h_to_4h": torch.nn.Linear(128, 512),
            "dense_4h_to_h": torch.nn.Linear(512, 128),
        }),
    }))
    model.lm_head = torch.nn.Linear(128, 256)
    model.embed_out = torch.nn.Linear(128, 256)
    selected = select_target_linears(model, profile="gpt_neox", include_lm_head=False)
    assert not any(name.endswith("embed_out") for name, _ in selected)
    report = target_coverage(model, profile="gpt_neox", include_lm_head=False)
    assert report["expected_selected_linear_tensors"] == 4
    assert report["target_count_ok"] is True


def test_unknown_profile_is_explicit():
    with pytest.raises(ValueError, match="unknown target profile"):
        get_profile("not-an-architecture")


def _moe_mlp():
    """The qwen3_5_moe MLP block: fused routed experts + shared expert."""
    mlp = torch.nn.Module()
    mlp.gate = torch.nn.Linear(128, 8, bias=False)  # router, stays FP
    experts = torch.nn.Module()
    experts.gate_up_proj = torch.nn.Parameter(torch.zeros(8, 256, 128))
    experts.down_proj = torch.nn.Parameter(torch.zeros(8, 128, 128))
    mlp.experts = experts
    mlp.shared_expert = torch.nn.ModuleDict({
        "gate_proj": torch.nn.Linear(128, 256, bias=False),
        "up_proj": torch.nn.Linear(128, 256, bias=False),
        "down_proj": torch.nn.Linear(256, 128, bias=False),
    })
    mlp.shared_expert_gate = torch.nn.Linear(128, 1, bias=False)
    return mlp


class _MoEGDNChild(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_attn = torch.nn.ModuleDict({
            "in_proj_qkv": torch.nn.Linear(128, 256, bias=False),
            "in_proj_z": torch.nn.Linear(128, 128, bias=False),
            "out_proj": torch.nn.Linear(128, 128, bias=False),
        })
        self.mlp = _moe_mlp()


class _MoEAttnChild(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.ModuleDict({
            "q_proj": torch.nn.Linear(128, 128, bias=False),
            "k_proj": torch.nn.Linear(128, 64, bias=False),
            "v_proj": torch.nn.Linear(128, 64, bias=False),
            "o_proj": torch.nn.Linear(128, 128, bias=False),
        })
        self.mlp = _moe_mlp()


class _MoEModel(torch.nn.Module):
    def __init__(self, layer_types=("linear_attention", "full_attention")):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="qwen3_5_moe_text",
            tie_word_embeddings=False,
            num_hidden_layers=len(layer_types),
            layer_types=list(layer_types),
        )
        kinds = {"linear_attention": _MoEGDNChild, "full_attention": _MoEAttnChild}
        self.layers = torch.nn.ModuleList([kinds[t]() for t in layer_types])
        self.lm_head = torch.nn.Linear(128, 256, bias=False)


def test_qwen35_moe_selects_attention_shared_expert_and_head():
    model = _MoEModel()
    names = [name for name, _ in select_target_linears(model, profile="auto")]
    assert "layers.0.linear_attn.in_proj_qkv" in names
    assert "layers.0.linear_attn.out_proj" in names
    assert "layers.1.self_attn.q_proj" in names
    assert "layers.0.mlp.shared_expert.gate_proj" in names
    assert "layers.0.mlp.shared_expert.down_proj" in names
    assert "lm_head" in names
    # the router, its shared scalar, and the fused expert banks are not targets
    assert not any(name.endswith("mlp.gate") for name in names)
    assert not any("shared_expert_gate" in name for name in names)
    assert not any("experts." in name for name in names)


def test_qwen35_moe_expected_count_and_coverage():
    report = target_coverage(_MoEModel(), profile="auto")
    assert report["profile"] == "qwen3_5_moe"
    # 3 GDN + 4 full-attention projections, 3 shared-expert linears per layer,
    # plus the untied lm_head.
    assert report["expected_selected_linear_tensors"] == 3 + 4 + 2 * 3 + 1
    assert report["target_count_ok"] is True
    assert report["selected_linear_tensors"] == report["expected_selected_linear_tensors"]


def test_unrecognised_moe_still_fails_explicitly():
    model = _MoEModel()
    model.config.model_type = "qwen3_6_moe_text"
    with pytest.raises(ValueError, match="mixture-of-experts"):
        select_target_linears(model, profile="auto")
