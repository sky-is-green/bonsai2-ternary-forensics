"""Offline tests for the DSpark export-inventory audit (no torch needed)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "pilot" / "dspark_export_check.py"
_spec = importlib.util.spec_from_file_location("dspark_export_check", _PATH)
mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mod  # dataclasses resolve fields via sys.modules[__module__]
_spec.loader.exec_module(mod)


def _small_spec(**kwargs) -> "mod.ForkDSparkSpec":
    base = dict(hidden_size=8, num_layers=1, num_heads=1, num_kv_heads=1, head_dim=8,
                intermediate_size=16, vocab_size=16, n_capture=2)
    base.update(kwargs)
    return mod.ForkDSparkSpec(**base)


def test_expected_uses_capture_times_hidden_for_fc() -> None:
    spec = _small_spec()
    expected = spec.expected()
    assert expected["dspark.fc.weight"] == (8, 16)  # (hidden, n_capture*hidden)
    assert expected["blk.0.attn_q_norm.weight"] == (8,)
    assert expected["blk.0.attn_k_norm.weight"] == (8,)


def test_map_local_name_handles_prefix_and_layers() -> None:
    assert mod.map_local_name("model.layers.3.self_attn.o_proj.weight") == "blk.3.attn_output.weight"
    assert mod.map_local_name("layers.0.mlp.down_proj.weight") == "blk.0.ffn_down.weight"
    assert mod.map_local_name("norm.weight") == "output_norm.weight"
    assert mod.map_local_name("pre_fc_norm_hidden.weight") is None  # explicit no-home


def _local_inventory(spec: "mod.ForkDSparkSpec") -> dict[str, tuple[int, ...]]:
    """Invert the audit's name map: build a local-name inventory the spec accepts."""
    global_inverse = {
        "token_embd.weight": "embed.weight",
        "output_norm.weight": "norm.weight",
        "output.weight": "lm_head.weight",
        "dspark.fc.weight": "fc.weight",
        "dspark.hidden_norm.weight": "hidden_norm.weight",
    }
    layer_inverse = {v: k for k, v in mod.LAYER_NAME_MAP.items()}
    actual: dict[str, tuple[int, ...]] = {}
    for name, shape in spec.expected().items():
        if name in global_inverse:
            actual[global_inverse[name]] = shape
        else:
            prefix, suffix = name.split(".", 1)  # blk, "<i>.<suffix>"
            index, suffix = suffix.split(".", 1)
            actual[f"layers.{index}.{layer_inverse[suffix]}"] = shape
    return actual


def test_a_matching_inventory_passes() -> None:
    spec = _small_spec()
    report = mod.audit(_local_inventory(spec), spec)
    assert report["ok"] and not report["missing"] and not report["extra"]
    assert not report["shape_mismatch"]


def test_smoke_draft_inventory_reports_the_known_mismatches() -> None:
    """A reduced POC inventory: no qk-norm/hidden_norm, concat-fc, extra norms."""
    actual = {
        "embed.weight": (16, 8),
        "norm.weight": (8,),
        "lm_head.weight": (16, 8),
        "fc.weight": (8, 8),  # hidden + correction, not n_capture*hidden
        "pre_fc_norm_embedding.weight": (8,),
        "pre_fc_norm_hidden.weight": (8,),
        "layers.0.input_layernorm.weight": (8,),
        "layers.0.post_attention_layernorm.weight": (8,),
        "layers.0.self_attn.q_proj.weight": (8, 8),
        "layers.0.self_attn.k_proj.weight": (8, 8),
        "layers.0.self_attn.v_proj.weight": (8, 8),
        "layers.0.self_attn.o_proj.weight": (8, 8),
        "layers.0.mlp.gate_proj.weight": (16, 8),
        "layers.0.mlp.up_proj.weight": (16, 8),
        "layers.0.mlp.down_proj.weight": (8, 16),
    }
    report = mod.audit(actual, _small_spec())
    assert "blk.0.attn_q_norm.weight" in report["missing"]
    assert "blk.0.attn_k_norm.weight" in report["missing"]
    assert "dspark.hidden_norm.weight" in report["missing"]
    assert "pre_fc_norm_embedding.weight" in report["extra"]
    assert "pre_fc_norm_hidden.weight" in report["extra"]
    assert report["shape_mismatch"]["dspark.fc.weight"] == {
        "expected": [8, 16], "actual": [8, 8]}
    assert report["ok"] is False
