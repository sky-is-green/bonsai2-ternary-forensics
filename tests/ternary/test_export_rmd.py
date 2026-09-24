"""Offline tests for the rmd -> PQ2_0 export bridge (no GPU)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from bonsai_forensics import export_rmd, run_quant

_RMD = Path(__file__).resolve().parents[2] / "scripts" / "pilot" / "rmd_kd.py"
_spec = importlib.util.spec_from_file_location("rmd_kd", _RMD)
rmd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rmd)


class _Toy(torch.nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="llama", tie_word_embeddings=True,
                                      num_hidden_layers=1)
        self.q_proj = torch.nn.Linear(dim, dim, bias=True)
        self.k_proj = torch.nn.Linear(dim, dim, bias=True)
        self.v_proj = torch.nn.Linear(dim, dim, bias=True)
        self.o_proj = torch.nn.Linear(dim, dim, bias=True)
        self.gate_proj = torch.nn.Linear(dim, dim, bias=True)
        self.up_proj = torch.nn.Linear(dim, dim, bias=True)
        self.down_proj = torch.nn.Linear(dim, dim, bias=True)


def test_unabsorb_inverts_wrap_rotated() -> None:
    """The trained master is the absorbed weight; un-absorb recovers the primal."""
    for mode in ("residual", "input"):
        torch.manual_seed(0)
        model = _Toy(dim=8)
        original = {name: value.detach().clone()
                    for name, value in model.named_parameters()}
        profile = rmd.infer_profile(model, "llama")
        rmd.wrap_rotated(model, seed=1337, block=8, rotation_mode=mode, profile=profile)
        state = export_rmd.trained_hf_state(model)
        assert set(state) == set(original), (mode, set(state) ^ set(original))
        for name, value in original.items():
            assert np.allclose(state[name], value.numpy(), atol=1e-5), (mode, name)


def _quant_config(axis: str) -> dict:
    return {
        "rotation": {"seed": 1337, "axis": axis},
        "quant": {"group_size": 128, "refine_iters": 0, "damp": 0.01,
                  "act_order": False, "block_size": 128},
    }


def test_last_axis_convention_differs_for_output_projections() -> None:
    """PQ2_0 rotates the last axis; the legacy role map rotates the output axis."""
    w = np.random.default_rng(0).standard_normal((128, 256))
    name = "model.layers.0.self_attn.o_proj.weight"
    legacy = run_quant._process_tensor(name, w, None, _quant_config("role"))
    prism = run_quant._process_tensor(name, w, None, _quant_config("last"))
    assert not np.array_equal(legacy["codes"], prism["codes"])

    from bonsai_forensics import quant, rotation
    rots = rotation.rotations_for(256, 1337)
    expected = quant.quantize_rtn_absmean(rotation.absorb_input(w, rots), 128)
    assert np.array_equal(prism["codes"], expected.codes)
    assert np.allclose(prism["scales"], expected.scales)


def test_export_pq2_0_writes_a_packed_artifact(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    state = {
        "model.embed_tokens.weight": rng.standard_normal((128, 128)).astype(np.float32),
        "model.norm.weight": np.ones(128, dtype=np.float32),
    }
    for stem in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        state[f"model.layers.0.self_attn.{stem}.weight"] = (
            rng.standard_normal((128, 128)).astype(np.float32))
    out = tmp_path / "model.gguf"
    result = export_rmd.export_pq2_0(state, out, rot_seed=1337)
    assert out.is_file() and out.stat().st_size > 0
    assert result.artifact == out
    assert len(result.processed) == len(state)


def test_export_rejects_refined_group_size(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="refine_iters 0"):
        export_rmd.export_pq2_0({}, tmp_path / "x.gguf", rot_seed=1337, refine_iters=4)
