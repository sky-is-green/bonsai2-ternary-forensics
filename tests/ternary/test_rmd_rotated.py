"""Offline tests for the harness's rotated + STE linear (no GPU)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_RMD = Path(__file__).resolve().parents[2] / "scripts" / "pilot" / "rmd_kd.py"
_spec = importlib.util.spec_from_file_location("rmd_kd", _RMD)
rmd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rmd)


def _rotated(ste: bool, dim: int = 8):
    base = torch.nn.Linear(dim, dim, bias=False)
    rot = torch.eye(dim)
    return rmd.RotatedLinear(base, rot, None, ste=ste)


def test_rotated_linear_ste_ternarizes_the_master() -> None:
    module = _rotated(ste=True)
    x = torch.randn(2, 8)
    expected = torch.nn.functional.linear(
        torch.nn.functional.linear(x, module.rot_in), rmd.ternary_ste(module.weight)
    )
    assert torch.allclose(module.forward(x), expected, atol=1e-5)


def test_rotated_linear_without_ste_uses_raw_master() -> None:
    module = _rotated(ste=False)
    x = torch.randn(2, 8)
    expected = torch.nn.functional.linear(
        torch.nn.functional.linear(x, module.rot_in), module.weight
    )
    assert torch.allclose(module.forward(x), expected, atol=1e-5)


def test_init_group_scale_shape_and_values() -> None:
    w = torch.ones(4, 256) * 0.1
    scale = rmd._init_group_scale(w)
    assert scale.shape == (4, 2, 1)
    assert torch.allclose(scale, torch.full_like(scale, 0.1), atol=1e-6)


def test_ternary_lsq_scale_receives_gradient() -> None:
    torch.manual_seed(0)
    w = (torch.randn(16, 256) * 0.05).requires_grad_(True)
    scale = torch.nn.Parameter(rmd._init_group_scale(w))
    out = rmd.ternary_lsq(w, scale)
    out.sum().backward()
    assert w.grad is not None and torch.isfinite(w.grad).all()
    assert scale.grad is not None and torch.isfinite(scale.grad).all()
    # Deployed (detached) output is codes * scale, i.e. on the integer grid.
    ratio = out.detach().reshape(16, 2, 128) / scale.detach()
    assert torch.allclose(ratio, ratio.round(), atol=1e-4)


def test_wrap_rotated_preserves_fp_function_in_both_modes() -> None:
    from types import SimpleNamespace

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="llama", tie_word_embeddings=True,
                                          num_hidden_layers=1)
            self.q_proj = torch.nn.Linear(8, 8, bias=False)
            self.o_proj = torch.nn.Linear(8, 8, bias=False)
            self.gate_proj = torch.nn.Linear(8, 8, bias=False)
            self.up_proj = torch.nn.Linear(8, 8, bias=False)
            self.down_proj = torch.nn.Linear(8, 8, bias=False)

    torch.manual_seed(3)
    x = torch.randn(2, 8)
    for mode in ("residual", "input"):
        model = Toy()
        expected = model.down_proj(model.up_proj(model.gate_proj(x)))
        profile = rmd.infer_profile(model, "llama")
        wrapped = rmd.wrap_rotated(model, seed=1337, block=8, rotation_mode=mode,
                                   profile=profile)
        actual = model.down_proj(model.up_proj(model.gate_proj(x)))
        assert len(wrapped) == 5
        assert torch.allclose(actual, expected, atol=1e-5)


def test_ternary_ste_output_is_codes_times_original_absmean() -> None:
    torch.manual_seed(0)
    w = torch.randn(16, 256) * 0.05
    q = rmd.ternary_ste(w)
    # The scale is the absmean of the *original* weights, per 128-group.
    scale = w.reshape(16, 2, 128).abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
    ratio = q.reshape(16, 2, 128) / scale
    assert torch.allclose(ratio, ratio.round(), atol=1e-4)
    assert ratio.abs().max() <= 1.0 + 1e-4
