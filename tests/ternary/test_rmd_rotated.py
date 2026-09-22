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


def test_ternary_ste_output_is_codes_times_original_absmean() -> None:
    torch.manual_seed(0)
    w = torch.randn(16, 256) * 0.05
    q = rmd.ternary_ste(w)
    # The scale is the absmean of the *original* weights, per 128-group.
    scale = w.reshape(16, 2, 128).abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
    ratio = q.reshape(16, 2, 128) / scale
    assert torch.allclose(ratio, ratio.round(), atol=1e-4)
    assert ratio.abs().max() <= 1.0 + 1e-4
