"""T28 — QAT/recovery offline tests: STE math, wrapping, KD loss."""

from __future__ import annotations

import numpy as np
import torch

from bonsai_forensics import recover


class _Tiny(torch.nn.Module):
    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(dim, dim, bias=False)
        self.down_proj = torch.nn.Linear(dim, dim, bias=False)
        self.lm_head = torch.nn.Linear(dim, 8, bias=False)

    def forward(self, x):
        return self.lm_head(self.down_proj(self.q_proj(x)))


def test_ternary_ste_forward_is_ternary_and_gradients_flow() -> None:
    torch.manual_seed(0)
    w = (torch.randn(32, 256) * 0.05).requires_grad_(True)
    out = recover.ternary_ste(w)
    groups = out.detach().reshape(32, 2, 128)
    scales = groups.abs().max(dim=-1, keepdim=True).values
    ratio = groups / scales.clamp_min(1e-8)
    assert torch.allclose(ratio, ratio.round(), atol=1e-5)
    out.sum().backward()
    assert w.grad is not None and torch.isfinite(w.grad).all()
    # STE passes the identity gradient through the quantization boundary
    assert torch.allclose(w.grad, torch.ones_like(w))


def test_ternary_ste_keeps_padded_tail_exact() -> None:
    w = torch.randn(4, 100) * 0.1  # 100 is not a multiple of 128
    out = recover.ternary_ste(w)
    assert out.shape == w.shape
    assert torch.isfinite(out).all()


def test_wrap_ternary_replaces_only_target_linears() -> None:
    model = _Tiny()
    replaced = recover.wrap_ternary(model)
    assert replaced == 2
    assert isinstance(model.q_proj, recover.TernaryLinear)
    assert isinstance(model.down_proj, recover.TernaryLinear)
    assert isinstance(model.lm_head, torch.nn.Linear)
    assert not isinstance(model.lm_head, recover.TernaryLinear)


def test_freeze_non_ternary_leaves_master_weights_trainable() -> None:
    model = _Tiny()
    recover.wrap_ternary(model)
    frozen = recover.freeze_non_ternary(model)
    assert frozen >= 1  # lm_head.weight
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert sorted(trainable) == ["down_proj.weight", "q_proj.weight"]
    assert not model.lm_head.weight.requires_grad


def test_distillation_loss_zero_for_identical_logits() -> None:
    logits = torch.randn(2, 4, 16)
    loss = recover.distillation_loss(logits, logits.clone(), temperature=2.0)
    assert float(loss) < 1e-6


def test_build_batches_yields_full_windows() -> None:
    ids = np.arange(1000)
    batches = recover.build_batches(ids, seq_len=100, batch_size=2, seed=0)
    batch = next(batches)
    assert batch.shape == (2, 100)
    assert int(batch.max()) < 1000
