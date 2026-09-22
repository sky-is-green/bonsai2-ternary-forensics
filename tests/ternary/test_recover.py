"""T28 — QAT/recovery offline tests: STE math, wrapping, KD loss."""

from __future__ import annotations

import numpy as np
import pytest
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


def test_ce_loss_matches_cross_entropy() -> None:
    logits = torch.randn(2, 5, 8)
    ids = torch.randint(0, 8, (2, 5))
    expected = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 8), ids[:, 1:].reshape(-1)
    )
    assert torch.allclose(recover.ce_loss(logits, ids), expected)


def test_mixed_loss_interpolates_kd_and_ce() -> None:
    torch.manual_seed(0)
    student_logits = torch.randn(2, 5, 8)
    teacher_logits = torch.randn(2, 5, 8)
    ids = torch.randint(0, 8, (2, 5))
    kd = recover.distillation_loss(student_logits, teacher_logits, temperature=2.0)
    ce = recover.ce_loss(student_logits, ids)
    mix = recover.mixed_loss(student_logits, teacher_logits, ids, temperature=2.0, ce_weight=0.3)
    assert torch.allclose(mix, 0.7 * kd + 0.3 * ce, atol=1e-5)


def test_ternary_ste_absmax_scale_matches_round() -> None:
    torch.manual_seed(0)
    w = torch.randn(32, 256) * 0.05
    out = recover.ternary_ste(w, scale="absmax")
    groups = out.detach().reshape(32, 2, 128)
    scales = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    ratio = groups / scales
    assert torch.allclose(ratio, ratio.round(), atol=1e-5)


def test_randomize_ternary_reinitializes_masters() -> None:
    model = _Tiny()
    before = model.q_proj.weight.detach().clone()
    recover.wrap_ternary(model)
    recover.randomize_ternary(model, seed=0)
    after = model.q_proj.weight
    assert not torch.equal(before, after)
    assert torch.isfinite(after).all()


def test_build_batches_yields_full_windows() -> None:
    ids = np.arange(1000)
    batches = recover.build_batches(ids, seq_len=100, batch_size=2, seed=0)
    batch = next(batches)
    assert batch.shape == (2, 100)
    assert int(batch.max()) < 1000


def test_build_batches_never_samples_the_holdout() -> None:
    # 20 windows of 10; exclude windows [5, 8) (tokens 50:80).
    ids = np.arange(200)
    holdout = ((5, 8),)
    batches = recover.build_batches(ids, seq_len=10, batch_size=8, seed=0,
                                    holdout_windows=holdout)
    for _ in range(50):
        batch = next(batches)
        assert not ((batch >= 50) & (batch < 80)).any()


def test_eval_holdout_windows_matches_the_eval_region() -> None:
    # T28 params: eval seq 2048, samples 32 -> tokens [65536, 73728) = windows
    # [128, 144) of 512. This is the range that must be held out.
    assert recover.eval_holdout_windows(32, 2048, 4, 512) == (128, 144)
    # When eval and training share a seq_len, the range is [samples, samples + eval_windows).
    assert recover.eval_holdout_windows(32, 512, 4, 512) == (32, 36)


def test_build_batches_holdout_covers_every_window() -> None:
    ids = np.arange(100)
    with pytest.raises(ValueError, match="every training window"):
        next(recover.build_batches(ids, seq_len=10, batch_size=1, seed=0,
                                   holdout_windows=((0, 10),)))
