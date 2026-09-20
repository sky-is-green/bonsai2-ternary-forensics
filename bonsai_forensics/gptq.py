"""T4 — GPTQ error compensation (spec §2.3, ADR-4).

Blockwise column-wise quantization with Hessian `H = XᵀX / N`, damping,
optional activation-order traversal, and real GGUF-group scales: scales are
`(out, in/group_size)` per-row absmean+LS values (the T3 recipe, vectorized in
torch), so `GptqResult.codes/scales` can be packed by T6 without a re-encode.
A torch twin of the group kernel is kept here so a 27B run never round-trips to
numpy per column; a consistency test pins both implementations to identical
outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from bonsai_forensics.quant import REFINE_ITERS

SPEC_SHA256 = "0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b"

DEFAULT_DAMP_FRACTION = 0.01
DEFAULT_BLOCK_SIZE = 128


def _round_half_away(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.floor(x.abs() + 0.5)


def _clip_codes(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(_round_half_away(x), -1.0, 1.0)


def group_scales_torch(w: torch.Tensor, group_size: int = 256, refine_iters: int = REFINE_ITERS) -> torch.Tensor:
    """Per-row group scales `(out, n_groups)` via absmean init + LS refine."""
    if w.ndim != 2:
        raise ValueError("expected a 2-D weight")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    work_dtype = w.dtype if w.dtype in (torch.float32, torch.float64) else torch.float32
    w = w.to(work_dtype)
    out_features, in_features = w.shape
    n_groups = math.ceil(in_features / group_size)
    padded = torch.zeros(out_features, n_groups * group_size, dtype=work_dtype, device=w.device)
    padded[:, :in_features] = w
    groups = padded.view(out_features, n_groups, group_size)

    absmean = groups.abs().mean(dim=-1, keepdim=True)
    s0 = torch.where(absmean > 0, absmean, torch.ones_like(absmean))
    t = _clip_codes(groups / s0)
    best_s = s0.squeeze(-1).clone()
    best_t = t.clone()
    best_r = ((groups - s0 * t) ** 2).sum(dim=-1)

    s = s0
    for _ in range(refine_iters):
        tt = (t * t).sum(dim=-1, keepdim=True)
        tg = (t * groups).sum(dim=-1, keepdim=True)
        step = tg / torch.where(tt > 0, tt, torch.ones_like(tt))
        s = torch.where((tt > 0) & (tg > 0), step, s)
        t = _clip_codes(groups / s)
    if refine_iters > 0:
        tt = (t * t).sum(dim=-1, keepdim=True)
        tg = (t * groups).sum(dim=-1, keepdim=True)
        step = tg / torch.where(tt > 0, tt, torch.ones_like(tt))
        s = torch.where((tt > 0) & (tg > 0), step, s)

    refined_r = ((groups - s * t) ** 2).sum(dim=-1)
    take_refined = refined_r <= best_r
    scales = torch.where(take_refined, s.squeeze(-1), best_s)
    chosen_t = torch.where(take_refined.unsqueeze(-1), t, best_t)
    empty = (chosen_t == 0).all(dim=-1)
    return torch.where(empty, torch.zeros_like(scales), scales)


def hessian_from_activations(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """`H = XᵀX / N` over calibration activations `X: (N, in_features)`."""
    x = torch.as_tensor(x)
    if x.ndim != 2:
        raise ValueError(f"activations must be 2-D, got shape {tuple(x.shape)}")
    h = x.transpose(0, 1) @ x
    return h / x.shape[0] if normalize else h


def _damped_cholesky_inverse(hessian: torch.Tensor, damp: float) -> torch.Tensor:
    if damp < 0:
        raise ValueError("damp must be >= 0")
    n = hessian.shape[0]
    mean_diag = torch.diagonal(hessian).mean()
    if not bool(torch.isfinite(mean_diag)) or bool(mean_diag <= 0):
        raise ValueError("Hessian diagonal is zero/NaN; cannot damp")
    damped = hessian + (damp * mean_diag) * torch.eye(n, dtype=hessian.dtype, device=hessian.device)
    chol = torch.linalg.cholesky(damped)
    return torch.cholesky_inverse(chol)


@dataclass
class GptqResult:
    codes: torch.Tensor
    scales: torch.Tensor
    group_size: int
    permutation: torch.Tensor | None = None

    def dequantize(self) -> torch.Tensor:
        expanded = self.scales.repeat_interleave(self.group_size, dim=-1)
        return self.codes.to(self.scales.dtype) * expanded[..., : self.codes.shape[-1]]


def gptq_quantize(
    w: torch.Tensor,
    hessian: torch.Tensor,
    *,
    group_size: int = 256,
    damp: float = DEFAULT_DAMP_FRACTION,
    act_order: bool = False,
    block_size: int = DEFAULT_BLOCK_SIZE,
    refine_iters: int = REFINE_ITERS,
    device: str | torch.device | None = None,
) -> GptqResult:
    """Quantize `W: (out, in)` with GPTQ error compensation using `H: (in, in)`.

    `device` moves the whole computation (Cholesky, inverts, error propagation)
    to GPU when available; the T10 canary runs GPTQ on the RX 7900 XT and the
    CPU path stays the deterministic default for tests.
    """
    w = torch.as_tensor(w)
    hessian = torch.as_tensor(hessian)
    if device is not None:
        target = torch.device(device)
        w = w.to(target)
        hessian = hessian.to(target)
    if w.ndim != 2:
        raise ValueError(f"W must be 2-D, got shape {tuple(w.shape)}")
    if hessian.shape != (w.shape[1], w.shape[1]):
        raise ValueError(f"Hessian {tuple(hessian.shape)} does not match W input dim {w.shape[1]}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    work_dtype = w.dtype if w.dtype in (torch.float32, torch.float64) else torch.float32
    w = w.to(work_dtype)
    hessian = hessian.to(work_dtype)

    out_features, in_features = w.shape
    permutation = (
        torch.argsort(torch.diagonal(hessian), descending=True, stable=True) if act_order else None
    )
    order = permutation if permutation is not None else torch.arange(in_features)
    h = hessian[order][:, order]
    h_inv = _damped_cholesky_inverse(h, damp)

    scales = group_scales_torch(w, group_size=group_size, refine_iters=refine_iters)
    work = w[:, order].clone()
    codes = torch.zeros_like(work, dtype=torch.int8)
    n_groups = scales.shape[-1]
    group_index = torch.clamp(order // group_size, max=max(n_groups - 1, 0))

    for i1 in range(0, in_features, block_size):
        i2 = min(i1 + block_size, in_features)
        w_block = work[:, i1:i2].clone()
        q_block = torch.zeros_like(w_block)
        err_block = torch.zeros_like(w_block)
        h_inv_block = h_inv[:, i1:i2]
        for i in range(i2 - i1):
            row = i1 + i
            column = w_block[:, i]
            scale = scales[:, int(group_index[row])]
            local = _clip_codes(column / scale.clamp(min=torch.finfo(work_dtype).tiny))
            q = local * scale
            codes[:, row] = local.to(torch.int8)
            q_block[:, i] = q
            err = (column - q) / h_inv_block[row, i]
            w_block[:, i:] = w_block[:, i:] - err.unsqueeze(1) * h_inv_block[row, i:].unsqueeze(0)
            err_block[:, i] = err
        if i2 < in_features:
            work[:, i2:] = work[:, i2:] - err_block @ h_inv[i1:i2, i2:]

    if permutation is not None:
        codes = codes[:, torch.argsort(permutation)]
    return GptqResult(codes=codes, scales=scales, group_size=group_size, permutation=permutation)
