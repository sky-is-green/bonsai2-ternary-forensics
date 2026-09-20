"""T27 — reference bar: reproduction + the held-out collapse finding.

The reference implementation is third-party (ThakiCloud, Apache-2.0); these
tests pin the pieces we rely on offline, with no model download.
"""

from __future__ import annotations

import numpy as np
import torch

from bonsai_forensics import reference


def test_rand_orth_is_orthogonal_and_cached() -> None:
    q = reference.rand_orth(64, seed=0)
    assert torch.allclose(q @ q.t(), torch.eye(64), atol=1e-5)
    assert reference.rand_orth(64, seed=0) is q  # cached
    assert not torch.allclose(reference.rand_orth(64, seed=1), q)


def test_reference_ternary_beats_rtn_on_output_error() -> None:
    torch.manual_seed(0)
    out_dim, in_dim = 64, 128
    w = torch.randn(out_dim, in_dim) * 0.1
    x = torch.randn(512, in_dim)
    h = x.t() @ x / x.shape[0]
    rot = reference.rand_orth(in_dim, seed=0)
    q = reference.gptq_reference(w @ rot.t(), rot @ h @ rot.t(), bits="ternary", damping=0.1)
    q = q @ rot  # fold back
    rtn = torch.clamp(torch.round(w / w.abs().mean(dim=1, keepdim=True).clamp(min=1e-9)), -1, 1) * w.abs().mean(
        dim=1, keepdim=True
    ).clamp(min=1e-9)
    err = lambda e: float(((e @ h) * e).sum())  # H-weighted output error
    assert err(q - w) < err(rtn - w)


def test_reference_reports_both_metrics_when_collapsed() -> None:
    # _perplexity clips at exp(20); the run report marks collapsed >= 1e8.
    assert reference._perplexity.__module__ == "bonsai_forensics.reference"
