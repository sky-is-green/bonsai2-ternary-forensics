"""Decision-margin probe — does ternarization cost confidence, not just accuracy?

Motivation (2026-09-22). Prism's whitepaper reports aggregate benchmark
retention of 98.2% for Bonsai 2 27B, but ~75% on long-horizon agentic
benchmarks (Terminal-Bench 2.1 52.8 vs 69.7; SWE-bench Verified 60.8 vs 80.6,
same harness, same agent scaffold). The community failure reports describe
trajectory instability, loops, and re-introduced errors rather than uniformly
worse answers.

The hypothesis this module tests: aggressive ternarization reduces the
*confidence margin* between competing next-token decisions before it costs
much accuracy. A model can hold its mean loss while becoming less decisive,
and per-step indecision compounds over a long trajectory.

`compare()` is pure numpy so the metric is unit-testable without torch. The
headline number is `margin_retention` on the *decisive* positions (where the
teacher was confident), because those are the ones a trajectory depends on.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.maximum(p, EPS))).sum(axis=-1)


def _top2_margin(p: np.ndarray) -> np.ndarray:
    """p(1st) - p(2nd): confidence the top choice beats the runner-up."""
    part = np.partition(p, -2, axis=-1)[..., -2:]
    hi, lo = part[..., 1], part[..., 0]
    return hi - lo


def compare(teacher_logits: np.ndarray, student_logits: np.ndarray,
            decisive_quantile: float = 0.9) -> dict:
    """Per-position teacher/student comparison.

    Both inputs are `(..., vocab)` logits; leading dims are treated as the
    position axis. Returns arrays so the caller can stratify.
    """
    t = np.asarray(teacher_logits, dtype=np.float64)
    s = np.asarray(student_logits, dtype=np.float64)
    if t.shape != s.shape:
        raise ValueError(f"shape mismatch: {t.shape} vs {s.shape}")
    if t.shape[-1] < 2:
        raise ValueError("need at least two vocabulary entries to measure a margin")

    p = softmax(t)
    q = softmax(s)
    margin_t = _top2_margin(p)
    margin_s = _top2_margin(q)
    return {
        "kl": (p * (np.log(np.maximum(p, EPS)) - np.log(np.maximum(q, EPS)))).sum(-1),
        "entropy_teacher": _entropy(p),
        "entropy_student": _entropy(q),
        "margin_teacher": margin_t,
        "margin_student": margin_s,
        "margin_delta": margin_s - margin_t,
        "agree": (p.argmax(-1) == q.argmax(-1)).astype(np.float64),
    }


def _stats(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return {"n": 0, "mean": None, "median": None, "p95": None}
    return {"n": int(v.size), "mean": float(v.mean()),
            "median": float(np.median(v)), "p95": float(np.percentile(v, 95))}


def summarize(stats: dict, decisive_quantile: float = 0.9) -> dict:
    """Aggregate a `compare()` result, overall and on the decisive positions."""
    mt = np.asarray(stats["margin_teacher"]).ravel()
    ms = np.asarray(stats["margin_student"]).ravel()
    kl = np.asarray(stats["kl"]).ravel()
    et = np.asarray(stats["entropy_teacher"]).ravel()
    es = np.asarray(stats["entropy_student"]).ravel()
    agree = np.asarray(stats["agree"]).ravel()

    if mt.size == 0:
        raise ValueError("no positions to summarize")

    def block(mask: np.ndarray) -> dict:
        n = int(mask.sum())
        if n == 0:
            return {"n": 0}
        mt_m, ms_m = mt[mask].mean(), ms[mask].mean()
        return {
            "n": n,
            "margin_teacher_mean": float(mt_m),
            "margin_student_mean": float(ms_m),
            "margin_retention": float(ms_m / mt_m) if mt_m > 0 else None,
            "kl_mean": float(kl[mask].mean()),
            "agree_rate": float(agree[mask].mean()),
            "entropy_teacher_mean": float(et[mask].mean()),
            "entropy_student_mean": float(es[mask].mean()),
        }

    all_mask = np.ones_like(mt, dtype=bool)
    decisive = mt >= np.quantile(mt, decisive_quantile)
    return {
        "n_positions": int(mt.size),
        "overall": block(all_mask),
        "decisive": block(decisive),
        "decisive_quantile": float(decisive_quantile),
        "kl": _stats(kl),
        "entropy_teacher": _stats(et),
        "entropy_student": _stats(es),
        "margin_delta": _stats(ms - mt),
        "frac_positions_less_decisive": float((ms < mt).mean()),
        "top1_agreement": float(agree.mean()),
    }
