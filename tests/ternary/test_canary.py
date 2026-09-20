"""T10 — canary report schema + live validation.

The canary run itself needs a model download and real compute, so it is
live-gated: set `TBR_CANARY_REPORT` to a `canary-report.json` produced by
`bonsai_forensics.canary`. The offline test pins the schema and the pass
criterion against a synthetic report.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from bonsai_forensics import canary

LIVE_REPORT = os.environ.get("TBR_CANARY_REPORT", "")


def _synthetic_report() -> dict:
    return {
        "task": "T10",
        "spec_hash": canary.rq.SPEC_SHA256,
        "model": "Qwen/Qwen3-1.7B",
        "corpus_sha256": "0" * 64,
        "config": {"samples": 1, "seq_len": 8, "eval_windows": 1},
        "timing_s": {"capture": 0.0, "quantize": 0.0},
        "tensors": {"total": 2, "gptq_ternary": 2},
        "artifact": {"path": "a.gguf", "sha256": "1" * 64},
        "eval": {"kld_ours": 0.1, "kld_naive": 0.2, "kld_improvement": 0.5},
        "replaced": {"ours": {"replaced": 1, "skipped": 0}},
        "verdict": {"pass": True, "criterion": "ours KLD <= naive RTN KLD on held-out windows"},
    }


def test_validate_report_accepts_a_good_report() -> None:
    canary.validate_report(_synthetic_report())


@pytest.mark.parametrize("key", canary.REPORT_KEYS)
def test_validate_report_rejects_missing_keys(key: str) -> None:
    report = _synthetic_report()
    report.pop(key)
    with pytest.raises(ValueError):
        canary.validate_report(report)


def test_validate_report_rejects_a_failed_verdict() -> None:
    report = _synthetic_report()
    report["verdict"]["pass"] = False
    with pytest.raises(ValueError):
        canary.validate_report(report)


def test_validate_report_rejects_rtn_only_run() -> None:
    report = _synthetic_report()
    report["tensors"]["gptq_ternary"] = 0
    with pytest.raises(ValueError):
        canary.validate_report(report)


@pytest.mark.skipif(not LIVE_REPORT, reason="set TBR_CANARY_REPORT=/path/to/canary-report.json")
def test_live_canary_report() -> None:
    report = json.loads(Path(LIVE_REPORT).read_text(encoding="utf-8"))
    canary.validate_report(report)
    assert report["eval"]["kld_ours"] <= report["eval"]["kld_naive"]


def test_recover_weight_inverts_both_absorbed_edges() -> None:
    """T10 canary: recovery must undo `Rᵀ` exactly, with γ removed."""
    import numpy as np

    from bonsai_forensics import rotation as rot

    rng = np.random.default_rng(5)
    d = 128
    gamma = 0.5 + rng.random(d)
    w_in = rng.standard_normal((d, d)) * 0.05
    w_out = rng.standard_normal((d, d)) * 0.05
    rots = rot.rotations_for(d, 1337)

    stored_in = rot.absorb_input(w_in * gamma, rots)
    got_in = canary.recover_weight(
        "rot_input", {"kind": "f16", "data": stored_in.astype(np.float16)}, gamma, 1337
    )
    assert np.allclose(got_in, w_in, atol=1e-3)

    stored_out = rot.absorb_output(w_out, rots)
    got_out = canary.recover_weight(
        "rot_output", {"kind": "f16", "data": stored_out.astype(np.float16)}, None, 1337
    )
    assert np.allclose(got_out, w_out, atol=1e-3)
