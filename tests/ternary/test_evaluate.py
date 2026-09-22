"""Tests for the reusable multi-region evaluation surface."""

from __future__ import annotations

import numpy as np
import pytest

from bonsai_forensics import evaluate


def test_make_regions_disjoint_and_in_bounds() -> None:
    seq = 512
    total = 100 * seq
    regions = evaluate.make_regions(total, seq, windows_per_region=4, n_regions=5)
    assert len(regions) == 5
    ranges = [r.window_range(seq) for r in regions]
    for start, stop in ranges:
        assert 0 <= start < stop <= total // seq - 2
    # disjoint and ordered
    for (_, prev_stop), (next_start, _) in zip(ranges, ranges[1:]):
        assert prev_stop <= next_start


def test_make_regions_rejects_oversized_request() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        evaluate.make_regions(10 * 512, 512, windows_per_region=4, n_regions=5)


def test_region_windows_matches_regions() -> None:
    regions = evaluate.make_regions(100 * 512, 512, windows_per_region=3, n_regions=4)
    assert evaluate.region_windows(regions, 512) == tuple(r.window_range(512) for r in regions)


def test_eval_region_validation() -> None:
    with pytest.raises(ValueError):
        evaluate.EvalRegion("bad", start_token=-1, n_windows=1)
    with pytest.raises(ValueError):
        evaluate.EvalRegion("bad", start_token=0, n_windows=0)


def test_summarize_ppls() -> None:
    stats = evaluate.summarize_ppls([2.0, 4.0])
    assert stats["n_regions"] == 2
    assert stats["mean_ppl"] == pytest.approx(3.0)
    assert stats["min_ppl"] == pytest.approx(2.0)
    assert stats["max_ppl"] == pytest.approx(4.0)
    assert stats["nats_mean"] == pytest.approx(float(np.log(np.sqrt(8.0))))


def test_add_ratios_and_markdown() -> None:
    result = {
        "regions": [
            {"name": "r0", "start_token": 0, "n_windows": 2, "ppl": 4.0, "nats": 1.386},
            {"name": "r1", "start_token": 512, "n_windows": 2, "ppl": 6.0, "nats": 1.791},
        ],
        "aggregate": {"n_regions": 2},
    }
    out = evaluate.add_ratios(result, [2.0, 3.0])
    assert out["regions"][0]["ratio"] == pytest.approx(2.0)
    assert out["aggregate"]["mean_ratio"] == pytest.approx(2.0)
    md = evaluate.result_markdown(out)
    assert "Aggregate ratio" in md
    assert "r0" in md and "r1" in md
