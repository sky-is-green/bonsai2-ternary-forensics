"""PlateauDecay — drift/plateau-triggered LR decay (management layer)."""

from __future__ import annotations

import pytest

from bonsai_forensics.schedule import PlateauDecay


def test_no_decay_while_improving():
    d = PlateauDecay(patience=2000)
    for step, ratio in [(500, 2.0), (1000, 1.5), (1500, 1.2)]:
        assert d.observe(step, ratio) == 1.0
    assert d.events == 0
    assert d.best == 1.2


def test_plateau_decays_after_patience():
    d = PlateauDecay(patience=2000, factor=0.5)
    assert d.observe(500, 1.2) == 1.0         # clock starts
    assert d.observe(2000, 1.25) == 1.0       # worse, patience not yet elapsed
    assert d.observe(2500, 1.30) == 0.5       # 2000 since best -> decay
    assert d.events == 1


def test_decay_is_multiplicative_and_bounded_by_cooldown():
    d = PlateauDecay(patience=1000, factor=0.5, cooldown=1000)
    d.observe(0, 1.0)
    assert d.observe(1000, 1.1) == 0.5        # first event
    assert d.observe(1500, 1.1) == 0.5        # cooldown blocks a second
    assert d.observe(2000, 1.1) == 0.25       # second event
    assert d.events == 2


def test_max_events_caps_decay():
    d = PlateauDecay(patience=100, factor=0.5, max_events=2, cooldown=1)
    d.observe(0, 1.0)
    for step in range(100, 1000, 100):
        d.observe(step, 1.1)
    assert d.events == 2
    assert d.multiplier == pytest.approx(0.25)


def test_improvement_resets_the_clock():
    d = PlateauDecay(patience=1000, factor=0.5)
    d.observe(0, 1.0)
    d.observe(2000, 1.1)                      # would decay
    assert d.events == 1
    d.observe(3000, 0.9)                      # new best resets
    assert d.observe(3500, 0.95) == 0.5       # only 500 since best -> no extra decay
    assert d.events == 1


def test_drift_decays_immediately_even_without_patience():
    d = PlateauDecay(patience=0, factor=0.5, drift_eps=0.02, cooldown=1000)
    d.observe(0, 1.0)
    assert d.observe(500, 1.05) == 0.5        # +5% > 2% -> immediate decay
    assert d.observe(600, 1.05) == 0.5        # cooldown blocks a second event
    assert d.events == 1


def test_min_delta_ignores_tiny_improvements():
    d = PlateauDecay(patience=100, factor=0.5, min_delta=0.01, cooldown=1)
    d.observe(0, 1.0)
    d.observe(100, 0.995)                     # not better by min_delta
    assert d.events == 1                      # does not reset the clock
    assert d.best == 1.0


def test_warmup_suppresses_early_noise():
    d = PlateauDecay(patience=100, factor=0.5, drift_eps=0.05, warmup=1000, cooldown=1)
    d.observe(0, 1.0)
    # a +19% bounce during exploration must not decay
    assert d.observe(500, 1.19) == 1.0
    assert d.observe(900, 1.30) == 1.0
    assert d.events == 0
    # past warmup the same drift does fire
    assert d.observe(1100, 1.19) == 0.5
    assert d.events == 1


def test_warmup_still_tracks_best():
    d = PlateauDecay(patience=100, factor=0.5, warmup=1000, cooldown=1000)
    d.observe(100, 1.0)
    assert d.best == 1.0                   # best tracked through warmup
    assert d.observe(900, 1.05) == 1.0      # still in warmup -> suppressed
    assert d.observe(1000, 1.05) == 0.5     # warmup ends; plateau long overdue -> decay
    assert d.events == 1


def test_apply_respects_floor():
    d = PlateauDecay(patience=10, factor=0.5, cooldown=1)
    d.observe(0, 1.0)
    for step in range(10, 200, 10):
        d.observe(step, 1.1)
    assert d.multiplier < 0.01
    assert d.apply(base_lr=5e-5, floor=5e-6) == 5e-6
    assert d.apply(base_lr=5e-5) == pytest.approx(5e-5 * d.multiplier)
