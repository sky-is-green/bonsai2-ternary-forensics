"""Managed decay for training — drift/plateau-triggered LR reduction.

Rationale (2026-09-22). The 20000-step rotation + STE run under a constant LR
peaked at **1.1603** mean ratio at step 16000 and then *degraded* to **1.2930**
by step 20000. The tail diverged because nothing in the recipe could react to
the held-out metric: a fixed schedule is a guess made before the run starts.

This is the same design vocabulary as the retention layer's Sharp Decay Matrix
(`splinter/retention/decay.py`, `remembrance.py`), applied to the learning rate
instead of to context chunks:

- **event-driven**, triggered by observed stagnation/drift rather than a step count;
- **multiplicative** per event (the remembrance ladder is ``*= 1.8 + 0.3*ts``);
- **bounded**: capped event count and a floor, the analogue of ``age_factor``
  being clamped at 3.0 — over-steep decay is destructive (the retention config
  walked ``decay_multiplier_init`` from 1.8 down to 1.1 for that reason).

Pure standard library, so it is unit-testable without torch or a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PlateauDecay:
    """Cumulative LR multiplier driven by a held-out metric.

    ``observe(step, ratio)`` is called at every evaluation and returns the
    cumulative multiplier to apply to the base learning rate. It never
    increases: the only events are decays.

    Args:
        patience: steps without a new best before a plateau decay (0 disables).
        factor: multiplicative drop applied per event.
        max_events: cap on decay events (0 = unlimited).
        min_delta: an improvement must beat ``best - min_delta`` to reset the clock.
        drift_eps: > 0 decays immediately when the metric worsens by this
            fraction of the best (the training analogue of the drift prior:
            penalise the drift, do not reward it).
        cooldown: minimum steps between events (defaults to ``patience``, or 1).
        warmup: no decay before this step. The early trajectory is legitimately
            noisy (the 20k run jumped +19% between adjacent evals at step 5000)
            while still trending down, so a reactive policy must not fire during
            exploration. Best/last-improvement are still tracked during warmup.
    """

    patience: int = 0
    factor: float = 0.5
    max_events: int = 0
    min_delta: float = 0.0
    drift_eps: float = 0.0
    cooldown: int = 0
    warmup: int = 0

    _best: Optional[float] = field(default=None, init=False)
    _last_improve: int = field(default=0, init=False)
    _next_ok: int = field(default=0, init=False)
    _events: int = field(default=0, init=False)
    multiplier: float = field(default=1.0, init=False)
    history: list = field(default_factory=list, init=False)

    # -- read-only views ---------------------------------------------------
    @property
    def best(self) -> Optional[float]:
        return self._best

    @property
    def events(self) -> int:
        return self._events

    @property
    def last_improve(self) -> int:
        return self._last_improve

    # -- core --------------------------------------------------------------
    def observe(self, step: int, ratio: float) -> float:
        """Feed one held-out measurement; return the cumulative multiplier."""
        self.history.append((step, ratio))

        if self._best is None or ratio < self._best - self.min_delta:
            self._best = ratio
            self._last_improve = step
            return self.multiplier

        if step < self.warmup:
            return self.multiplier
        if self.max_events and self._events >= self.max_events:
            return self.multiplier
        if step < self._next_ok:
            return self.multiplier

        drifted = (
            self.drift_eps > 0
            and self._best is not None
            and ratio > self._best * (1.0 + self.drift_eps)
        )
        plateaued = (
            self.patience > 0
            and step - self._last_improve >= self.patience
        )
        if drifted or plateaued:
            self.multiplier *= self.factor
            self._events += 1
            self._next_ok = step + max(self.cooldown or self.patience or 1, 1)
        return self.multiplier

    def apply(self, base_lr: float, floor: float = 0.0) -> float:
        """The learning rate for the current multiplier, never below ``floor``."""
        return max(float(floor), float(base_lr) * self.multiplier)
