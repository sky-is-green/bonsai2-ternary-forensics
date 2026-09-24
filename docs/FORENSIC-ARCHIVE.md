# Forensic archive — the Mirror-Descent question, settled

**Status:** archived 2026-09-24. This repo is frozen read-only; this file records
the conclusions the investigation set out to reach.

## The question

Does Ternary Bonsai 2's training use the large-`q` Mirror Descent / regularized
MD (RMD) recipe of Azizan, Lale & Hassibi (arXiv:1906.03830, 2202.10788; and US
patent application 20260220467 for the quantization case) as its *mechanism*?

## The answer

**No, not as a mechanism.** Mirror Descent is part of the *intellectual lineage*
— a regularizer-implicit optimizer that can move a model along the
overparameterized solution manifold while preserving the function — and the
binary one-bit case fits it neatly (a single magnitude shell, with the sign
choosing the side). But nothing in our experiments supports it as the algorithm
Bonsai 2 runs. The mechanism our evidence supports is the conventional one:
**rotation inside the loop + ternary QAT/STE + KD + a carefully managed
schedule.**

## Evidence

1. **The mirror map fails inside the ternary loop.** Tested directly on the
   rotated masters (the coordinates that get ternarized), `--update md q=8/q=16`
   scored 23.01× against a 2.93× STE control in the deployed metric, worse at
   every checkpoint, and 8.54× once retested in the rotated basis
   (`docs/EXPERIMENTS.md`).
2. **Shape mismatch, not a tuning problem.** In magnitude the mirror map forms a
   single shell at `|w*| = η^(1/(q-1))`. Binary wants mass at one magnitude;
   ternary wants mass at `{0, ±s}`, i.e. **bimodal** in magnitude. A one-well
   force drains the zero shelf. The ternary zero does not need to be a third
   attractor the optimizer creates — it can be the quantizer's central band.
3. **The zero is the quantizer's band.** Measured `zero_frac = 0.328` on the
   released 27B, uniform to ±0.0006 across attention / MLP / linear-attn /
   embedding, and predicted by the absmean rule from the weight distribution's
   shape alone (~0.309 near-Gaussian, ~0.328 at kurtosis ≈ 4.5). No tuned knob
   (`docs/RETENTION-VS-SCALE.md`, `docs/HADAMARD-VERIFICATION.md`).
4. **The ~92% / 8% split is boundary placement, not an optimizer signature.**
   Rotate-then-round (RTN) reproduces 0.896–0.948 of Prism's released trits; the
   residual single-digit percent is where training crosses decision boundaries.
   That is evidence for QAT doing functional work — not for a particular
   optimizer.
5. **What actually moves the clean number** (multi-region, held-out): rotation
   inside the loop (unrotated STE 2.09× → rotated 1.37× at 5000 steps),
   length (1.2152× at 10000), and managed decay (best 1.1373× / 87.9% at 16000;
   constant LR diverges past 16000). Alternating projection and magnitude gating
   added nothing (the one alternating-projection "win" was an eval-accounting
   ghost; gating was inside the region noise).

## Kept caveats

- Retention is scale-dependent: our clean numbers are at 1.7B, where Prism's own
  table is ~85–88%, not the 97–98% that is a 27B figure. The canary is a scaling
  probe, not a like-for-like 27B result.
- A *mixture* prior with mass at both `0` and `s` remains the natural untested
  variant the shape argument suggests; it was not needed.

## What else this repo established

- **Export parity:** the rmd → PQ2_0 bridge round-trips on real checkpoints
  (cosine ≥ 0.999998, MAE ~1e-9; `docs/EXPORT-PARITY.md`).
- **DSpark track (Path 1):** a fork-compatible Bonsai-2 DSpark drafter was built
  and run end-to-end — target-embed/head init, torch trunk matching
  `src/models/dspark.cpp` (forward parity verified), first real accept-rate
  measured against the released 27B target (`docs/DSPARK-PATH1-PLAN.md`,
  `DSPARK-EXPORT-AUDIT.md`). Acceptance is quality-bound (draft top-1 ≈ depth-1
  accept) and the fork's host-side Markov correction dominates wall time; both
  are documented next steps, not blockers.

## Shelved: the DSpark drafter track (2026-09-24)

The fork-compatible Bonsai-2 DSpark drafter runs end to end, but it is
**shelved, not pursued**. Diagnosis: acceptance is **data/domain-bound**. The
draft is distilled on UltraChat and collapses on the harness's code/reasoning
prompts (drafted blocks degenerate to `1`, `0`, `101010;`); context length is
ruled out (offline top-1 flat across 32–256) and the runtime path is faithful
(depth-1 accept equals the draft's first-token accuracy, and only 13% of misses
have the target token anywhere in the drafted block). Separately, the runtime's
host-side Markov correction dominates wall time on ROCm, so speedup stays below
1 independently of quality. Reaching a *working* drafter needs on-policy data at
scale, likely more capacity, and a ROCm correction path, i.e. a multi-day
project with uncertain payoff. The record stands as a negative result.
See [`DSPARK-PATH1-PLAN.md`](DSPARK-PATH1-PLAN.md) and
[`DSPARK-EXPORT-AUDIT.md`](DSPARK-EXPORT-AUDIT.md).
