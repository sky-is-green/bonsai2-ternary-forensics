# Retention vs scale — Prism published numbers vs our recipe

**Why this exists.** The mission bar (≥97% retention) is Prism's **27B** number.
We have been measuring a **1.7B** canary against it. Retention in Prism's own
released tables is strongly scale-dependent, so "our canary is short of 97%" is
not a valid negative — it may be measuring against a bar that does not exist at
that scale. The canary's real job is to show our recipe **scales like Prism's**.

## Prism's published retention (ternary vs its own FP16 base)

### 6-benchmark suite — MMLU-Redux, MuSR, GSM8K, HumanEval+, IFEval, BFCLv3
Source: `ternary-bonsai-8b-whitepaper.pdf`, Table 7.

| model | FP base | ternary | retention |
|---|---|---|---|
| Ternary Bonsai 1.7B | 66.57 | 58.47 | **87.8%** |
| Ternary Bonsai 4B | 77.10 | 70.65 | **91.6%** |
| Ternary Bonsai 8B | 79.30 | 75.48 | **95.2%** |

### 10-benchmark suite
Source: same whitepaper, Tables 8 (8B), 9 (4B), 10 (1.7B).

| model | FP base | ternary | retention |
|---|---|---|---|
| Ternary Bonsai 1.7B | 58.24 | 49.58 | **85.1%** |
| Ternary Bonsai 4B | 68.31 | 62.83 | **92.0%** |
| Ternary Bonsai 8B | 71.02 | 65.14 | **91.7%** |

### 27B
| model | suite | FP base | ternary | retention |
|---|---|---|---|---|
| Bonsai 27B | 15-bench | 85.07 | 80.49 | **94.6%** |
| Bonsai 2 27B | 20-bench | 85.4 | 83.9 | **98.2%** |

### 1-bit, for reference (10-bench)
| model | FP base | 1-bit | retention |
|---|---|---|---|
| 1-bit Bonsai 1.7B | 58.24 | 40.88 | 70.2% |
| 1-bit Bonsai 8B | 71.02 | 59.86 | 84.3% |

## Our canary (Qwen3-1.7B, PPL ratio — a *different ruler*)

`ratio = student PPL / teacher PPL`; retention = 1/ratio. Clean multi-region
holdout, 8 regions × 8 windows.

| run | best ratio | retention |
|---|---|---|
| unrotated STE control (5000 st) | 2.0855 | 48.0% |
| rotation + STE (5000 st) | 1.3683 | 73.1% |
| rotation + STE (10000 st) | 1.2152 | 82.3% |
| + decay, seed 2 (best point) | 1.1052 | 90.5% |
| + decay, seed 1337 (best) | 1.1373 | 87.9% |
| + decay, seed 3 (final) | 1.1532 | 86.7% |

**Reading.** At 1.7B, Prism's *own* model retains ~85–88% on benchmarks. Our
canary sits at ~87–90% on PPL. The metrics are not the same, so this is not a
"we beat Prism" claim — but it does mean our canary is **in the same regime as
Prism at that scale**, and the 97% target belongs to the 27B.

## The test that follows

Run the recipe as a **size ladder** (0.6B → 1.7B → 4B → 8B) with everything else
matched, and compare the *shape* of our retention-vs-size curve to Prism's:

- **If our retention rises with size like Prism's**, the recipe is on the right
  track and the 27B run becomes a confirmation, not a gamble.
- **If our retention is flat in size**, the recipe is missing a structural piece,
  and no amount of scale will fix it.

That is the falsifiable question. "We could not reach 97% at 1.7B" is not.

## Ladder results so far (2026-09-23) — and three confounds

TinyShakespeare, 10k steps, seed 1337, 8×8 holdout. Holdout verified against the
F11 leak: 589 windows − (8 regions × 8) = 525 train windows, matching the log.

| rung | teacher PPL | best step | best ratio | retention | ΔPPL (best) |
|---|---|---|---|---|---|
| 0.6B (blk 1024) | 73.36 | 6500 | 1.1033 | 90.6% | 7.00 |
| 1.7B (blk 1024) | 48.98 | 10000 | 1.2152 | 82.3% | 10.04 |
| 4B (blk 512, split) | — | — | ~3.3 (killed @6000) | — | — |

**Both metrics agree** — the smaller model degrades less in ratio *and* in
absolute ΔPPL, so it is not a metric artifact. The 1.7B's trajectory also
plateaus (oscillating ~1.22–1.34) rather than still descending, so it is not
under-training either. On this evidence the recipe shows the **opposite** of
Prism's scale trend. Three confounds must be cleared first:

1. **Corpus.** Teacher PPL is 73 (0.6B) / 49 (1.7B) on tinyshakespeare — both far
   out of distribution, so "retention of a struggling FP model" may measure
   corpus quirks. The `wiki-*` runs (teacher PPL ~25) address this.
2. **Rotation block.** The spec rule gives block 512 for the 4B (widths 2560,
   9728) but 1024 for 0.6B/1.7B, so the ladder varies block as well as size.
   Early signal on the 1.7B control: block 512 is ~3× worse at init and ~10%
   worse at step 2000. `ste-rotate-10k-block512` settles the converged effect.
3. **Convergence / schedule.** All rungs are constant-LR with no schedule, and
   the 1.7B sits near its divergence boundary. None is a converged, scheduled run.

**Protocol implication.** A fixed-step ladder is iso-token, which under-trains
larger models; and a ratio measured against a per-model teacher is not
scale-comparable when teacher PPL varies. A defensible scaling test needs
(a) iso-compute or iso-convergence (e.g. managed decay per rung), (b) a corpus
where the teacher is in-distribution, and (c) a common reference for the metric.
Until then the ladder supports no scaling claim in either direction.

Scripts: `scripts/pilot/run_size_ladder.sh`, `run_wiki_ladder.sh`,
`run_rotblock_control.sh`, `plot_retention.py` (writes
`artifacts/rmd/ladder-summary.md` + `docs/figures/retention-vs-scale.png`).
