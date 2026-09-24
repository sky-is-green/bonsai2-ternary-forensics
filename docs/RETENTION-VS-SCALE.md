# Retention vs scale — Prism published numbers vs our recipe

> **Evidence status (2026-09-24):** the 0.6B/1.7B WikiText rungs are completed
> legacy validation results; the 4B rung was **dropped by decision** (2026-09-24;
> the ladder question is superseded by the archived conclusion). The
> pre-registered primary-corpus PPL threshold is not met by the available rungs,
> and all “best” values are validation-selected. See
> [`REPRODUCIBILITY-AUDIT.md`](REPRODUCIBILITY-AUDIT.md).

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
"we beat Prism" claim. It is only a same-scale sanity check; the 97% target
belongs to the 27B and the scaling rule is not complete.

## The test that follows

Run the recipe as a **size ladder** (0.6B → 1.7B → 4B → 8B) with everything else
matched, and compare the *shape* of our retention-vs-size curve to Prism's:

- **If our retention rises with size like Prism's**, the recipe is on the right
  track and the 27B run becomes a confirmation, not a gamble.
- **If our retention is flat in size**, the recipe is missing a structural piece,
  and no amount of scale will fix it.

That is the falsifiable question. "We could not reach 97% at 1.7B" is not.

## Ladder results — partial evidence (2026-09-24)

10k steps, seed 1337, 8×8 holdout (verified against the F11 leak). Two corpora
plus a block-size control:

| run | corpus | teacher PPL | best@ | ratio | retention | ΔPPL |
|---|---|---|---|---|---|---|
| 0.6B (blk1024) | tinyshakespeare | 73.36 | 6500 | 1.1033 | 90.6% | 7.00 |
| 1.7B (blk1024) | tinyshakespeare | 48.98 | 10000 | 1.2152 | 82.3% | 10.04 |
| 1.7B (**blk512**) | tinyshakespeare | 48.98 | 9000 | 1.2050 | 83.0% | 9.76 |
| 0.6B (blk1024) | **wikitext** | 27.99 | 10000 | 3.1506 | 31.7% | 56.01 |
| 1.7B (blk1024) | **wikitext** | 21.92 | 10000 | 1.9798 | 50.5% | 20.79 |

**Confound 3 (rotation block) — resolved: nil.** Same size, corpus and steps:
blk512 gives 1.2050 vs blk1024's 1.2152 — marginally *better*. The block explains
nothing once trained; its ~3× penalty is an init-only effect.

**Confound 1 (corpus) — resolved: decisive.** On WikiText (teacher PPL 22–28,
in-distribution) retention is 31.7% / 50.5%, versus 90.6% / 82.3% on
tinyshakespeare. The toy corpus was flattering by a very large margin.

**The earlier "reversal" was a corpus artifact.** On tinyshakespeare smaller
looked better (90.6% → 82.3%); on WikiText it inverts to **31.7% → 50.5%** —
retention rising with size, matching Prism's direction, with ΔPPL agreeing
(56.0 → 20.8). So the recipe's scale trend is right once the teacher is
in-distribution.

**Confound 2 (convergence) — resolved: decisive.** Both WikiText rungs peaked at
their final 10k step, i.e. under-trained. Re-run to iso-convergence (20k steps +
managed decay; same corpus, seed, eval set):

| rung | best@ | ratio | retention | ΔPPL |
|---|---|---|---|---|
| 0.6B (blk1024) | 19500 | 1.9263 | 51.9% | 24.50 |
| 1.7B (blk1024) | 17000 | 1.5731 | **63.6%** | 12.08 |

The two available rungs show a promising direction, **51.9% → 63.6%**, with
ΔPPL agreeing (24.5 → 12.1), but this is not the completed pre-registered
result: the primary-corpus teacher-PPL threshold (largest rung ≤20) is not met,
and the earlier 10k values were floors, not final levels.

**The 4B rung was dropped (2026-09-24).** It was the deciding third point, but
the ladder question is superseded by the archived conclusion
([`FORENSIC-ARCHIVE.md`](FORENSIC-ARCHIVE.md)), so it was not run. The first
attempt had already been abandoned as an instrument (no schedule, diverging by
step 5500). The scaling question is therefore **not resolved**: the 0.6B → 1.7B
iso-convergence trend is the available partial evidence, and the pre-registered
rule is not met.

Scripts: `scripts/pilot/run_size_ladder.sh`, `run_wiki_ladder.sh`,
`run_wiki_convergence.sh`, `run_wiki_convergence_4B.sh`, `run_rotblock_control.sh`,
`plot_retention.py` (writes `artifacts/rmd/ladder-summary.md` +
`docs/figures/retention-vs-scale.png`).
