# Recipe ledger — the accumulated runs, categorised

**Purpose.** We have ~50 pilot runs in `artifacts/rmd/`. They already answer most
of the "which lever matters?" questions; this doc *categorises* them so we stop
re-deriving, and so the remaining gaps are explicit. Every number is the deployed
metric (`--ste`, ternary forward in the loop), best `projected_ratio` per run,
`retention = 100 / ratio`.

Corpus matters enormously: shakespeare runs flatter (86–90%), WikiText is honest
(51.9–63.6%). Read the corpus column before the number.

## The ledger (best point per run)

| run | best@ | ratio | retention | corpus |
|---|---|---|---|---|
| ladder-0.6B | 6500 | 1.1033 | 90.6% | shakespeare |
| ste-rotate-20k-decay-s2 | 10000 | 1.1052 | 90.5% | shakespeare |
| ste-rotate-20k-s2 | 10000 | 1.1052 | 90.5% | shakespeare |
| ste-rotate-20k-s3 | 13000 | 1.1396 | 87.8% | shakespeare |
| ste-rotate-20k-decay-s3 | 7500 | 1.1430 | 87.5% | shakespeare |
| ste-rotate-20k | 16000 | 1.1603 | 86.2% | shakespeare (diverges to 1.2930) |
| ste-rotate-10k-block512 | 9000 | 1.2050 | 83.0% | shakespeare |
| ladder-1.7B / ste-rotate-10k | 10000 | 1.2152 | 82.3% | shakespeare |
| ste-rotate-s2 | 4500 | 1.2512 | 79.9% | shakespeare |
| ste-rotate-8x8 | 4000 | 1.3215 | 75.7% | shakespeare |
| ste-rotate-lsq | 5000 | 1.3397 | 74.6% | shakespeare |
| ste-rotate-s3 | 5000 | 1.4015 | 71.4% | shakespeare |
| **wiki-conv-1.7B** | 17000 | **1.5731** | **63.6%** | wikitext |
| **wiki-conv-0.6B** | 19500 | **1.9263** | **51.9%** | wikitext |
| ste-regions-8x8 | 4000 | 1.9792 | 50.5% | shakespeare |
| wiki-1.7B | 10000 | 1.9798 | 50.5% | wikitext |
| ste-control-7k | 6500 | 1.9922 | 50.2% | shakespeare |
| ste-gate02-7k | 6500 | 2.2225 | 45.0% | shakespeare |
| ste-rotate-wiki | 5000 | 2.4515 | 40.8% | wikitext |
| ste-control | 3000 | 2.9256 | 34.2% | shakespeare |
| ste-rotate-5k-lr2e-4 | 4000 | 3.0058 | 33.3% | shakespeare |
| wiki-0.6B | 10000 | 3.1506 | 31.7% | wikitext |
| ste-rotate-md | 3500 | 7.4179 | 13.5% | shakespeare |
| ste-md-q16-sh020 | 3000 | 23.01 | 4.3% | shakespeare |
| rot-md-q16-s6.55e-22 | 3000 | 28.28 | 3.5% | shakespeare |
| md-q8-lam0-s2.6e-8 | 500 | 32.19 | 3.1% | shakespeare |
| rot-md-q16-sh034 | 3000 | 32.28 | 3.1% | shakespeare |
| rot-md-q16-sh050 | 3000 | 34.22 | 2.9% | shakespeare |
| rot-md-q8-s1e-6 | 1500 | 34.86 | 2.9% | shakespeare |
| md-q8-s1e-6 | 3000 | 39.85 | 2.5% | shakespeare |
| md-q16-s6.55e-22 | 500 | 40.26 | 2.5% | shakespeare |
| rot-md-q8-s2.6e-8 | 1500 | 43.62 | 2.3% | shakespeare |
| md-q8-s2.6e-9 | 3000 | 206.3 | 0.5% | shakespeare |
| rotate-lam0 | 500 | 2872 | ~0% | shakespeare |
| tern-lam0.1-ap-full | 2500 | 7909 | ~0% | shakespeare |
| control / q8-lam0.2 / tern-lam0.1 / gate0.2-tern-lam0.1 | — | 2.8e4–2.4e5 | ~0% | shakespeare |

(`probe-b*`, `smoke*`, `ste-*smoke` are short init probes, not results.)

## Component verdicts

| lever | verdict | evidence |
|---|---|---|
| **Rotation in the loop** | **essential** | `ste-rotate-*` 71–90% vs `ste-control` 34.2% vs unrotated `tern-*`/`md-*` ~0% |
| **Managed LR decay** | **essential for stability** | `ste-rotate-20k` peaks 1.1603 then **diverges to 1.2930**; decay variants hold (1.1052/1.1430) |
| **KD from a teacher** (temp 2.0) | **present in every winning run — never ablated** | all `ste-*` runs use it |
| **Adafactor, lr 5e-5** | works | — |
| **Higher LR (2e-4)** | **worse** | 3.0058 vs 1.3683 at 5k |
| **LSQ (learnable scales)** | **marginal** | 1.3397 vs 1.3683 — within noise |
| **Rotation block 512 vs 1024** | **nil** | 1.2050 vs 1.2152, same size/corpus/steps |
| **Mirror descent / q-norm map** | **falsified** | all `md-*` 7–206×; fights KD and oscillates |
| **Gate (freeze top-\|w\|)** | **falsified** | `ste-gate02-7k` 2.2225 vs control 1.9922 |
| **Reproject cadences** | **falsified** | see `EXPERIMENTS.md` |
| **Additive pow potential / per-group attractor (continuous)** | **falsified** | `tern-*`/`lam*` collapse |

## Structural comparison — Prism's public models vs ours

Sparsity is the fraction of ternary codes equal to **0** (the third state). The
no-training RTN baseline of the rotated base is Gaussian ≈ 0.31.

| model | overall | attn | ffn | embed / output | format |
|---|---|---|---|---|---|
| RTN baseline (Qwen3-1.7B, rotated, **no training**) | 0.310 | 0.309 | 0.311 | — | absmean g128 |
| **our trained 0.6B** | **0.319** | — | — | — | absmean g128 |
| **our trained 1.7B** | **0.318** | 0.318 | 0.318 | — | absmean g128 |
| Prism `Ternary-Bonsai-1.7B` | 0.383 | 0.382–0.402 | 0.391–0.409 | 0.309 | PQ2_0 g128 |
| Prism `Ternary-Bonsai-4B` | 0.371 | 0.362–0.381 | 0.371–0.388 | 0.310 | PQ2_0 g128 |
| Prism `Ternary-Bonsai-8B` | 0.372 | 0.368–0.395 | 0.371–0.399 | 0.309 / 0.319 | PQ2_0 g128 |
| Prism `Ternary-Bonsai-2-27B` | 0.328 | 0.3274–0.3276 | 0.3275–0.3277 | 0.328 | PQ2_0 g128 |
| Prism `Ternary-Bonsai-27B` (older) | *unparsed* | | | | Q2_0_G64 |

**Readings.**

1. **Sparsity falls toward the Gaussian baseline with scale** in Prism's line
   (1.7B 0.383 → 4B/8B 0.371 → 27B 0.328), and their **older 1.7B/4B/8B are
   non-uniform** (embed Gaussian ≈ 0.309, projections sparser 0.36–0.41) while the
   **newer 27B is uniform** (0.3276 everywhere).
2. **Our trained models are uniform and near-baseline** (0.318–0.319), i.e. they
   resemble Prism's *newer* 27B, not their older line. **Structure is not the
   differentiator**: their 27B at the same near-baseline sparsity reaches 98.2%,
   ours reaches 63.6% (PPL) — so the quality gap lives in the *values / training*,
   not the sparsity or the format. This rules out the "missing sparsity knob"
   hypothesis.
3. **"Prism's recipe" is not one recipe.** The older line uses a different
   quantizer (`Q2_0_G64`, group 64) than the newer 27B (`PQ2_0`, group 128), and
   the older line's non-uniform sparsity implies a different mechanism
   (projection-targeted) that the newer generation dropped.
4. **Prism keeps γ** (hidden norms not folded; T30), matching our pilot.

## What this does *not* answer

Structure is not quality. Neither our sparsity match to the newer 27B nor Prism's
sparser older 1.7B predicts the retention gap (77.5% vs 85–88% at 1.7B). The
**level** question needs the scale runs; the ledger and the structural table
settle the *component* and *format* questions, not the quality one.

## Still unablated

- **KD on/off** at convergence (the one lever with no off-switch data).
- **Managed-decay schedule parameters** (warmup/patience/factor/drift) at convergence.
- **STE scale mode** (absmean vs absmax) at convergence.
