# Recipe ledger — the accumulated runs, categorised

**Purpose.** We have ~50 pilot runs in `artifacts/rmd/`. They already answer most
of the "which lever matters?" questions; this doc *categorises* them so we stop
re-deriving, and so the remaining gaps are explicit. Every number is the deployed
metric (`--ste`, ternary forward in the loop), best `projected_ratio` per run,
`retention = 100 / ratio`.

**Scope warning (2026-09-24):** the ladder and most rows below are the legacy
Qwen3-1.7B canary. Qwen3.8-27B is a hybrid `qwen3_5` model with a different
projection inventory and is not covered by these rows. Architecture selection,
rotation/export parity, provenance, and the cross-architecture protocol are
tracked in [`REPRODUCIBILITY-AUDIT.md`](REPRODUCIBILITY-AUDIT.md) and
[`../configs/model_registry.yaml`](../configs/model_registry.yaml). Gate/T29
and private-project numbers elsewhere in this ledger are historical references;
their large artifacts are not all present in this public checkout.

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
| **Mirror descent / q-norm map (one well)** | **falsified** | all `md-*` 7–206×; fights KD and oscillates |
| **Two-well (many-to-one) mirror map** | **falsified** | `wiki-tw-0.6B` loss held ~172–177 vs baseline 5.75 (killed at step 1500); the additive form of the same potential (`tern-lam*`) collapsed too |
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

## Public material (mined 2026-09-23)

**The recipe is not public.** All four whitepapers (`bonsai-2-27b`,
`bonsai-27b`, `ternary-bonsai-8b`, `1-bit-bonsai-8b`), the docs site
(`docs.prismml.com`), and the `PrismML-Eng` repos disclose only the **storage
format** (ternary g128 / `Q1_0`, group-wise FP16 scale), the **weight basis** for
Bonsai 2 (blockwise Hadamard, block 1024, fixed ±1 signs), inference kernels, and
benchmarks. The method is labelled *"proprietary Caltech intellectual property"*.
An independent reproduction (ThakiCloud) reaches the same conclusion: *"It cannot
be reproduced from public materials."*

**What *is* public and useful:**

- **Approach class** — *"starts from an off-the-shelf pretrained model and moves
  it into a binary or ternary representation"*: not from-scratch (BitNet), not
  *"bespoke calibration, auxiliary metadata, or custom runtimes."*
- **The generations differ.** The older Ternary-Bonsai line (1.7B/4B/8B,
  `Q2_0_g64`) uses **no rotation** — it runs on stock llama.cpp. Bonsai 2
  (`PQ2_0`/`PTQ1_0`) **adds the Hadamard rotation** (needs the fork's FWHT). Our
  forensics targets Bonsai 2.
- **Independent reproduction** ([`ThakiCloud/bonsai-1bit-repro`](https://github.com/ThakiCloud/bonsai-1bit-repro)),
  for the **older** line: fingerprint = *"~28% of signs flipped vs the base, group
  scales ~2.26× the naive mean → the signature of error compensation (GPTQ/OBQ
  family) on an unmodified base (consistent with 'no retraining')."* Public
  frontier: GPTQ ~10× over naive → +salient ~3000× → **QuIP (rotation + error
  comp) reaches 2.1× FP16 at pure 1.125 bpw**.

**Note on attribution (avoid a false lead).** Two *different* public leads are
easily conflated: ThakiCloud's is **error compensation** (GPTQ/OBQ), for the
*older* line; the **mirror-map / potential** idea came from a **community
commenter** on the HF post (Caltech → Hassibi mirror descent), *not* from Prism
or ThakiCloud. Prism's own text names neither — only *"proprietary Caltech
intellectual property"* and *"a representation transformation… while preserving
its behavior."* Both leads have now been tested for Bonsai 2: error
compensation rejected (Gate 3 + scale ratio ≈1×), mirror maps falsified
(one-well `md-*`, two-well `wiki-tw-0.6B`). Nothing Prism published *claims* a
mechanism, so there is no published method to be inconsistent with — the most
likely residual explanation is **scale/compute on the lever we already have**
(rotation + QAT/KD), not a hidden potential.

**The fork this exposes.** Our Gate 1–3 concluded Bonsai 2's ~8% residual is
**trained weights** (QAT/KD); ThakiCloud concludes the **older** line is
**error-compensated PTQ, no retraining**. Different models (unrotated older vs
rotated Bonsai 2), so both can hold — but it raises the live question: **is
Bonsai 2 rotated-PTQ (QuIP-style error compensation) rather than QAT/KD?** If so,
the lever is error compensation (a Hessian), not distillation, and our
KD-centric recipe is aimed at the wrong target.

**Cheap decisive test — run 2026-09-23.** Applied ThakiCloud's fingerprint
(sign-flip over non-zero codes + scale ratio vs the naive absmean) to the older
ternary line, and compared to Bonsai 2:

| model | sign-flip (non-zero) | scale ratio | basis |
|---|---|---|---|
| older `Ternary-Bonsai-1.7B` | **12.7%** | **1.96×** | unrotated (comparable) |
| ThakiCloud older 1-bit | ~28% | ~2.26× | unrotated |
| `Ternary-Bonsai-2-27B` | ~chance vs raw base → **rotated** | **≈1×** (via Gate 1) | rotated |

The older line reproduces ThakiCloud's error-compensation signature (scales
~2× naive) and is genuinely **unrotated**. Bonsai 2 is **rotated**, so signs
aren't comparable to the raw base — but Gate 1's **92% absmean-RTN agreement on
the rotated base** is only possible if Bonsai 2's scales are ≈ naive (a 2×
inflation would drop the agreement far below 92%).

**Resolution: different methods per generation.** Older line = error-compensated
**PTQ** (unrotated); **Bonsai 2 = rotated, scales ≈ naive → not
error-compensated** → its residual is training, matching our Gate 3. Our recipe
(rotation + QAT/KD) targets Bonsai 2's actual method, and it already beats the
public PTQ frontier (our 1.57× vs QuIP's 2.1× at 1.7B). The remaining gap is
scale/data/compute or the undisclosed detail — the runs, not a public technique.

## Still unablated

- **KD on/off** at convergence (the one lever with no off-switch data).
- **Managed-decay schedule parameters** (warmup/patience/factor/drift) at convergence.
- **STE scale mode** (absmean vs absmax) at convergence.
