# DSpark drafter for Bonsai 2 — track plan

**Goal.** Build the **first DSpark draft model for Bonsai 2** (and, as a second
novelty, the first **ternarized** drafter — every published draft is Q8).  This
is the tractable, local, *completable* contribution: the 27B ternary target
fits on one card, so nothing here needs a 27B training run.

Motivation and the crowded prior art:
[`PRIOR-ART.md`](PRIOR-ART.md).  Reference bundle:
[`REFERENCE-PROCREATIONS-MTP.md`](REFERENCE-PROCREATIONS-MTP.md).

> **Status (2026-09-24).** Features, the target, and the runtime are ready, but
> the smoke draft is **not** the fork's `Qwen3DSparkModel` topology and cannot be
> exported/benchmarked as-is: [`DSPARK-EXPORT-AUDIT.md`](DSPARK-EXPORT-AUDIT.md).
> The next real step is a draft trained to the fork layout (qk-norm + raw-tap
> `fc`), which for a 27B target is a GPU run, not a small local one.

## Why this and not "reproduce Bonsai 2"

- The community has solved **format + interop** (four MTP grafts), **drafters**
  (DFlash2), and **post-hoc code refinement** (GSQ/RCO).  Nobody retrains the
  ternary base — the recipe is unclaimed but the 27B run is infeasible locally
  (student + teacher ≈ 111 GB).
- A drafter is **small** (~5 layers) and distilled from a target whose ternary
  form is **~7.2 GB** — it fits in one 20 GB card.  So we can actually finish
  and measure it.

## The mechanism (why DSpark)

DSpark = DFlash block diffusion **+ a semi-autoregressive Markov head** (a
low-rank logit bias keyed on the previous token, chained across the block).  It
keeps drafting at one decode per block while recovering the left-to-right signal
pure diffusion loses → **higher acceptance at nearly the same cost**.  It also
adds hidden correction, log-SNR conditioning and an optional confidence head
(`--spec-draft-conf-min` truncates blocks where predicted acceptance is low).

## Assets

| need | asset | status |
|---|---|---|
| target | `prism-ml/Ternary-Bonsai-2-27B-gguf` (PQ2_0, ~7.2 GB) | cached |
| backbone donor | `z-lab/Qwen3.8-27B-DFlash2` (BF16, 3.85 GB) | pulled |
| DSpark template | `deepseek-ai/dspark_qwen3_4b_block7` | pulled |
| runtime | PrismML-Eng/llama.cpp (ROCm build present, `draft-dspark`) | built |
| feature collector | `scripts/pilot/dspark_collect.cpp` | **validated** on Bonsai 2, layers `[5,19,33,47,61]` |
| draft module | `bonsai_forensics/dspark.py` | **reduced POC** — not fork-loadable, see audit |
| smoke trainer | `scripts/pilot/dspark_train.py` | loss 12.85 → 7.52 in 20 steps |
| data prep | `scripts/pilot/prep_dspark_texts.py` (UltraChat 200k, pinned) | |
| collect runner | `scripts/pilot/run_dspark_collect.sh` | |

## Pipeline

1. **Features** — `prep_dspark_texts.py` → `run_dspark_collect.sh` (Bonsai 2,
   intermediate layers `5,19,33,47,61`).
2. **Train** — distill the draft on (token, target-hidden) with next-token KD;
   stage 2 on on-policy generations (as ProCreations).
3. **Export** — an HF checkpoint with `architectures:["Qwen3DSparkModel"]` and the
   fork's exact inventory, converted by `convert_hf_to_gguf.py`'s `DSparkModel`
   path (`conversion/dspark.py`). **Blocked:** the smoke draft is not that
   architecture — `docs/DSPARK-EXPORT-AUDIT.md`; `scripts/pilot/dspark_export_check.py`
   audits any checkpoint before conversion.
4. **Benchmark** — `--spec-type draft-dspark`; acceptance + tok/s vs DFlash2/MTP.

## Novel variants (stackable)

1. **First ternarized drafter** — export the draft as PQ2_0; decoding is exact,
   so only acceptance can move.
2. **GSQ/RCO on the draft** — teacher-guided code refinement + bit allocation.
3. **Ternary draft + n-gram + conf-min** — learned and training-free drafting.
4. **Fully low-bit speculative stack** — ternary target + ternary draft +
   quantized KV + batch-invariant kernels.

## Caveat to keep honest

The wall is **verification cost**, not drafting (sudoingx/BoldingBuilds) — at low
bpw the target verify dominates.  The headline gain may be modest; the point is
that it is **finishable and measurable**, unlike the 27B recipe gap.
