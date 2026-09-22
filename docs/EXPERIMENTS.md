# EXPERIMENTS — Mirror-descent attractor search (1.7B canary)

Status of the recipe search for the last 8% (mission bar: projected ratio
<= 1.031x at 1.7B, i.e. >= 97% retention). All numbers are from
`scripts/pilot/rmd_kd.py` on the Qwen3-1.7B canary (student and teacher on one
card, KD T=2, 585 train windows x 512, 4 eval windows x 512, teacher PPL 32.93).
Ratio = model PPL / teacher PPL. The original baseline "T28 STE+KD 1.103x" is
**retracted** (see "T28's 1.103x is not a clean holdout" below and F11): it was
measured on training windows, and the clean-holdout value for that recipe is
**~2.0x**.

## Hypothesis

A community commenter on the HF post connected the residual 8% to the Caltech
mirror-descent line: Azizan, Lale, Hassibi (arXiv:1906.03830) and Explicit
Regularization via Regularizer Mirror Descent (arXiv:2202.10788, Algorithm 1),
plus Ajanthan et al. (arXiv:1910.08237). A large-q potential `|w|^q`, q >> 1,
should pull the weight distribution toward the {-1,0,+1} attractors while KD
preserves the function, making the final ternary projection nearly lossless.

## Honest ledger (committed to `artifacts/rmd/`)

Control: KD only, 3000 steps. Init 240,579x -> final 27,978x. Kurtosis flat
~2.0. This whole table scores the *projection cost* of FP-trained masters, a
different ruler from a deployed ternary model's PPL ratio (see the STE section
below). The 1.103x once cited as the T28 baseline is retracted.

| run | config | result | verdict |
|---|---|---|---|
| Control | KD only, 3000 st | 240,579x -> 27,978x; kurtosis flat ~2.0 | baseline |
| Additive pow potential | q=8, lam=0.2 | 32,773x; kurtosis 1.94->1.47, both attractor fractions fell | falsified |
| Per-group attractor, bugged | lam=0.1, eval mutated masters | 16,125x @500, 8,457x @1000, then diverged (21,891 @1500 ... 1.04e9 final) | artifact (in-place projection = accidental alternating projection) |
| Per-group attractor, clean | lam=0.1, deep-copy eval | 108,042,728x @500, 4,557,917x @1000, ... 1,278,940x @3000; kurtosis 2 -> 30 | continuous form falsified |
| Per-group attractor | lam=0.02, deep-copy eval | 9,501,374x @500, 439,430,657x @1000, 3,733,210x @1500, loss 2.3-2.5; terminated early @1750 (operator: switch to GPU1-only) | continuous form falsified at low lam too |
| Alternating projection | tern lam=0.1, reproject+project every 500 | 1,410,076x @500 (no step-0 snap; trajectory then killed for pivot) | falsified as rescue of the potential form |
| Mirror map (RMD) | --update md q=8, lam=0, scale 1.0 | loss nan by step 500 (raw dual step swamps u = |w|^7) | falsified (diverged) |
| Mirror map (RMD) | scale 2.6e-8, bf16 math | loss nan by step 250 (bf16 rounding drift: mean|w| 0.025 -> 0.12 in 300 sim steps; f32 stable) | falsified (precision) |
| Mirror map (RMD) | scale 2.6e-8, f32 math | loss nan by step 250 (real KD grads: shell step ~= u makes first moves ~+/-50%, logits -> inf through 30 layers) | INVALID: GPU1 driver wedge (see below) |
| Strict accident replication | tern lam=0.1, init snap + reproject every 500, honest pre/post-snap evals | training from a grid-snapped init diverges (loss nan @250); the accident provably did NOT snap at init (its per-checkpoint losses are identical to the clean run's), so this variant is NOT the accident | INVALID: GPU1 driver wedge |
| Full honest AP trajectory | tern lam=0.1, reproject every 500, A (pre-snap cost) + B (post-snap quality) | A: 108M @500 -> 967k @1000 -> 24.9k @1500 -> 19.2k @2000 -> **7.9k @2500** -> 17.5k @3000. B: 831k @500 -> 40.9k @1000 -> 21.5k @1500 -> 22.3k @2000 -> 49.4k @2500 -> 814k @3000 | AP anchors weights (A collapses 13,700x), but the deployed state degrades late (B ends 814k; kurtosis 2->8); not the goal |
| Mirror map (RMD) retest | q=8, scale 2.6e-8, lam=0, f32 math, healthy GPU | **32.19x @500, 94.31x @1000, 35.67x @1500, 113.25x @2000, 90.55x @2500, 54.36x @3000**; kurtosis ~2.6-3.0 | **LEAD CANDIDATE**: 515x better than control final (27,978x); oscillates 30-115x; the wedge was the only thing that killed the earlier attempts |
| Rotated basis, pure KD | --rotate lam=0 | init **15,805x** (15x better than unrotated 240,579x), **2,872x @500** (best single point of the search), then oscillates 45k-189k, final 69,021x; KD loss 0.91 -> 0.71 (best of any run); kurtosis flat 3.7 | geometry helps at short horizon, weights drift off-projectable over time; candidate for combination |
| Gate 0.2 + tern attractor | lam=0.1, gate 0.2 (frozen top 20% by |w|) | 127M @500, 4.7G @1000, 5.1M @3000; kurtosis 4.4 -> 3.8 | falsified; freezing top weights does not rescue the tern potential |
| md tuning battery (healthy GPU, 3 phases x 2 cards) | | | |
| rotate + md q8, shell 0.020 | | 52.43x final | rotation + shell; no synergy at q8/base shell |
| md q16, shell 0.020 | | 1,479x final | stiffer shell alone is bad |
| rotate + md q16, shell 0.020 | | **28.28x final (best yet)** | rotation rescues q16; best config |
| md q8, shell 0.034 | | 39.85x final | larger shell beats base (54.4x) |
| md q8, shell 0.015 | | 206.28x final | smaller shell worse |
| rotate + md q8, shell 0.034 | | 54.84x final | rotation does not help at shell 0.034 |

## Deployment metric: the STE arm (2026-09-22)

Every row above scores `projected_ratio` of FP-trained masters: the cost of
snapping an unconstrained solution to the grid. That is not the number T28
reported. T28 trained with the ternary forward in the loop (`ternary_ste`), so
its 1.103x is the deployed ternary model's own PPL ratio. Those are the same
quantity (ternary PPL / teacher PPL) but produced by different procedures, so
the FP-projection leaderboard was the wrong ruler for the mission bar.

`scripts/pilot/rmd_kd.py --ste` closes that gap: it wraps the target linears in
`TernaryLinear` (forward = absmean STE) and reports the model's own PPL ratio,
directly comparable to T28. Single card GPU1, student+teacher both `cuda:1`,
3000 steps, 4 eval windows x 512, peak 10.18 GiB allocated / 11.38 GiB reserved
(so the run fits one 20 GiB card; it does not need the two-card split).

| run | init | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | verdict |
|---|---|---|---|---|---|---|---|---|
| STE control (KD + Adafactor, T28 recipe) | 655,974x | 16.23x | 14.62x | 5.69x | 4.94x | 3.93x | **2.93x** | still descending |
| STE + mirror map (q16, shell 0.020) | 655,974x | 94.53x | 46.45x | 44.34x | 65.05x | 55.66x | 23.01x | **falsified** (worse than control at every checkpoint) |

Reading:

- The STE control reproduces the T28 regime: a monotone descent toward the
  teacher. The 7000-step run plateaus at 2.01x, so the descent stalls well above
  the mission bar on this corpus. The 1.103x it was once compared to was never a
  valid target (see the holdout section below).
- The mirror-map update does **not** help inside the ternary loop at this
  config; it fights KD and oscillates. This is the apples-to-apples test the
  FP-projection ledger could not provide.
- The FP runs bottoming at ~28x are far above the STE control's 2.93x, which
  confirms those numbers were a different metric, not progress toward 1.031x.

## T28's 1.103x is not a clean holdout (2026-09-22)

The 7000-step STE runs (anchor and `--gate 0.2`) plateaued near 2.0x, not
1.103x. The gap is a measurement artifact in the T28 harness, not a training
regression.

- `bonsai_forensics/recover.py::build_batches` samples windows from
  `ids[:usable]` with **no holdout** (recover.py:138-142), so every window is in
  the training pool.
- `evaluate_perplexity` evaluates at `start = samples * seq_len` = 32 * 512 =
  16384, i.e. tokens `[16384, 18432)` (recover.py:166-175), which lies inside
  that pool. Over 5000 steps at batch 2 each eval window was sampled ~17 times.
- T28's `heldout_ppl_student` 42.988 / teacher 38.966 = 1.103x (private record,
  `hivebench/artifacts/ternary/recover/run1/recover-report-s5000.json`, config
  `seq_len 512, samples 32, batch 2, steps 5000`) is therefore a train-set
  evaluation.

`rmd_kd.py` holds the last 4 windows out entirely (`train = ids[:-4]`,
`eval = ids[-4:]`), so its numbers are clean. Clean 7000-step anchor: **2.0097x**.

| run (7000 steps, clean holdout) | 500 | 1000 | 2000 | 3000 | 5000 | 6000 | 7000 |
|---|---|---|---|---|---|---|---|
| STE control (T28 recipe) | 16.23x | 14.62x | 4.94x | 2.93x | 2.22x | 2.14x | **2.01x** |
| STE + `--gate 0.2` | 29.17x | 35.29x | 10.58x | 5.27x | 2.70x | 2.91x | **2.42x** |

A region diagnostic (`/tmp/opencode/diag_regions.py`) confirms the harness is not
miscalculating: on both T28's region (tokens 16384:18432, FP PPL 66.0) and the
rmd region (last 2048, FP PPL 35.6), absmean-STE quantization of the base model
is catastrophic at init (145,702x and 135,953x respectively), matching the
harness's ~656,000x init to within the expected numeric spread at those
magnitudes.

Consequence: the mission bar `<= 1.031x` was set against 1.103x, which is
optimistic. The honest 1.7B retention for the T28 recipe is ~2.0x on unseen
tokens. This needs to be re-baselined before any variant is judged, and the
whitepaper's 1.103x figure (docs/WHITEPAPER.md sections 5-6) is affected.

## Runtime note: what actually sets the pace

CORRECTION: the cards run at the same pace. A same-config 200-step tern probe
takes 138s / 0.69 s/step on BOTH cards, with byte-identical losses and final
ratio (80848989.3167). The apparent "GPU0 is 1.5x faster" split was workload
confounding, not hardware:

- the tern potential (`--lam 0.1`, computed on all 196 target linears every
  step) costs ~+0.2 s/step vs `lam=0` (0.69-0.77 vs 0.49-0.52 s/step);
- `--update md` skips the Adafactor step entirely;
- eval deepcopies (`--project-every`/`--eval-pre-snap`) add ~5-6 min per run
  (6 x ~50s);
- both cards reproduce the same numbers exactly (healthy GPU1 matches GPU0).

3000-step ETA: ~25-27 min for lam=0 runs, ~38-39 min for tern-potential runs,
either card.

## GPU1 driver wedge (2026-09-21 ~05:25+)

After an OOM crash at init plus repeated SIGKILLs on GPU1 (PCI 07:00.0,
cuda:1), training-backward on that card produces deterministic NaN at step
~100 (loss 9.38, 9.27, 9.31, then nan; final projected ratio 1.64e61,
byte-identical across runs). Controls: same config on cuda:0 trains clean;
teacher forward on cuda:1 is clean; simple ops (matmul/softmax/pow) on cuda:1
are clean. Verdict: wedged driver state on GPU1, not code or config.

Consequence: every rmd run from ~05:25 to the reboot is invalid (mirror-map
f32 retest and the full honest AP trajectory must be re-run on a healthy
GPU). Results before 05:16 (control, pow/tern potentials, clean tern runs,
tern-lam0.02, honest AP B(500) = 1,410,076x) stand.

## The artifact, settled

The bugged eval never perturbed training: its 250-step losses match the clean
run's exactly (9.19, 9.02, 8.75, 8.59, 8.22, 8.02, 8.27, 8.06, 7.59, 7.44), so
the in-place "reset" had no effect on the trajectory. The 16,125x / 8,457x
numbers were a broken measurement of the same weights that honestly project to
108,042,728x @500 (clean eval). There was never a collapse in the weights; the
"accidental alternating-projection winner" was an eval-accounting ghost. The
honest alternating-projection numbers (1,410,076x @500) are the truth.

## Lesson (the bug that matters)

The first attractor run evaluated projection by mutating the TRAINING weights
in place at each checkpoint, secretly snapping the masters to the ternary grid
every 500 steps. That accidental alternating-projection schedule produced the
best numbers (16k -> 8.4k). The fix (deep-copy eval) revealed the continuous
form is bad. Projection evals must never mutate training masters; alternating
projection is now an explicit, legitimate variant.

## Current candidates, in order

All candidates are judged in the STE/deployed metric (`--ste`). The honest
anchor is the clean-holdout **~2.0x** plateau; the retracted 1.103x is not a
valid target.

1. DONE: plain STE+KD at 7000 steps reached **2.01x** (clean holdout). This is
   the standing anchor.
2. `--reproject-every 500` explicit alternating projection, now inside the STE
   loop (AP was only ever tested on FP masters).
3. `--gate 0.2` + tern attractor inside STE (freeze the top 20% by |w|, train
   the flexible 80%; encodes the Gate-1 20/80 structure).
4. `--rotate` inside STE (train in the spec-rotated basis, PRF signs seed 1337:
   q/k/v, gate/up absorb `W R^T`; o_proj/down absorb `R W`, bias' = R b).
5. `--update md` (RMD mirror map, Algorithm 1 of arXiv:2202.10788) is
   **FALSIFIED in-loop** at q16/shell 0.020: 23.01x vs the control's 2.93x at
   3000 steps. Revisit only with a changed shell/q and a mechanism argument.
6. If anything beats the 2.0x anchor materially: replicate across 2-3 seeds
   before any 27B work.

## Papers

- Azizan, Lale, Hassibi. Stochastic Mirror Descent on Overparameterized
  Nonlinear Models. arXiv:1906.03830.
- Azizan et al. Explicit Regularization via Regularizer Mirror Descent.
  arXiv:2202.10788 (Algorithm 1).
- Ajanthan et al. Mirror Descent View for Neural Network Quantization.
  arXiv:1910.08237.