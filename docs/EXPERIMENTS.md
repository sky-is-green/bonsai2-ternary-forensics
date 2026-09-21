# EXPERIMENTS — Mirror-descent attractor search (1.7B canary)

Status of the recipe search for the last 8% (mission bar: projected ratio
<= 1.031x at 1.7B, i.e. >= 97% retention). All numbers are from
`scripts/pilot/rmd_kd.py` on the Qwen3-1.7B canary (student cuda:0, teacher
cuda:1, KD T=2, 585 train windows x 512, 4 eval windows x 512, teacher PPL
32.93). Ratio = projected PPL / teacher PPL at absmean RTN g128. Baseline to
beat: T28 STE+KD 1.103x.

## Hypothesis

A community commenter on the HF post connected the residual 8% to the Caltech
mirror-descent line: Azizan, Lale, Hassibi (arXiv:1906.03830) and Explicit
Regularization via Regularizer Mirror Descent (arXiv:2202.10788, Algorithm 1),
plus Ajanthan et al. (arXiv:1910.08237). A large-q potential `|w|^q`, q >> 1,
should pull the weight distribution toward the {-1,0,+1} attractors while KD
preserves the function, making the final ternary projection nearly lossless.

## Honest ledger (committed to `artifacts/rmd/`)

Control: KD only, 3000 steps. Init 240,579x -> final 27,978x. Kurtosis flat
~2.0. Projection error of the trained masters is the baseline; T28's 1.103x
was the STE+KD recipe, this harness is the FP-forward KD control.

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

1. `--reproject-every 500` (explicit alternating projection; the accidental
   winner). No potential needed (or with small lam).
2. `--gate 0.2` + tern attractor (magnitude-gated: freeze the top 20% by |w|
   per tensor at init, train the flexible 80%). Encodes the Gate-1 20/80
   significance structure.
3. `--update md` (true RMD mirror-map update, Algorithm 1 of arXiv:2202.10788):
   mirror map `u = sign(w)|w|^(q-1)`, step in dual space, map back
   `w = sign(u)|u|^(1/(q-1))`. The map is the regularizer; run with `--lam 0`.
4. `--rotate` (train in the spec-rotated basis, PRF signs seed 1337: q/k/v,
   gate/up absorb `W R^T`; o_proj/down absorb `R W`, bias' = R b). Matches the
   27B geometry; the 1.7B is unrotated. The forward preserves the unrotated
   function exactly (float32-exact on all four edge shapes).
5. If anything hits <= 1.031x: replicate across 2-3 seeds before any 27B work.

## Papers

- Azizan, Lale, Hassibi. Stochastic Mirror Descent on Overparameterized
  Nonlinear Models. arXiv:1906.03830.
- Azizan et al. Explicit Regularization via Regularizer Mirror Descent.
  arXiv:2202.10788 (Algorithm 1).
- Ajanthan et al. Mirror Descent View for Neural Network Quantization.
  arXiv:1910.08237.