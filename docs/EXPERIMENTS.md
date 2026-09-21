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
| Alternating projection | tern lam=0.1, reproject+project every 500 | running (GPU1 only: student+teacher cuda:1) | pending |

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