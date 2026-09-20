# Gate 3 — is the 8% trit residual a quantizer trick or trained weights?

**Task:** R12 diagnostics (2026-09-20) · **Cost:** $0, local only
**Scripts:** `artifacts/gate3/scripts/{scr_test,prefix27,sweep27,hobj27}.py`
**Evidence:** `gate3/scr-report.json`, `gate3/prefix27/{prefix-report,sweep-layer0,h-objective}.json`,
Hessians in `gate3/prefix27/hessians/` (layers 0+3) and `gate3/hessians-noise5/` (1.7B).

## Verdict

**Their weights were trained (QAT/KD), not error-compensated on the public base.**
Every candidate "rounding trick" was tested with real activations and made the match
worse, while their codes fit the activations *less* well than plain RTN. The 8% residual
is the weight movement from training; it cannot be reproduced by any quantizer applied
to `Qwen/Qwen3.8-27B`. The reverse-engineering track (T4 GPTQ / rentals for GPTQ) is closed.

## Experiment 1 — Stochastic Calibration Resonance (falsified)

1.7B, clean vs 5%-random-token-noise calibration Hessians, GPTQ g128, damp 0.1,
against Prism's released 1.7B PQ2_0 trits (byte-exact decode vs their F16 dequant):

| variant | agreement |
|---|---|
| RTN absmean | 0.61196 |
| GPTQ clean | 0.61771 |
| GPTQ noise5 | 0.61774 |
| codes moved by noise | 4.28% |

Noise changes 4.3% of the codes and changes agreement by **+0.003 pp** — no effect.
(Side finding: the T7 `Q2_0_g64` decoder in `oracle.py` decodes garbage; T7's 0.612
actually came from the F16 dequant. PQ2_0 decode is byte-exact.)

## Experiment 2 — real 27B Hessians from a 4-layer prefix

No 55 GB residency needed: layers 0–3 + embedding were loaded alone (55 tensors,
`prefix27.py`), real Shakespeare windows, float32 Hessians on the *original* norm
output. For each tensor the weight was layout-adjusted (T30) and rotated with the
explicit signs; Hessian rotated `R H Rᵀ`; GPTQ ran in that basis.

| variant (layer 0 aggregate) | agreement vs their trits |
|---|---|
| **RTN absmean** | **0.9145** |
| GPTQ damp 0.01 | 0.8086 |
| GPTQ damp 0.1 | 0.8395 |
| GPTQ damp 0.3 | 0.8497 |
| GPTQ act-order | 0.8065 |
| GPTQ with `Rᵀ H R` | 0.8086 |
| GPTQ with raw (unrotated) H | 0.7991 |
| group scale search, diagonal-H | 0.8625 |
| group scale search, block-H | 0.8506 |

Monotone in damping: as the compensation is suppressed, agreement converges back to
RTN. **Every** error-compensated assignment moves codes *away* from theirs.

## Experiment 3 — whose codes fit the activations?

Per-group H-metric LS scales for both code sets, evaluated on the same Hessians:

| | median error / base energy |
|---|---|
| their codes | 1.072 × RTN's error |

Their trits are not a better fit to *our* activation distribution — they are optimized
for a different objective/distribution. Combined with Experiment 2, the only mechanism
left standing is that the weights themselves moved during training.

## Corroborating evidence (T30 + Gate 2)

- Top 20% of weights by |w|: 100% trit agreement; mismatches live in the threshold band.
- Exempt tensors drifted: norms up to 0.07 absolute, `in_proj_a/b` decorrelated (bf16).
- Their weight-space error is ~10% lower than RTN's (training reduces quantization error).
- Gate 2: RTN of the base = 23,606 PPL vs their 18.59 (1,270×) — the trained weights *are* the quality.

## Consequences

- **Close the quantizer-reverse-engineering track.** No PTQ variant of the public base
  reaches their codes; GPTQ at 27B is not worth a rental for parity.
- **Training is the only path to our own artifact**: T28-style QAT/KD (1.7B reached
  1.10×). Local options: LoRA/block-wise KD with entropy-selected data (T24/T17
  machinery); full-master needs 1×48–80 GB.
- **T29 stays the pragmatic arm** (their released weights already run at 39.6 t/s, PPL 17.86).
- **Spec follow-up stands**: our LS-refined scale loses to absmean RTN on their artifact.
- Optional: fix `oracle.decode_q2_0_g64` (type-42 layout) — latent bug, not on the 27B path.

Tags: [verified] for all measured numbers (one corpus, 8 windows, layers 0/3; direction
is uniform across tensors); [inferred] for "QAT/KD specifically" vs any other training
method — the data proves *some* weight movement, not which recipe.
