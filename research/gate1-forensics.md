# Gate-1 forensics — does rotate+RTN in Prism's basis reproduce the released trits?

**Task:** T30 (QUEEN, Round 10, 2026-09-20) · **Cost:** $0, local only
**Module:** `bonsai_forensics/gate1_forensics.py` (16 offline tests)
**Report:** `artifacts/gate1/gate1-report.json` (26 tensors, layers 0/3/62/63)
**Base shards:** `artifacts/base27/model-00001/000017-of-00018.safetensors` (6.1 GB)

## Verdict

**Rotation basis: cracked. Quantizer: not.** Their trits are reproduced at
**0.89–0.95 per tensor (mean 0.920, n=26)** by plain absmean RTN of the rotated
base — versus 0.33 for chance and 0.612 for the 1.7B precedent (R7). The
remaining 5–11% is *not* a scale/basis artifact: it has the fingerprint of
error-compensated quantization (GPTQ/OBQ-family) or QAT. The no-rental
rotate+RTN route therefore produces a strong **initialization**, not a parity
artifact; the next lever is still recovery/hessians, now in the correct basis.

## What had to be right (all [verified] against the released file)

1. **Base identity.** HF `main` sha = `1d4bf0f2…` (the pin). GGUF norms are
   `1 + w` (HF's `(1+w)` RMSNorm parameterization): corr 0.98–0.9996 with the
   base, mean |diff| 0.002–0.014, max 0.032–0.071 (trained drift, see below).
   `A_log` matches `-exp(base)` and `dt_bias` matches base after the converter's
   V-head reorder (bf16 precision).
2. **Rotation convention.** Input-axis, sign-then-Hadamard (FWHT), block 1024,
   the explicit `prism.hadamard.*` signs, **no norm fold**. Plain -> 0.333,
   in_rot -> 0.89–0.95; `fwht-then-sign` and output-axis variants stay at
   chance. Mean over widths 5120/6144/17408 and both layer types.
3. **Layout.** llama.cpp's Qwen3.5 converter tiles V heads grouped->tiled. The
   released artifact has this on **`in_proj_qkv` V rows** (0.56 -> 0.927) and
   **`in_proj_z` rows** (0.36 -> 0.927), but **`ssm_out` stays grouped**
   (0.936; applying the reorder drops it to 0.34) — their converter predates
   or omits mainline's out_proj column reorder. `ffn_*` / `attn_q/k/v/o` are
   plain.
4. **Their scale is LS-optimal for their codes** (`d / s_ls` 1.007–1.034) and
   ≈ 1.37 x our absmean, but using it directly on our rotated base scores
   *worse* (0.88–0.91) than absmean (0.89–0.95). Their effective decision
   boundary is near `0.5 x absmean`.
5. **Our spec quantizer is worse than plain RTN here.** `quantize()` (absmean
   init + LS refine) scores 0.8539–0.8743; `absmean` alone 0.8958–0.9479.
   The LS refinement moves 6–7% of trits the wrong way relative to Prism.
6. **The fork's public tool is not their quantizer.**
   `quantize_row_pq2_0_ref` (exported by their `libggml-base.so`) uses
   `d = max|x|`; run on the rotated base it matches only **0.493** of their
   trits with `d_fork/d_stored` ~2.55 (`--output-tensor-type pq2_0` is pure
   max-scaled RTN, as R2 guessed).
7. **Residual 5–11% is structural, not noise.**
   - Zero fraction: theirs 0.327–0.328 *identical across all tensors*; ours
     0.309. Round(float)-off-stored-scale gives 0.58 nonzero, so their codes
     are denser than a threshold at their own `d`.
   - Mismatch decomposes as ours->0 = 1.6M and 0->theirs = 1.0M elements;
     no `-1 <-> +1` flips. Concentrated at `|w| ≈ 0.5 x scale`, zero at
     `|w| > 1.2 x scale`.
   - Within-group codes are **non-monotone** in `|w|` (viol_frac ≈ 1.0):
     impossible for any pure threshold RTN in this basis.
   - Weight-space relative error vs the rotated base: **theirs 0.460–0.482,
     ours 0.510–0.511** (their assignment is ~10% better).
   - Exempt tensors: `in_proj_a/b` (bf16) are elementwise uncorrelated with the
     base but sorted-norm corr 0.93–0.98 / global std identical -> permutation
     plus drift; norms drifted up to 0.07 absolute with slope 0.99 / corr 0.98.
   Signature: GPTQ/OBQ error compensation initialized from absmean RTN and/or
   QAT of the same base. This is the proprietary step.

## Implications (decision inputs for QUEEN)

- **No-rental hypothesis (R9 handoff §7): partially falsified.** Local
  rotate+RTN in their basis reproduces the container, basis and ~92% of trits,
  but not their weights: expected artifact quality is *not* their 1.44×.
  Building it locally is still worth ~$0 to quantify the residual (KLD/PPL,
  Gate 2), and it doubles as the initialization for T28-style recovery.
- **If the goal is trit parity**, the candidate algorithm is their-type
  GPTQ/OBS: our T4 GPTQ in the *correct* basis (T26 signs + layouts from this
  report) with calibration Hessians (T24/T17). Hessians need teacher forwards
  -> the 55.6 GB memory math stands (4x24 GB or offload). The QAT route (T28)
  now also gets a much better start (absmean RTN in their basis, not ours).
- **Spec/quantizer follow-up (QUEEN hotspot):** if we want our pipeline closer
  to Prism, the LS refinement in `quant.py` should be optional/off for
  PQ2_0-class runs; absmean RTN dominates on this artifact.
- **Layout follow-up:** `run_quant`/`pack_gguf` must reproduce the observed
  per-suffix layouts (qkv/z tiled, ssm_out grouped) before any artifact is
  expected to run in the Prism fork.

## Reproduce

```bash
SHARDS=artifacts/base27   # model-00001/000017 + index.json (hf_hub_download pinned)
python -m bonsai_forensics.gate1_forensics \
  --gguf artifacts/oracle/bonsai27/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --layers 0,3,62,63 --out artifacts/gate1/gate1-report.json
# fork reference quantizer probe: ctypes quantize_row_pq2_0_ref from
# artifacts/oracle/prism-fork/bin/.../libggml-base.so (d = max|x|)
```

Tags: [verified] for everything measured against the files above (single
revision, one binary); [inferred] for "GPTQ/OBQ vs QAT" — both fit the
fingerprint, neither is confirmed.
