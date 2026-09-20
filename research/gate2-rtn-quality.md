# Gate 2 — quality cost of the 8% trit residual (RTN artifact vs released PQ2_0)

**Task:** T31 (Round 11, 2026-09-20) · **Cost:** $0, local only
**Module:** [`bonsai_forensics/gate2_rtn_artifact.py`](../bonsai_forensics/gate2_rtn_artifact.py) (8 offline tests)
**Evidence:** `artifacts/gate2/{patch-report.json,eval-report.json}`,
artifact `artifacts/gate2/rtn-absmean.gguf` (7.21 GB)

## Verdict

**The residual is the model.** A byte-controlled swap of all 402 PQ2_0 payloads
with our `rotate + absmean RTN` weights of the public base collapses held-out
PPL from **18.59 to 23,606 (1,270×)** in the same binary, corpus and settings —
even though those payloads still agree with Prism's trits on 90–95% of weights.
The rotation basis buys the *container*; the proprietary assignment buys the
*quality*. Local rotate+RTN (the no-rental shortcut) is dead for parity.

## Method (controlled)

1. [`gate2_rtn_artifact.py`](../bonsai_forensics/gate2_rtn_artifact.py) copies Prism's GGUF and rewrites every mapped PQ2_0
   payload in place: base tensor -> converter layout (qkv/z V-head tiling,
   ssm_out grouped, embeddings plain) -> `R = H·diag(S)` on the input axis ->
   absmean RTN g128 -> PQ2_0 pack. Metadata, tokenizer, F32/BF16 exemptions and
   the tensor table stay byte-identical, so file size and offsets do not move.
2. Repack is byte-exact: `unpack` -> `pack` of their payloads reproduces the
   source bytes (checked on 4 tensors); post-patch re-read verifies all 402
   SHA-256s (`verify.ok=true`, 0 mismatches).
3. Post-patch per-tensor trit agreement re-checked: 0.896–0.948, identical to
   T30's numbers — the swap did what it claims.
4. Eval: fork `llama-perplexity`, `-ngl 99 -c 512 --chunks 8`,
   `tinyshakespeare.txt`, ROCm1, one process at a time.

## Numbers

| model | PPL (8×512) | notes |
|---|---|---|
| Prism `Ternary-Bonsai-2-27B-PQ2_0` | **18.5851 ± 1.209** | source artifact |
| ours = 402 payloads RTN, same file | **23606.19 ± 1231.6** | ratio **1270×** |
| smoke: layers 0+3 patched only (2 chunks) | 53.89 vs 20.41 | 2.6× from 13/402 tensors |

The smoke result localizes the blow-up: replacing layer 0 (GDN/recurrent) plus
layer 3 (full attention) costs 2.6×, and the full swap explodes. Small weight
perturbations in the recurrent path amplify over the sequence; the rest is the
cumulative ternary-PTQ collapse documented at 1.7B (T10, 3514×).

## Consequences

- **No-rental parity is falsified.** 92% trit agreement is not a quality proxy:
  the missing 8% is where Prism's value lives (T30 fingerprinted it as
  error-compensated/QAT weights, ~10% lower weight-space error than RTN).
- **The corrected basis is still the big win.** Every future candidate (GPTQ,
  QAT/KD, distillation) should run in T26's basis and T30's layouts; the old
  PRF basis + wrong layouts would have made any comparison meaningless.
- **Next levers, in order:**
  1. **T4 GPTQ in the corrected basis** with real calibration Hessians
     (T24 capture / T17 cache). The public ThakiCloud QuIP is 1.2e3–3.5e4× at
     1.7B, so plain GPTQ may not be enough either — but at 27B with Prism's
     exact basis it is the cheapest falsifiable step.
  2. **T28 QAT/KD initialized from this artifact** (better start than PRF-RTN;
     the 1.7B run already reached 1.10× with KD).
  3. **T29** stays the pragmatic end-state on their released weights.
- **Spec follow-up (T25 escalation, now with evidence):** our `quantize()` LS
   refinement *loses* to plain absmean on this artifact (0.85–0.87 vs
   0.90–0.95). For PQ2_0-class runs the LS refine should be optional.

## Caveats

One corpus (1.1 MB Shakespeare), one binary, 8×512 context, single run each;
PPL of a broken model is noisy (±1231 on 23606). The direction and magnitude are
far outside that noise, and the smoke gradient (2.6× from 13 tensors) is
monotone with the swap size. KLD against their logits would sharpen the number
but cannot change the verdict.

Tags: [verified] for the PPL and hashes above; [inferred] for the
recurrent-amplification reading of the smoke run.
