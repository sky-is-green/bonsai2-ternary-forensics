# RUN-CANARY — T10 1.7B end-to-end gate

**Verdict: FAIL (quality criterion); PASS as a bug gate.** Do not rent the 27B
GPU until the GPTQ quality gap below is closed.

## Run

| Item | Value |
|---|---|
| Model | `Qwen/Qwen3-1.7B` (bf16, the Prism 1.7B base) |
| Corpus | `artifacts/canary/tinyshakespeare.txt`, sha256 `86c4e6aa…` |
| Calibration | 32 windows × 2048 tokens (A-style deterministic windows) |
| Eval | 4 held-out windows × 2048 tokens |
| Device | RX 7900 XT (HIP torch 2.11), GPTQ on CPU |
| Timing | capture 295 s (197 Hessians), quantize 524 s (311 tensors, 198 via GPTQ) |
| Artifact | `artifacts/canary/canary-tq2_0.gguf` (TQ2_0 packer) |
| Report | `artifacts/canary/canary-report.json` |

## Results (next-token KLD/PPL vs the reference model, mean over 4 windows)

| Arm | KLD | PPL | Ratio vs ref |
|---|---|---|---|
| Reference (bf16) | — | 39.95 | 1.0× |
| **Ours** (rotation + fold + GPTQ) | **8.83** | **1.40e5** | **3514×** |
| Naive (absmax RTN, no rotation) | 16.27 | 1.04e8 | 2.6e6× |

Component scan (absmean+LS ternary, no rotation, one window, ref PPL 51.6):
`o_proj` 133 · `layer0_all` 406 · `down` 5.2e5 · `qkv` 1.3e5 · `gate_up` 3.3e7.
Control: int8-group256 over every linear → PPL 52, so the replacement/eval path
is sound and the collapse is genuinely the ternary codec.

Interpretation: ours beats naive by 21% KLD but is nowhere near the public
reference (ThakiCloud QuIP: 1.1×–2.1× PPL on this base). The rotation is applied
and GPTQ runs, but per-layer output error is still 4–10%, which compounds over
28 layers. The GPTQ gap (vs QuIP's reference implementation) is the blocker.

## Bugs found and fixed by this canary

1. **`q_norm`/`k_norm` were classified as hidden norms** (they end with
   `norm.weight`) and would have been stored as all-ones. Fixed in spec
   **tbr-1.2** (`hidden_norm_exact` split; head norms excluded first).
2. **`rotate_hessian` computed `Rᵀ H R` instead of `R H Rᵀ`** (`R` is asymmetric
   when `S ≠ 1`), so GPTQ optimized against the wrong Hessian in the rotated
   basis. Fixed + pinned by `test_rotate_hessian_is_r_h_r_transpose`.
3. **Canary recovery used the wrong transpose** (`Rᵀ` vs `R` per role direction);
   fixed + pinned by `test_recover_weight_inverts_both_absorbed_edges`.

Also noted for the 27B export (T12/T13): real GGUFs need HF→GGUF tensor-name
mapping and tokenizer metadata before llama.cpp can serve the artifact; the
canary evaluates by round-trip in PyTorch to keep those concerns separate.

## Next (see HIVE-PLAN T27/T28)

- T27: reproduce the ThakiCloud QuIP reference on this base as a working
  quality bar, then diff our GPTQ Hessian/scale/compensation layer-by-layer.
- T28: sweep `damp` / `act_order` / salience (PB-LLM top-k%) and, if needed,
  adopt QuIP-style error compensation wholesale; re-run this canary.
