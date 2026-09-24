# The recipe against the 2026 quantization landscape

**Purpose.** Place the ternary QAT/KD recipe (rotation + per-group absmean
ternary STE + teacher KD) on the same axes as the community-standard
quantization methods, so we stop comparing labels ("2-bit") and start comparing
*measured* bits per weight on the same base model, corpus, and metric.

**Provenance.** Community figures are standard, well-known method properties
(gathered 2026-09-24, external); our own numbers are computed from
`inspect_model_targets.py` target censuses in `artifacts/preflight/` and are
reproducible offline. Nothing here changes a scientific claim; it bounds one.

---

## 1. Effective bits per weight

"Effective bpw" counts **every** parameter, including anything kept at higher
precision, because that is what determines file size and memory:

```
bpw_eff = (Σ_quantized N·bpw + Σ_kept N·16) / N_total
```

| method | effective bpw | PTQ / QAT | needs calibration | custom kernel/runtime |
|---|---:|---|---|---|
| BitNet b1.58 (I2_S ternary) | **~2.0 packed** (1.58 bits entropy) | QAT | no | yes (`bitnet.cpp`) |
| GGUF `Q2_K` | **2.625** | PTQ | imatrix strongly advised | llama.cpp |
| EXL2 (mixed, e.g. `2.0bpw`) | **2.0–8.0** per-layer | PTQ | yes | ExLlama (NVIDIA) |
| GGUF `Q3_K_M` | **3.94** | PTQ | imatrix advised | llama.cpp |
| bnb NF4 + double-quant | **~4.13** | PTQ | no (data-free quantiles) | bnb |
| GPTQ / AWQ 4-bit g128 | **4.125** | PTQ | yes (128–1024 samples) | ExLlama/Marlin/vLLM |
| GGUF `Q4_K_M` | **4.85** | PTQ | optional | llama.cpp |
| GGUF `Q6_K` | **6.56** | PTQ | no | llama.cpp |
| bnb LLM.int8() | **~8.0** (int8 + fp16 outliers) | PTQ | no | bnb |
| GGUF `Q8_0` | **8.5** | PTQ | no | llama.cpp |
| **our recipe (as implemented)** | **3.35–5.66** (see §2) | QAT + KD | no (teacher logits) | rotation absorbed; ternary GEMM |

Two clarifications that matter for honesty:

- **"1.58-bit" is an information rate, not a storage width.** Packed ternary is
  ~2 bpw in practice; the entropy is 1.58 bits per weight.
- **"4-bit" is not one number.** bnb NF4+double-quant (~4.13) < GPTQ/AWQ g128
  (4.125) < GGUF `Q4_K_M` (4.85). Quote the class, not the digit.

---

## 2. Our actual effective bpw is not ~2

The recipe ternarizes the **selected decoder linears** and leaves the token
embedding (and norms) at FP16. On Qwen3, `lm_head` is tied to the embedding, so
the FP16 remainder is essentially `vocab × hidden`. The measured censuses:

| model | total params | ternary | FP16 | **effective bpw** | ternary tensors |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 0.596B | 0.440B | 0.156B | **5.66** | 196/196 |
| Qwen3-1.7B | 1.721B | 1.409B | 0.311B | **4.53** | 196/196 |
| Qwen3-4B | 4.022B | 3.633B | 0.389B | **3.35** | 252/252 |

The FP16 remainder is `151936 × hidden` in every case (the Qwen3 vocab), which
is why the overhead is *worse at smaller scale*: the embedding is a larger
fraction of a smaller model. **A ~2 bpw claim is only valid once the embedding
is also ternarized.**

This is not a cosmetic accounting point. The released Prism/Bonsai-2 format
packs the embedding as the **402nd PQ2_0 tensor** — the ledger's structural
table reports `embed/output` sparsity ≈ 0.309 (Gaussian baseline) for Prism's
line, i.e. their embedding *is* ternary. Our current recipe is therefore
**larger than the thing it is trying to reproduce**, by ~0.3B parameters at
1.7B and ~0.39B at 4B. Two consequences:

1. Any "our model is smaller/equal" statement must use the measured bpw, not
   the ternary-only count.
2. Ternarizing the embedding (with its separate inverse-rotation path) is a
   concrete, missing piece of Bonsai-2 parity — see
   [`REPRODUCIBILITY-AUDIT.md`](REPRODUCIBILITY-AUDIT.md) P0.

`docs/RECIPE-LEDGER.md` and `docs/RETENTION-VS-SCALE.md` should report
`effective bpw` beside every retention number.

---

## 3. Fair comparison framework

1. **Matched effective bpw over all parameters**, computed as in §1 — never the
   ternary-only fraction.
2. **Same base model and revision** (e.g. `Qwen/Qwen3-1.7B`, pinned).
3. **Same corpus and tokenizer.** PPL is tokenizer-dependent; compare
   bits-per-byte, or always normalize by the FP teacher on identical bytes.
4. **Teacher-relative metric.** Report retention *and* absolute PPL, one
   downstream task, and **KLD** (more sensitive to tail damage than mean PPL).
5. **Runtime equivalence.** Memory ≠ speed. Ternary in `bitnet.cpp`, GGUF in
   llama.cpp, AWQ/GPTQ in Marlin — otherwise you benchmark kernels, not methods.
6. **No calibration leakage.** Our recipe uses teacher logits, not a corpus
   prefix; GGUF/GPTQ/AWQ/EXL2 use calibration data. Keep the calibration set
   disjoint from evaluation.

The recipe's structural edge is exactly this last point: it needs **no
calibration corpus**, only the same-architecture teacher's logits.

---

## 4. Unsloth (2026)

Unsloth ships three distinct things:

- **Dynamic GGUF** (`UD-Q2_K_XL`, `UD-Q4_K_XL`, …): per-tensor mixed precision,
  larger calibration set, runs on **stock llama.cpp**. Inference only.
- **Dynamic 4-bit safetensors** (`*-unsloth-bnb-4bit`): bnb NF4 with selective
  tensors left unquantized; usable for 4-bit LoRA fine-tuning.
- **4-bit fine-tuning** (LoRA on those safetensors).

Unsloth's contribution is **per-layer bit allocation**, not a new datatype; it
is a quality-per-bit improvement on existing GGUF kernels. Natural ~2 bpw peers
for us: `UD-Q2_K_XL` and `IQ2_M`; bnb NF4 (~4.13) and `Q8_0` (8.5) bracket from
above.

```sh
# ~2 bpw peer and a 4-bit peer for the same base model
hf download unsloth/Qwen3-1.7B-GGUF --include "*UD-Q2_K_XL*" --local-dir qwen3-1.7b-gguf
hf download unsloth/Qwen3-1.7B-unsloth-bnb-4bit --local-dir qwen3-1.7b-bnb4
llama-cli -m qwen3-1.7b-gguf/*UD-Q2_K_XL.gguf --ctx-size 8192
```

---

## 5. Can the recipe shrink an already-quantized model? (e.g. Q8)

The recipe needs two things a quantized model does not cleanly provide: a
**high-precision teacher** for KD, and a **trainable continuous master** for STE.

- **Q8 is near-lossless.** Dequantizing Q8_0 → BF16 leaves a tiny, roughly
  zero-mean, block-correlated error. The ternary fit is not materially damaged,
  so **Q8→ternary ≈ BF16→ternary** within noise.
- **The ternary step still needs BF16/FP32 masters** (STE), so starting from Q8
  only halves the *initial download/storage* footprint. It buys nothing at
  training time.
- **Rotation × pre-quantization.** The rotation is orthogonal and preserves
  norms, but applying it to a **k-quant** base smears that base's structured
  per-block error across channels — dense and unrecoverable. On Q8 the error is
  negligible; on Q4/Q3 it is not. **Always rotate the BF16 graph, never a
  Q4/Q3 tensor.**
- **Quantized-teacher distillation** is the genuinely useful variant: an 8-bit
  teacher is functionally indistinguishable from BF16, so it can be used to fit
  a teacher that would not otherwise fit on a 20 GiB card. A 4-bit teacher
  **caps** the student at the teacher's (already degraded) accuracy.
- **Ceiling.** You cannot shrink an already-lossy model and claim FP parity: the
  ceiling is the Q8 model, not the base.

**Verdict: start from BF16.** Q8-dequant is a legitimate, near-identical,
memory-motivated stand-in when BF16 is unavailable; Q4/Q3 is not advisable.

---

## 6. Concrete experiment (one GPU-day on one card)

On `Qwen/Qwen3-1.7B`, the identical ternary QAT/KD recipe (same seed, corpus,
schedule), three paths:

| arm | student init | teacher | question |
|---|---|---|---|
| A | BF16 | BF16 | baseline |
| B | Q8_0-dequant BF16 | BF16 | does the dequant path hurt? |
| C | BF16 | Q8-dequant BF16 | is a quantized teacher safe? |
| D | BF16 | Q4-dequant BF16 | where does the teacher ceiling bite? |

Report retention, ΔPPL, KLD, and **effective bpw** against `UD-Q2_K_XL` and
`IQ2_M` on the same bytes. Prediction: **A ≈ B**, **C ≈ A**, **D < A**.

---

## 7. Bottom line

- **Our real bit budget is 3.35–5.66 bpw, not ~2**, until the embedding is
  ternarized (Prism packs it as the 402nd tensor).
- **Fair ~2 bpw peers are only BitNet b1.58, GGUF `Q2_K`/`IQ2_M`, and EXL2
  @2.0bpw.** Q4–Q8 (4.1–8.5 bpw) are not peers.
- **The recipe's edge is calibration-free QAT**, versus calibration-dependent
  PTQ (GPTQ/AWQ/EXL2/GGUF imatrix).
- **Unsloth = per-layer bit allocation**, not a new datatype; stock llama.cpp.
- **Q8 is effectively lossless**, so Q8→ternary is a valid BF16 stand-in and a
  memory optimization, but not a training-time shortcut.
- **8-bit teachers are safe; 4-bit teachers cap retention.**
- **Rotate BF16, never Q4/Q3.**
- **Speed claims require matched kernels.**
