# Model registry and architecture test plan

The canonical machine-readable inventory is
[`configs/model_registry.yaml`](../configs/model_registry.yaml).  It pins the
Hub revision, architecture, target profile, and local feasibility decision for
each candidate.  This page records the reasoning behind the order.

## Constraints on the current node

Measured 2026-09-24: two AMD Radeon RX 7900 XT cards (`gfx1100`), with
`21,458,059,264` bytes (~19.98 GiB) usable per card.  The current PyTorch build
is ROCm 7.2 / Torch 2.11.0.  A student and teacher are run on separate cards
for the larger pilots.

The Qwen3.8 family is not a set of small Qwen3 checkpoints:

| model | architecture | BF16/FP size | decision |
|---|---|---:|---|
| Qwen/Qwen3.8-27B | `Qwen3_5ForConditionalGeneration`, hybrid Gated-DeltaNet/full attention | 55.6 GB | architecture port; not full-resident here |
| Qwen/Qwen3.8-27B-FP8 | same | 30.9 GB FP8 | reference only; not a clean FP teacher |
| Qwen/Qwen3.8-Flash-Next | `Qwen4ExpForConditionalGeneration`, MoE/vision/indexer | 360 GB | deferred architecture project |
| Qwen/Qwen3.8-2.4T-A95B | `Qwen3_5MoeForCausalLM`, 512 experts | 4.89 TB | deferred |

There is no official Qwen3.8 0.6B/1.7B/4B checkpoint in the inspected Hub
inventory.  The old `Qwen/Qwen3.8-0.6B` config entry was invalid and has been
corrected to pinned `Qwen/Qwen3-0.6B`.  (The *Qwen3.5* family does have a small
series — 0.8B/2B/4B — and that series shares the 27B target's architecture; see
the next section.)

## Qwen3.5 small series — target-architecture stand-ins

`Qwen/Qwen3.8-27B` is a `qwen3_5` hybrid, and the Qwen3.5 small series shares
that architecture.  These are the tractable way to exercise the 27B port
(Gated-DeltaNet `in_proj_qkv`/`in_proj_z`/`out_proj`, text-only extraction,
input-axis rotation) before a larger allocation exists.  Each was verified with
`inspect_model_targets.py` on a pinned revision: `target_count_ok` is true and
the profile selects only the text decoder.

| model | revision | params | BF16 | text layers (lin-attn/full) | selected | input widths |
|---|---|---:|---:|---|---:|---|
| `Qwen/Qwen3.5-0.8B` | `2fc06364…` | 853M | 1.71 GB | 24 (18/6) | 150/150 (0.497B) | 1024, 2048, 3584 |
| `Qwen/Qwen3.5-2B` | `15852e8c…` | 2.21B | 4.43 GB | 24 (18/6) | 150/150 (1.372B) | 2048, 6144 |
| `Qwen/Qwen3.5-4B` | `851bf6e8…` | 4.54B | 9.08 GB | 32 (24/8) | 200/200 (3.565B) | 2560, 4096, 9216 |

The vision tower and the `in_proj_a`/`in_proj_b` recurrent controls are
intentionally outside the target set (they stay FP, matching the released
format's F16 control set).  `lm_head` is tied, so it also stays FP.  Use the 2B
as the primary architecture stand-in and the 4B as the matched-size control
against the legacy Qwen3-4B rung.

## First wave

### 1. SmolLM2-360M — smoke

- `HuggingFaceTB/SmolLM2-360M`
- revision `f8027fd0eaeea54caa13c31d31b9fdc459c38b49`
- `LlamaForCausalLM`, 362M parameters, ~0.72 GB BF16
- Tests the real loader, checkpoint round trip, optimizer, and output shape
  before a long run.
- Its 960/2560 input widths also exercise the quantizer's right-padding edge
  case with g128.

### 2. SmolLM2-1.7B — primary control

- `HuggingFaceTB/SmolLM2-1.7B`
- revision `effd688a12921b4cc83e3312b6feb579f70f9c71`
- `LlamaForCausalLM`, 1.711B parameters, ~3.42 GB BF16, Apache-2.0
- 24 layers, hidden 2048, intermediate 8192, MHA, tied embeddings/head.
- Profile selects 168 linears / 1.611B projection parameters; tied `lm_head`
  stays FP.
- This is the cleanest size-matched comparison to the Qwen ladder, but it is
  still a Llama-family control rather than proof of broad generalization.

### 3. OLMo-2-1B-Instruct — architecture diversity

- `allenai/OLMo-2-0425-1B-Instruct`
- revision `48d788eca847d4d7548f375ad03d3c9312f6139e`
- `Olmo2ForCausalLM`, 1.485B parameters, ~2.97 GB BF16, Apache-2.0
- 16 layers, hidden 2048, intermediate 8192, untied `lm_head`, post-norm/QK
  normalization structure.
- Profile selects 113 linears / 1.279B parameters, including the head.
- This is the best first non-Llama quality/structure comparison.

### 4. Pythia-1.4B — tensor-name/parallel-residual stress

- `EleutherAI/pythia-1.4b`
- revision `fedc38a16eea3bd36a96b906d78d11d2ce18ed79`
- `GPTNeoXForCausalLM`, 1.414B parameters, ~2.93 GB safetensors, Apache-2.0
- Fused `query_key_value`, `dense`/`dense_h_to_4h`/`dense_4h_to_h`, parallel
  residual, partial RoPE, affine norms, and untied `embed_out`.
- Default profile selects 97 linears / 1.311B parameters.  Use the explicit
  `--target-suffixes` plus `--no-include-lm-head` form for a projection-only
  control if desired.
- It is an older model, so use it for mechanics and robustness, not as a
  modern quality leaderboard claim.

## Second wave: fused Phi

Use `microsoft/Phi-3-mini-4k-instruct` first (revision
`f39ac1d28e925b323eae81227eaba4464caced4e`) rather than the 131K-context
Phi-3.5-mini sibling. Both are `Phi3ForCausalLM`, MIT, ~3.821B parameters, and
use fused `qkv_proj`/`gate_up_proj` plus an untied `lm_head`.  The profile
selects 129 linears / ~3.722B parameters.

Run a 100–500 step memory/forward smoke on separate cards before a 5k pilot.
The long-context sibling (`2fe192450127e6a83f7441aef6e3ca586c338b77`) is a
later model-specific forward/configuration test.

## Rotation policy by experiment

The first cross-architecture runs use `--rotation-mode input`: rotate each
selected linear's input/last axis, which is the closer edge-local proxy to the
Prism storage convention.  A matched `--rotation-mode residual` control is
needed if comparing against the historical Qwen ladder.  Do not attribute a
difference between those modes to architecture alone.

The input widths and effective PRF blocks for the first wave are:

| model | input widths | blocks |
|---|---|---|
| Qwen3.5-0.8B | 1024, 2048, 3584 | 1024, 1024, 1024 |
| Qwen3.5-2B | 2048, 6144 | 1024, 1024 |
| Qwen3.5-4B | 2560, 4096, 9216 | 512, 1024, 512 |
| SmolLM2-360M | 960, 2560 | 128, 256 |
| SmolLM2-1.7B | 2048, 8192 | 1024, 1024 |
| OLMo-2-1B | 2048, 8192 | 1024, 1024 |
| Pythia-1.4B | 2048, 8192 | 1024, 1024 |
| Phi-3 | 3072, 8192 | 1024, 1024 |

## Fair comparison rules

1. Pin model and tokenizer revision; record the manifest.
2. Use the same corpus bytes, sequence length, update budget, KD temperature,
   fixed rotation seed, and managed schedule.
3. Run a config/meta target census before downloading weights.
4. Compare within-model teacher-relative PPL ratio/retention, ΔPPL, KL, and
   common benchmark accuracy. Raw PPL across tokenizers is not commensurate.
5. Report selected parameter fraction, target count, basis digest, peak VRAM,
   sampled-window hash, and whether the result is in-memory or packed.
6. Start with a 500–1,000-step smoke; only a 20k convergence run is evidence
   for architecture robustness.

## Commands

```sh
# Preflight only: no weights, no GPU.
python scripts/pilot/inspect_model_targets.py \
  --model HuggingFaceTB/SmolLM2-1.7B \
  --revision effd688a12921b4cc83e3312b6feb579f70f9c71

# Preflight through the same runner without starting training.
MODEL=HuggingFaceTB/SmolLM2-1.7B \
MODEL_REVISION=effd688a12921b4cc83e3312b6feb579f70f9c71 \
TAG=smollm2-1.7b-preflight PREFLIGHT_ONLY=1 \
  ./scripts/pilot/run_cross_arch_pilot.sh

# Explicit pilot; do not run while another job owns the selected card.
MODEL=HuggingFaceTB/SmolLM2-1.7B \
MODEL_REVISION=effd688a12921b4cc83e3312b6feb579f70f9c71 \
TAG=smollm2-1.7b-input VIS=0 \
  ./scripts/pilot/run_cross_arch_pilot.sh

# Qwen3.5-2B: the target-architecture stand-in.  VIS is derived from the
# student/teacher placement (student cuda:0, teacher cuda:1 -> both visible).
MODEL=Qwen/Qwen3.5-2B \
MODEL_REVISION=15852e8c16360a2fea060d615a32b45270f8a8fc \
TAG=qwen3.5-2b-smoke STEPS=500 PROJECT_EVERY=50 LOG_EVERY=50 \
ROTATION_MODE=input DEVICE=cuda:0 TEACHER_DEVICE=cuda:1 \
  ./scripts/pilot/run_cross_arch_pilot.sh
```

For a projection-only Pythia arm, add:

```sh
PROFILE=gpt_neox \
TARGET_SUFFIXES=query_key_value,dense,dense_h_to_4h,dense_4h_to_h \
NO_INCLUDE_LM_HEAD=1 \
ALLOW_TARGET_COUNT_MISMATCH=1
```

The generic runner accepts the explicit override through the environment; the
preflight and training command receive the same suffix/head policy.
