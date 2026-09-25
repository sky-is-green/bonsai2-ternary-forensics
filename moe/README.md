# MoE ternary harness

Experiment harness behind [`../docs/MOE-EXTENSION.md`](../docs/MOE-EXTENSION.md):
in-place ternary quantisation of a pretrained Mixture-of-Experts model, the
router-drift probe, and the low-rank correction-branch trainers ("Doctors").

Two targets:

- **`allenai/OLMoE-1B-7B-0924`** — the frozen proxy body (16 layers, 64 experts,
  top-8). Everything trains here.
- **a `qwen3_5_moe`-class 35B-A3B prefix** — the real target; only the target
  census, role map, and routing-drift probe run against it in this harness.

## Layout

| file | role |
|---|---|
| `gguf_header.py` | remote GGUF header census over HTTP range requests |
| `role_map.py` | MoE role map + ternary projection (8.82 GiB / 2.186 bpw for the 35B distill) |
| `probe_router.py` | E1 routing-drift prefix probe |
| `olmoe_proxy.py` | in-place ternary QAT + teacher cache + KD loop + eval |
| `moe_proxy.py` | MoTE-style up-cycle proxy (Qwen3-1.7B) |
| `olmoe_rotate_rtn.py` | rotation vs RTN quantizer comparison |
| `olmoe_doctors.py` | per-layer residual-stream correction branches (route A); STE branch formats and mixed-precision sidecars |
| `olmoe_experts.py` | per-expert correction branches (placement control) |
| `eval_ckpts.py` | checkpoint trajectory + router diagnostics |
| `save_ternary_olmoe.py` | materialise a ternary build to an HF dir |
| `qwen35_moe_proxy.py` | 35B-A3B (`qwen3_5_moe`) port: fused-bank STE patch, prefix smoke, cache/train/eval stages |
| `branch_sensitivity.py` | per-layer sidecar sensitivity scan (which layers a mixed sidecar should keep fp16) |

`results/` holds the JSON evidence quoted in the write-up: per-run 8-window
evals, the E1 routing probe, the rotation comparison, the AUTOGRID scans, and
the 35B role maps.

[`PORT-QWEN35.md`](PORT-QWEN35.md) is the plan for the real target
(`qwen3_5_moe`, 35B-A3B): validated target inventory, required harness
changes, cache/build/correction plan, and the local smoke-test sequence.

## Setup

All scripts read and write one artifact root, `$MOE_ARTIFACTS` (default:
`moe/artifacts/`, gitignored):

```
$MOE_ARTIFACTS/
  olmoe-hf/     FP OLMoE-1B-7B (HF dir)
  olmoe/        outputs: teacher-cache.pt, correction checkpoints, eval JSONs
  canary-hf/    Qwen3-1.7B dense base (only needed by moe_proxy.py)
  empero-hf/    35B prefix checkpoint (only needed by probe_router.py)
```

```sh
pip install -r ../requirements.txt -r ../requirements-gpu-lock.txt
pip install -e ..          # bonsai_forensics (rotation/quant helpers used here)
export MOE_ARTIFACTS=$PWD/artifacts
huggingface-cli download allenai/OLMoE-1B-7B-0924 \
    --local-dir $MOE_ARTIFACTS/olmoe-hf
```

## Run

```sh
# 1. teacher cache: top-50 logits + router top-8 for the KD loss
HIP_VISIBLE_DEVICES=1 python olmoe_proxy.py cache \
    --windows 4096 --corpus-chars 50000000 --device cuda:0

# 2. per-layer residual-stream corrections (route A); both cards for the
#    brief teacher/student coexistence
HIP_VISIBLE_DEVICES=0,1 python olmoe_doctors.py train --device-map auto \
    --rank 512 --windows 4096 --corpus-chars 50000000 --epochs 1 \
    --lr-half-every 500 --lr-decay-start 2000 --router-weight 0

# 3. held-out 8-window eval; single card
HIP_VISIBLE_DEVICES=1 python olmoe_doctors.py eval --rank 512 \
    --load $MOE_ARTIFACTS/olmoe/olmoe-doctors-r512-step4096.pt

# placement control: corrections inside the experts instead of the stream
HIP_VISIBLE_DEVICES=0,1 python olmoe_experts.py train --device-map auto ...
```

The cache build and the training run must use the same `--windows` /
`--corpus-chars` / `--seed`, or the cached teacher targets will not line up
with the student inputs. Branch formats: `--branch-quant {fp32,g128,rank}`
(STE-trained when not fp32; `rank` is the TAARDIS V3 per-rank-component
scheme).

## Ops notes

- One heavy GPU process at a time. Training needs both cards only for the
  brief teacher/student coexistence (`device_map auto` with the `max_memory`
  caps in the script); single-card stages (cache, eval) should pin the free
  card, not the display card.
- Run heavy jobs under `systemd-run --user --scope -p MemoryMax=..`; kill by
  PID, never by pattern.
- Corrections and masters must be fp32: bf16 masters swallow adapter-scale
  updates (bulk update RMS ~5e-6 vs bf16 ULP ~6e-5).
- With `device_map="auto"` the pre-patch experts `forward` is bound onto each
  instance; rebind through `patch_experts(group, model)`.
- `F.kl_div(..., reduction="batchmean")` on a 3-D tensor divides by batch only;
  reshape to `[-1, k]` first.
- The router-KD term is redundant on this stack: a `--router-weight 0` control
  matched the KD-on trajectory within ~1.5% at every checkpoint.
