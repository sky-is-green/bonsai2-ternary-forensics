# The MoE extension: does the ternary recipe survive routing?

Status: research note, 2026-09-25.  This document re-frames the dense-27B
forensics (Parts 1 and 2 of the public write-up) onto the question that
follows it: **can ternary-class compression reach Bonsai-level retention on a
Mixture-of-Experts model?**

The dense analysis is settled and is not repeated here (see `WHITEPAPER.md`,
`FORENSIC-ARCHIVE.md`, `FAILURES.md`).  The one-line summary that matters for
this extension: the container is public, the residual is ordinary
rotation-in-the-loop ternary QAT, and the public PTQ route collapses
off-calibration (1,178x to ~35,000x).  Quality is trained placement.

## 1. Why MoE is a different problem

Dense models have one failure mode under ternarisation: the per-weight
approximation error, absorbed by the redundancy of the whole network.  MoE
models have two, and the second one is not a capacity problem:

1. the expert weights are ternarised (same as dense), and
2. **routing is a discrete decision** made from the hidden state, so any drift
   in that state flips expert selection and compounds downstream.

97% of MoE weights are dormant per token, so per-token gradient coverage is
sparse, and top-k selection is non-differentiable.  Neither property exists in
the dense case.

## 2. Measurements (all local, artifacts under `hivebench/artifacts/ternary/moe`; the harness lives in the private research tree, see §5)

### 2.1 Routing drift is real and compounds

Prefix probe on a 35B-A3B-class target (`qwen3_5_moe`, 256 experts, top-8 +
shared; embedding + layers 0-3; experts rotate+absmean RTN at g128; router and
attention exact):

| layer | router top-8 agreement vs FP | hidden-state rel L2 drift |
|---|---|---|
| 0 | 1.000 | 0.202 |
| 1 | 0.807 | 0.311 |
| 2 | 0.760 | 0.346 |
| 3 | 0.652 | 0.373 |

Expert weight rel err 0.511, zero share 0.309.  Layer 0 cannot flip (its input
is the exact embedding), so 1.000 is a correctness check.  From layer 1 the
router weights are still exact; the flips come entirely from hidden drift.

### 2.2 Rotation is not the missing lever for MoE

Rotating the expert banks before RTN does not improve the quantizer:
weight rel err 0.5156 (raw) vs 0.5109 (rotated), zero share 0.309 in both.
In-place RTN on OLMoE-1B-7B: PPL 11.02 -> 40,121 (3,640x) raw, 56,611 (5,135x)
rotated.  The archive's own Gate-2 result (Prism's exact basis + RTN on the
dense 27B -> 1,270x) predicted this: the basis buys the container, the trained
placement buys the model.

### 2.3 In-place QAT at local budgets does not recover it

Full-model bf16 QAT (6.4B trainable, 0.5M tokens): 1.9x recovery, routing
unchanged (0.463 -> 0.464).  fp32 masters on a 4-layer subset removed the
precision question but training still did nothing (23.94 -> 23.79).  bf16
masters partially swallow the updates (bulk update RMS ~5e-6 vs bf16 ULP
~6e-5), but precision is not the binding constraint.

### 2.4 Trained corrections recover it, and placement decides how much

TAARDIS-style low-rank correction branches on the same frozen RTN body, same
loss (LM + output KD + router KD), same 1024 steps unless noted:

| arm | trainable | PPL (8 windows) | router agreement |
|---|---|---|---|
| RTN, no corrections | - | 40,121 | 0.463 |
| per-layer branches (residual stream), rank 64 | 6.29M | 253.85 | 0.506 |
| per-layer branches, rank 256, 2048 steps | 18.9M | 85.50 | 0.674 |
| **per-layer branches, rank 512, 4096 steps** | **35.7M** | **71.14** (best, step 2000; 75.64 at 4096) | **0.710** (0.726 at 4096) |
| per-expert branches (inside experts), rank 8 | 52.4M | 6,551.74 | 0.425 |

Trainable counts include the trainable routers (2.1M of the totals).  The
per-expert arm was evaluated in a separate pass whose RTN baseline was 39,727;
re-evaluating the same frozen body at this collapsed scale moves the number by
about 1%.

8x more parameters inside the experts, 25x worse PPL, and routing ends below
the untrained baseline.  Mechanism: routing is decided from the residual stream
entering the MoE block, so a correction there can steer the next layer's
decisions; a correction inside an expert acts after the decision and cannot
influence it.

**Capacity in the right place keeps helping, with sharply diminishing
returns.**  Rank 64 -> 256 took PPL from 253.85 to 85.50 and routing agreement
from 0.506 to 0.674; rank 512 improves the best 8-window PPL again, to 71.14
(564x recovery from the in-place collapse), on 35.7M trainable parameters
(33.6M branches plus routers) against a 6.4B frozen body.  But the rank-512
run turns over: PPL is 71.14 at step 2000 and 75.64 at step 4096, while router
agreement keeps climbing (0.710 -> 0.726).  More capacity buys a lower floor;
the last stretch is not a capacity problem.  At the rank-512 best the teacher
gap is 6.5x (11.02 vs 71.14), and the best checkpoint is not the last one.

**Design rule: for the routing bottleneck, corrections must live on the
residual stream, not inside the experts.**  TAARDIS's per-matmul placement was
derived on a dense model with no routing, so it does not transfer.

### 2.4b The AUTOGRID map (noise-floor classification)

CodeMasterCody3D/autogrid (MIT) classifies every tensor as FREE@k (conversion
quieter than the container's own rounding), TERNARY (already on the grid), or
STEER (needs a steered pipeline).  Run locally on OLMoE:

| model | max-k | result |
|---|---|---|
| OLMoE FP | 8 | 100% free, recommended k=6 (err 2.3e-3 vs bf16 floor 3.9e-3) |
| OLMoE FP | 2 | **100% STEER** (0 free, 0 ternary) |
| OLMoE ternary build | 2 | **93.1% already ternary** (the 3,072 expert tensors, 6.44B params); 6.9% STEER (attention, embeddings, head, norms, router, left FP by design) |

The k=2 row states the same boundary this document reaches from the other
side: at ternary budgets nothing converts for free, and the STEER set is
exactly what the correction pipeline has to carry.

### 2.5 Where the cheap route already works

MoTE-style up-cycling (pretrained FFN kept as a frozen BF16 shared expert,
ternary routed experts on top): 1.048x with **zero training**, and training the
experts further on a small corpus made it worse (1.122x).  The frozen FP
component carries the function; the ternary experts only learn a correction.
For Qwen3.6-35B-A3B this is not directly applicable (its shared expert is
~0.4% of weights), so it implies re-architecture rather than in-place work.

## 3. Prior art, and the cliff

- **APEX** (localai-org/apex-quant): MoE-aware mixed-precision PTQ; 35B-A3B at
  10.9-12.2 GB with ~98% benchmark retention.  Bonsai-class retention on a MoE
  already exists, at ~2.8 bpw rather than ~2.2.
- **MoTE** (arXiv 2506.14435): ternary routed experts on a frozen BF16 shared
  expert; matches/beats a full-precision MoE baseline at 1.5B and 3B.  Shared
  expert precision is worth 7.6 points.
- **DeepSeek-V4-Flash** (284B total, 13B active, natively FP8/FP4, heavily
  trained): per the MLX conversion measurements, **3-bit experts collapse the
  model (PPL 1.4e7, degenerate loops)** and 4-bit sits just above a cliff.  The
  same source finds non-expert weights ~50x more quantization-sensitive per
  parameter.  Over-training and parameter count do not buy tolerance at these
  bit widths.
- **EAQuant / ExpertQuant / RSPO**: independent confirmation that router
  sensitivity dominates MoE quantization loss, and that rank flips around the
  top-k boundary are the mechanism.

## 4. The research path

The target: `qwen3_5_moe` (35B total, 3B active), ternary experts, correction
sidecar, evaluated teacher-relative on held-out windows with the archive's
fairness rules.

| route | mechanism | cost | status |
|---|---|---|---|
| A | in-place ternary + trained residual-stream corrections | single 80 GB card or 2x40 GB for the correction run; no 263 GB fp32-master QAT | placement rule established; rank scaling flat (best 71.14 at rank 512); loss/checkpoint selection is the open axis |
| B | MoTE-style re-architecture (frozen FP component carries the function) | cheap training, larger artifact | demonstrated at 1.5B/3B |
| C | MoE-aware mixed-precision PTQ (APEX-style) | cheapest, ~2.8 bpw | published, Bonsai-class retention |

Route A is the one this work opens: it reaches ternary bit budgets without the
4xH100 QAT bill, because only the corrections train.  Open items:

1. correction capacity and rank allocation on the residual stream (rank 512
   reaches a best 8-window PPL of 71.14 at step 2000, then turns over; the
   rank/step curve has flattened, so the next question is loss and checkpoint
   selection rather than more capacity),
2. router-aware losses: the router KD term did not improve agreement at the
   budgets tried, so the correction has to repair the state the router reads,
3. ternarising the correction sidecar (TAARDIS ships branches at 2 bits per
   factor) so the artifact stays near 2 bpw,
4. serving: a grouped ternary GEMM does not exist; the dense path (Prism fork,
   TAARDIS fork) has no MoE kernels,
5. the iso-compute ladder for capacity-vs-compute separation (0.6B 51.9% and
   1.7B 63.6% on WikiText; the primary ladder closed without the 4B rung, see
   [`SCALING-PROTOCOL.md`](SCALING-PROTOCOL.md)).

## 5. Reproduce

The MoE runs live in the private research tree (`hivebench/`), which is not
published; this section records the harness for provenance and audit.  The
dense-forensics scripts in this repository are self-contained.

```
hivebench/experiments/moe_ternary/
  gguf_header.py        remote GGUF header census over HTTP range requests
  role_map.py           MoE role map + ternary projection (8.82 GiB / 2.186 bpw)
  probe_router.py       E1 routing-drift prefix probe
  olmoe_proxy.py        in-place ternary QAT on OLMoE (negative control)
  moe_proxy.py          MoTE-style up-cycle proxy (Qwen3-1.7B)
  olmoe_rotate_rtn.py   rotation-vs-RTN quantizer comparison
  olmoe_doctors.py      per-layer residual-stream corrections (route A)
  olmoe_experts.py      per-expert corrections (placement control)
  eval_ckpts.py         checkpoint trajectory + router diagnostics
  save_ternary_olmoe.py materialise a ternary build to an HF dir
```

Artifacts and write-ups: `hivebench/RESEARCH/moe-ternary-plan.md` (private).
