# Reproducibility and cross-architecture audit

**Audit date:** 2026-09-24  
**Scope:** the complete public repository, with emphasis on the ternary QAT/KD
pilot, the Qwen3.8/Bonsai-2 target, launchers, reports, tests, and the model
inventory.  This document distinguishes a *legacy canary* from a claim of
Bonsai-2 parity.

## Executive finding

The repository has a reproducible **legacy Qwen3-1.7B canary path**, but it did
not have a self-contained, architecture-aware Bonsai-2/Qwen3.8 experiment path.
The distinction matters:

- The successful ladder uses `Qwen3ForCausalLM`, 28 all-attention layers, and
  196 decoder projections. It keeps tied embeddings/`lm_head` in FP and
  reports an in-memory ternary forward.
- `Qwen/Qwen3.8-27B` is `Qwen3_5ForConditionalGeneration` (`qwen3_5`), with a
  hybrid Gated-DeltaNet/full-attention language tower. Its Bonsai-2-aligned
  target inventory is 400 layer linears plus `lm_head` (401 forward-rotated
  linear tensors), with the embedding handled by a separate inverse-rotation
  path. It is not the same model scope as the Qwen3 ladder.
- The current local node has two AMD RX 7900 XT cards with
  `21,458,059,264` bytes (~19.98 GiB) each. A full BF16 Qwen3.8-27B teacher
  and student cannot be resident there; that model needs a text-only/sharded or
  block-wise runner and a larger-memory allocation.

The working tree now contains the first fixes for this gap: architecture
profiles, a config-only target inspector, a Qwen3.5 text-only loader, pinned
model registry entries, a run manifest, a generic cross-architecture launcher,
locked-region KLD selection, atomic pilot checkpoints, and data-prep manifests.
The remaining blockers are listed below; none should be hidden behind a
successful Qwen3 canary number.

## What the current recipe actually establishes

The canonical legacy launch is assembled in
`scripts/pilot/run_decay_replication.sh` and uses:

- `--ste --rotate`;
- per-group absmean ternary STE, g128, half-away-from-zero;
- teacher-guided KD at temperature 2.0;
- Adafactor, learning rate `5e-5`, with the currently installed Transformers
  defaults for the remaining options;
- 20,000 updates, sequence length 512, and the historical `batch=2` loop;
- managed decay: warmup 10k, patience 2k, factor 0.5, drift epsilon 0.10,
  cooldown 1k, maximum three events, floor `5e-6`;
- eight disjoint evaluation regions and best-checkpoint selection on that
  same evaluation signal.

This is evidence for the **legacy canary**. It is not evidence that the
current code is a complete 27B Bonsai-2 trainer, that packed PQ2_0 inference
matches the training forward, or that the reported holdout is an unbiased test
set.

## Findings, ordered by risk

### P0 — architecture scope and parity

**Was:** `scripts/pilot/rmd_kd.py` selected only the Qwen3 suffix vocabulary
(`q/k/v/o/gate/up/down`). That misses Qwen3.5's large
`linear_attn.in_proj_qkv`, `in_proj_z`, and `out_proj` matrices and does not
handle an untied output head. The old `configs/0.6b.yaml` also named a
non-existent `Qwen/Qwen3.8-0.6B` checkpoint.

**Now:** `bonsai_forensics/targets.py` defines explicit profiles for Qwen3,
Llama, OLMo-2, Qwen3.5, Phi-3, GPT-NeoX/Pythia, and OPT. It records selected
names, parameter coverage, rotation widths, and expected tensor counts.
`inspect_model_targets.py` performs a config/meta-only preflight. The Qwen3.8
profile selects 401 linear tensors, matching the published 401 forward set;
the embedding remains a separately documented 402nd PQ2_0 tensor.

**Still open:** the rmd path does not yet implement the persistent embedding /
hidden-norm basis or a complete 27B export. The formal `run_quant.py` path is
still governed by the frozen `tbr-1.2` role set; adding Qwen3.5's
`in_proj_qkv`/`in_proj_z` there requires an explicit spec version bump rather
than a silent suffix edit. Do not call the current 401-linear selection alone
“Bonsai-2 parity.”

### P0 — rotation semantics are not one thing

The repo contains two related but distinct ideas:

1. **Edge-local / residual-proxy QAT:** each selected `nn.Linear` absorbs a
   rotation at its own edge. `RotatedLinear` immediately applies the inverse
   rotation on output edges, so the residual stream and hidden norms are not
   persistently rewritten.
2. **Prism storage basis:** rotate each packed tensor's input/last axis,
   including output projections whose HF input width is not the residual width;
   additionally handle the inverse embedding, hidden norm folding, and the
   F16 recurrent-control set.

The repository also has a direct contract conflict: the frozen `spec.md`
classifies output-side suffixes in one convention, `HADAMARD-VERIFICATION.md`
describes Prism's embedded input/last-axis convention, and the older
`run_quant.py` role map still uses the former (including its embedding rule).
This must be resolved as one deliberate spec-versioned change across the
spec, exporter, materializer, and tests. It is not safe to patch one suffix
list in isolation.

**Acceptance:** add a mathematical FP equivalence test for the exact intended
mode, then an export/unrotation test against the packed artifact. Keep the
legacy edge-local result under a separate recipe name.

### P0 — trained ternary forward is not proven identical to PQ2_0

`recover.ternary_ste` computes scales/codes in the weight dtype (historically
BF16), while `quant.quantize_rtn_absmean` and the GGUF tools use NumPy
floating-point paths, and PQ2_0 stores FP16 scales. A diagnostic comparison
found non-zero code disagreements at BF16/NumPy boundaries. The rmd snapshot is
continuous master weights; there is no rmd-checkpoint → PQ2_0 → runtime-PPL
round trip.

**Acceptance:** define one canonical quantizer, export its codes/scales without
recomputing them in another precision, require 100% code and scale parity, and
evaluate the packed/dequantized artifact. Until then, report “in-memory ternary
forward,” not “deployed PQ2_0 parity.”

### P0 — provenance and model identity

The old rmd report recorded a small subset of flags. It did not reliably record
model revision, tokenizer identity, corpus bytes/token IDs, effective rotation
seed/signs, target inventory, package versions, GPU/ROCm details, or source
state. Several launchers used mutable `main` revisions and one sibling-repo
path.

**Now:** `bonsai_forensics/provenance.py` writes `run-manifest.json` and embeds
it in checkpoints; rmd accepts `--model-revision`, records the resolved target
coverage and PRF basis digest, and hashes the corpus/token IDs. The pool and
text preparation helpers now accept dataset/tokenizer revisions and emit
sidecar manifests. `configs/model_registry.yaml` pins the models used by the
new plan.

**Still open for historical/T28 reports:** token-ID hashes, output checkpoint
hashes, failure records, and a strict clean-tree/diff policy for final claims.
New RMD runs record token-ID hashes and report-sidecar fingerprints; the older
recover path and historical artifacts do not.

### P1 — rotation identity is not independently seeded

The old code used `args.rot_seed or args.seed`, making zero ambiguous and
silently changing the basis when only the data seed changed. Checkpoints did
not contain materialized rotations or a sign digest.

**Now:** `--rot-seed` is independent; unset follows the training seed, while an
explicit value (including zero) is preserved. The report records the effective
seed, width set, block policy, and PRF digest via `rotation.basis_digest`.

**Still open:** support a strict explicit sign manifest for the published
Prism basis, make checkpoint loaders reject basis/profile mismatches, and test
that changing the training seed leaves the sign digest unchanged.

### P1 — `batch=2` is not a real batch

The historical loop selects up to two windows from a permutation and performs
two sequential optimizer updates. It does not stack examples, average their
losses, or accumulate gradients. The actual sampled-window count is recorded
now as `sequential_windows_per_reshuffle`; the argument must not be used for
token/cost calculations as if it were a tensor batch.

**Decision required:** preserve this as a named legacy behavior, or implement a
real batch/gradient-accumulation mode and rerun a new baseline. Do not silently
change the historical ladder.

### P1 — adaptive validation is being called held-out test

The evaluation regions are excluded from training, which prevents direct token
leakage, but the same metric drives decay and best-checkpoint selection. The
reported “best retention” is therefore adaptive validation performance.

**Now:** reports label the selection metric accordingly. A final claim needs
three disjoint splits: train, validation, and a test set touched only after
the recipe is frozen.

### P1 — resume and controller state are incomplete

Snapshots contain model state and raw arguments, but not optimizer state,
Adafactor step counters, scheduler state, best metric, RNG state, data cursor,
or the complete resolved config. There is no tested rmd `--resume`.

**Acceptance:** a 20+30 resume must match an uninterrupted 50-step run for
weights, optimizer, LR events, sampled-window hash, and metrics.

### P1 — resource planning is not enforced

The 27B loader can expose a text-only view, but the current rmd path still
expects to place a full teacher and student on explicit devices. The local
2×~20 GiB node cannot host the full BF16 Qwen3.8-27B pair. Dense rotation
matrices also need a memory estimate.

**Acceptance:** a resource preflight must reject impossible device plans before
weight download/load, and the 27B path must use a sharded or block-wise recipe.

### P1 — launchers can report false success

Several historical scripts use `set -u` and background jobs without collecting
all child statuses. A failed Python arm can be followed by an “ALL DONE” echo
and a zero shell status. Output directories can also be reused.

**Now:** the new cross-architecture launcher and the queued secondary launcher
collect child statuses, refuse existing outputs, and preflight before training.
The older launchers should be migrated or explicitly marked legacy.

### P1 — numerical defaults are mutable

The Adafactor call leaves most options to the installed Transformers version;
dtype, attention implementation, cache policy, deterministic mode, and ROCm
kernel/workspace settings are not all recorded. The code/docs also disagree
about FP32 versus BF16 embeddings/head/teacher precision.

**Acceptance:** freeze a recipe profile with every optimizer/numerical option,
record the resolved environment, and declare whether repeatability is bitwise
or tolerance-based.

## Cross-architecture pilot plan

The machine-readable inventory is `configs/model_registry.yaml`.

| order | model | architecture | why it matters | local status |
|---|---|---|---|---|
| 1 | `HuggingFaceTB/SmolLM2-1.7B` | Llama | clean, matched-size, immediately runnable | yes |
| 2 | `allenai/OLMo-2-0425-1B-Instruct` | OLMo-2 | different architecture, untied head | yes |
| 3 | `EleutherAI/pythia-1.4b` | GPT-NeoX | unusual fused/named projections | yes |
| 4 | `microsoft/Phi-3-mini-4k-instruct` | Phi-3 | fused `qkv_proj` and `gate_up_proj`, no LongRoPE | memory smoke first |
| 5 | `microsoft/Phi-3.5-mini-instruct` | Phi-3 | same fused geometry plus 131K context config | after 4K smoke |
| control | `Qwen/Qwen3-1.7B` | Qwen3 | historical regression | yes |
| later | `Qwen/Qwen3.8-27B` | Qwen3.5 hybrid | Bonsai-2 target and architecture port | not full-resident locally |
| defer | Qwen3.8 Flash-Next / 2.4T-A95B | Qwen4Exp / Qwen3.5 MoE | new routing/architecture and enormous memory | no |

### Fairness rules

- Use the same corpus bytes, sequence length, token/update budget, seed,
  rotation policy, KD temperature, and managed schedule.
- Compare teacher-relative PPL ratio/retention, ΔPPL, KL, and common benchmark
  accuracy; do not compare raw PPL across tokenizers/models as if it were
  commensurate.
- Run a config-only target inventory before downloading weights.
- Report selected parameter fraction, effective rotation blocks, peak VRAM,
  sampled-window hash, and packed/export status.
- Treat tied embeddings, untied heads, and exempt recurrent controls as
  explicit scope decisions, not accidental omissions.
- Start with a 500–1,000-step smoke, then 20k convergence; only the latter is
  evidence for architecture robustness.

### Qwen3.8-specific decision

Do **not** queue a naive `rmd_kd.py --model-dir Qwen/Qwen3.8-27B` run on the
current cards. The model is a multimodal Qwen3.5 hybrid, not a Qwen3 model.
First choose and test either:

- a persistent absorbed-basis implementation with embedding/head/norm/export
  parity; or
- a clearly labelled edge-local QAT proxy with a mathematically verified
  conversion to the intended packed basis.

The full BF16 checkpoint is a valid later target on a larger allocation; the
FP8 checkpoint is not a clean substitute for the FP teacher, and Flash-Next /
2.4T are separate architecture projects.

## Evidence reconciliation for this checkout

A repository-wide audit found several claims that are historical or not
locally verifiable rather than current evidence:

- The old `kld-wikiconv-1.7B` report selected prefix chunks from the training
  corpus. Its `0.7064` KLD is a contaminated-prefix diagnostic, not a held-out
  gate result. `kld_eval.py` now defaults to the checkpoint's locked
  `eval-regions.json`; rerun before using a KLD number.
- The visible 0.6B/1.7B WikiText reports are useful legacy trajectories, but
  they predate manifests and completion markers. The 4B directory currently has
  no final report; the scaling rule is incomplete.
- `artifacts/gate1`, `gate2`, `gate3`, `eval`, `reference`, `base27`, and the
  private historical commit evidence are not all present in this checkout.
  Their documented numbers must be labelled historical/unavailable, not treated
  as independently rerun facts.
- ARC/HellaSwag/PIQA is a screening proxy, not Prism's benchmark suite. The
  current `mc_bench.py` report explicitly records `comparable_to_prism: false`.
- The older type-42 `oracle.decode_q2_0_g64` path is recorded as an open
  layout defect in `FAILURES.md`; it is separate from the verified type-142
  PQ2_0 path. The 27B materializer/pack_gguf paths also accumulate shard
  payloads in memory despite being described as streaming, so their resource
  claims need correction before a large run.
- The shipped `configs/calib_a/b/c.yaml` files intentionally contain empty
  `corpus_paths`; a run labelled A/B/C is not evidence of calibration until
  real paths and hashes are supplied.
- Several older shell launchers still have weak status propagation and private
  absolute paths. The new cross-architecture and queued secondary launchers are
  hardened; historical launchers should be treated as legacy until migrated.

These distinctions are part of the result, not cosmetic documentation: they
determine which numbers can be compared, rerun, or cited.

## Acceptance checklist for a new repeatable runner

A run is not accepted as a paper-grade result until it has:

1. immutable model/tokenizer/data revisions and hashes;
2. a canonical, hashed recipe profile;
3. strict architecture/profile/target-count validation;
4. a recorded rotation basis digest and independent training seed;
5. explicit train/validation/test manifests;
6. a canonical quantizer with code/scale/export parity;
7. a packed artifact or an explicitly labelled in-memory-proxy result;
8. complete controller state and tested resume;
9. atomic completion/failure markers and nonzero failure propagation;
10. a resource preflight and environment manifest;
11. a committed source state (or a recorded source diff hash);
12. a multi-seed result and a labelled legacy-canary regression.

## Useful preflight commands

```sh
# No weights, no GPU: inspect a Hub config and target inventory.
python scripts/pilot/inspect_model_targets.py \
  --model HuggingFaceTB/SmolLM2-1.7B \
  --revision effd688a12921b4cc83e3312b6feb579f70f9c71

# Qwen3.8-27B: confirms the 401-linear hybrid inventory.
python scripts/pilot/inspect_model_targets.py \
  --model Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

# Run a new architecture pilot only after the preflight report is reviewed.
MODEL=HuggingFaceTB/SmolLM2-1.7B \
MODEL_REVISION=effd688a12921b4cc83e3312b6feb579f70f9c71 \
TAG=smollm2-1.7b-input VIS=0 \
  ./scripts/pilot/run_cross_arch_pilot.sh
```

The pilot launcher is intentionally not automatic: the current convergence
jobs own the cards, and model downloads/allocations should be an explicit
decision.
