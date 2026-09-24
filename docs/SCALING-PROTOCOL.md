# Scaling protocol — does our recipe scale like Prism's?

Status: **primary ladder in progress (2026-09-24).** The 0.6B and 1.7B
WikiText rungs are complete; the 4B rung is running. The cross-architecture
robustness extension is specified in [`REPRODUCIBILITY-AUDIT.md`](REPRODUCIBILITY-AUDIT.md).
Companion: [`RETENTION-VS-SCALE.md`](RETENTION-VS-SCALE.md) (the ladder results
and the confounds this protocol removes).

## The question

Prism's released tables show retention rising with size (1.7B ~85–88%, 8B
~92–95%, 27B 94.6–98.2%). Our tinyshakespeare ladder shows the **opposite**
(0.6B 90.6% vs 1.7B 82.3%, in both ratio and absolute ΔPPL). One of these is
wrong, or they measure different things. The question here:

> **Does our recipe's quantization degradation shrink as model size grows?**

Note what this is *not*: it does not reproduce Prism's benchmark retention, and
it does not show our recipe matches theirs. It tests the **direction of the
trend**, on our metric, on a corpus where the measurement is meaningful.

## Why the current ladder cannot answer it

Four confounds, worst first:

1. **Corpus.** Teacher PPL is 73 (0.6B) and 49 (1.7B) on tinyshakespeare — both
   far out of distribution. "Retention of a struggling FP model" may measure
   corpus quirks, not quantization robustness. (WikiText: teacher PPL ~25.)
2. **Compute regime.** Fixed steps is *iso-token*, which under-trains larger
   models. The 1.7B at 10k steps has not had the compute the 0.6B had per
   parameter.
3. **Rotation block.** The spec rule gives block 512 for the 4B (widths 2560,
   9728) but 1024 for 0.6B/1.7B/8B, so the ladder varies block as well as size.
   The 1.7B@512 control suggests this is minor once trained (1.3838 vs 1.4364 at
   step 3500), but it must be settled at convergence.
4. **Metric.** Ratio and ΔPPL can disagree when teacher PPL varies. Here they
   agree, so the reversal is **not** a metric artifact — but both must be
   reported so a future disagreement is visible rather than hidden.

## Protocol

### 1. Corpus
- **Primary:** a corpus where every rung's FP teacher is in-distribution.
  Selection rule: the *largest* rung's teacher PPL ≤ 20 and the spread across
  rungs ≤ 2×. Candidates: full WikiText-103 (~100M tokens), an OpenWebText or
  FineWeb-Edu slice.
- **Secondary:** a second, differently-distributed corpus, to confirm the trend
  is not corpus-specific.
- **Holdout:** 8 disjoint regions excluded from training via the existing
  `load_windows` path — verified against the F11 leak (train windows = total −
  regions×windows, exactly). Treat this as adaptive validation; freeze a
  separate test split before making a final claim.

### 2. Metric
Report all three and read the **trend across rungs**, never a single value:
- `retention = teacher_ppl / student_ppl` (ratio view)
- `ΔPPL = student_ppl − teacher_ppl` (absolute view)
- the FP and quantized PPLs themselves, so the gap is visible on a log axis.

If ratio and ΔPPL give **opposite** trends, that is itself the finding (and
means the metric is the story, not the model).

### 3. Compute regime
- **Primary: iso-convergence.** Train each rung to its best point under a managed
  decay schedule, then compare best points. This asks the *attainable* question.
- **Secondary: report the full trajectory**, so iso-token comparisons can be
  read off the same runs without new ones.
- (Iso-compute — `params × steps = const` — is a third option, but it gives the
  large rungs very few steps and answers a question nobody is asking.)

### 4. Rungs
0.6B, 1.7B, 4B (block 512, controlled by the 1.7B@512 run), and optionally 8B
(block 1024, needs the student/teacher 2-card split). Everything else matched:
recipe, seed, eval set, schedule.

### 5. Recipe and schedule
- rotation inside the loop + STE with per-group absmean scale + KD (temp 2.0)
- **managed decay** (warmup 10000, patience 2000, factor 0.5, drift_eps 0.10,
  cooldown 1000, max 3, floor 5e-6) — proven to prevent late divergence, so a
  rung is not scored on a diverged run
- 20k steps, checked against the trajectory: if a rung has not plateaued, extend
  it rather than reporting an under-trained point

### 6. Evaluation
Clean multi-region holdout; report the **per-region spread**, not just the mean
(the mean hides a wide range: e.g. 1.077–1.514 within one 8-region run).

### 7. Pre-registered decision rule
Fixed before any run:
- **Trend matches Prism** if `retention(0.6B) < retention(1.7B) < retention(4B)`
  on the primary corpus, with the same ordering on ΔPPL.
- **Trend does not match** otherwise.
Either outcome is reportable. The point of pre-registering is that neither can be
spun after the fact.

**Status (2026-09-24):** the first two rungs show the expected direction at
iso-convergence — 0.6B **51.9%** < 1.7B **63.6%** (WikiText, 20k + managed
decay; ΔPPL 24.5 → 12.1 agrees) — but the primary-corpus teacher-PPL threshold
is not met and the **4B** rung is still running
(`run_wiki_convergence_4B.sh`). The rule is not fully met until that run lands.
See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the benchmark-retention,
capability/access and KLD results, with their current validity caveats.

### 8. Controls
- **block:** 1.7B@512 (`ste-rotate-10k-block512`) — settles confound 3
- **corpus:** 1.7B on both corpora — settles confound 1
- **no-rotation floor:** the existing STE control (2.09× / 48%) as the lower bound

### 9. Cost (measured rates, one card)
| rung | rate | 20k steps |
|---|---|---|
| 0.6B | ~0.30 s/step | ~1.7 h |
| 1.7B | ~0.63 s/step | ~3.5 h |
| 4B | ~1.75 s/step | ~9.7 h |
| 8B | — | needs both cards (16 GB student + 16 GB teacher) |

Plus one-time corpus prep (download + tokenize). Two cards make the 0.6B/1.7B
phase parallel.

## What this does not settle

- It does not reproduce Prism's benchmark retention — different metric, and their
  recipe is unknown.
- It does not establish that our recipe *matches* theirs, only whether it scales
  in the same direction.
- It does not fix the deeper issue that our canary corpus is small: an
  in-distribution corpus fixes the *measurement*, not the data budget.

## 10. Cross-architecture extension (added 2026-09-24)

The size ladder is a Qwen3-family result, not evidence that the recipe is
architecture-agnostic. The extension is deliberately separate:

1. **Preflight (no weights):** run `scripts/pilot/inspect_model_targets.py` for
   the pinned model in `configs/model_registry.yaml`; record architecture,
   target coverage, effective rotation widths/blocks, and expected tensor count.
2. **Smoke:** run a small clean checkpoint (SmolLM2-360M or SmolLM2-1.7B) for
   500–1,000 updates to catch loader, fused-projection, and checkpoint issues.
3. **Matched convergence:** run SmolLM2-1.7B, OLMo-2-1B, Pythia-1.4B, and
   Phi-3.5-mini with the same corpus, token/update budget, fixed rotation seed,
   KD temperature, and managed schedule. Use teacher-relative metrics, not raw
   cross-tokenizer PPL.
4. **Qwen3.8 separately:** treat `Qwen/Qwen3.8-27B` as an architecture-port
   project. Its hybrid `qwen3_5` target has 401 linear tensors plus the
   separately handled embedding; it is not a drop-in 27B invocation of the
   legacy Qwen3 canary. Flash-Next and 2.4T-A95B are deferred.

The runner writes a manifest and target census, but the audit's P0 export and
persistent-basis gates remain prerequisites for a Bonsai-2 parity claim.
