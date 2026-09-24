# TBR Failure Register

> **Companion:** [`docs/WHITEPAPER.md`](WHITEPAPER.md) — the source of truth.
> This register is the grounding document: every failed or falsified route,
> recorded so it can be re-assessed objectively later.

**Created:** 2026-09-20 (initial seeding from Rounds 1–13).
**Policy:** append-only. New entries get a new id and a timestamp; existing
entries are never rewritten — corrections are appended as `[rev <date>]` notes.

**Two series.** `F1`–`F11` are the load-bearing falsifications referenced by the
white paper ([`docs/WHITEPAPER.md`](WHITEPAPER.md)). `U1`–`U3` are **unrelated**
items — operational incidents and a latent bug that do not bear on the findings
— recorded here for completeness and re-assessability only.

## Categories

- **(a) falsified hypotheses** — scientific claims that failed a controlled test.
- **(b) implementation dead-ends** — engineering routes that could not work as built.
- **(c) resolved incidents** — post-mortems of crashes/blocks, with the mitigation.
- **(d) open bugs/risks** — known defects or hazards still live.

## Entry schema

`ID · claim/hypothesis · method · outcome · verdict · evidence (commit/artifact/report) · status (open/closed/resolved) · what it rules out · cost to revisit`

## Summary

| ID | Category | One line | Verdict | Status | Worth re-assessing? |
|---|---|---|---|---|---|
| [F1](#f1--public-ptq-collapses-off-calibration) | (a) | Public PTQ collapses off-calibration; published 2.1× is a calibration-passage artifact | Falsified | closed | No |
| [F2](#f2--gptqhessian-variants-move-codes-away-from-prisms) | (a) | GPTQ/Hessian variants move codes *away* from Prism's; RTN 0.9145 vs 0.8086–0.8497 | Falsified | closed | No |
| [F3](#f3--stochastic-calibration-resonance-falsified) | (a) | Stochastic calibration resonance (5% noise) has no effect: +0.003 pp | Falsified | closed | No |
| [F4](#f4--no-rental-rotateabsmean-rtn-parity-is-dead) | (a) | rotate+absmean RTN of the base = 23,606 PPL vs 18.5851 (1,270×) | Falsified | closed | No |
| [F5](#f5--the-last-8-was-localized-to-end-to-end-qat-and-deliberately-not-funded) | (a)/(b) | Last ~8% localized to end-to-end QAT; 27B proof-run priced and deliberately not funded | Deferred by decision | closed (deferred) | Only if capability ownership is required |
| [F6](#f6--student-stream-block-wise-kd-dead-end) | (b) | Student-stream block-wise KD = 1.25 M PPL | Dead-end | closed | Only inside end-to-end QAT |
| [F7](#f7--teacher-forced-block-wise-kd-dead-end) | (b) | Teacher-forced block-wise KD = 1.06 M PPL; local per-layer KD compounds | Dead-end | closed | No |
| [F8](#f8--entropyexcess-loss-data-selection-lost-to-random) | (a)/(b) | Entropy/excess-loss selection lost to random: 172.3 vs 115.9 PPL | Falsified (pilot) | closed | Yes — larger pool/seeds |
| [F9](#f9--27b-full-model-cpu-offload-swap-thrashes) | (b)/(c) | 27B full-model offload swap-thrashes (55.6 GB vs 40 GB VRAM + ~20 GB RAM) | Mitigated | resolved | No |
| [F10](#f10---ngl-99-allocation-failure-on-one-20-gb-card) | (c) | `-ngl 99` on one 20 GB card fails to allocate; fixed by auto-fit | Resolved | resolved | No |
| [F11](#f11--t28-holdout-leak-evaluation-on-training-windows) | (a) | T28's 1.103× was measured on training windows; clean multi-region holdout is ~2.09× (~48%, range 1.55-2.71×) | Retracted | open | Yes — re-baseline |
| [U1](#u1--internal-tooling-drift-blocked-harness-imports) | (c)/(d) | Internal tooling drift blocked harness imports; pinned dependency workaround is interim | Mitigated | resolved (fix pending) | Yes — T23 |
| [U2](#u2--two-rocm-contexts-hang-gpu1-at-firmware-level) | (c)/(d) | Two ROCm contexts hang GPU1 at firmware level | Mitigated | resolved (policy) | If driver changes |
| [U3](#u3--oracledecode_q2_0_g64-decodes-garbage) | (d) | `oracle.decode_q2_0_g64` (type 42) decodes garbage | Open | open | Yes — cheap fix |

`U1`–`U3` are unrelated to the white-paper findings (operational incidents and
a latent bug); they are recorded for completeness only.

---

## F1 — Public PTQ collapses off-calibration

- **Category:** (a) falsified hypothesis (scientific).
- **Claim/hypothesis:** A public PTQ stack (GPTQ + salient + QuIP rotation) can
  reach Bonsai-2-class quality at ~2 bpw, and the published **2.1×** is the
  attainable off-calibration bar.
- **Method:** Ported ThakiCloud's `quip.py` verbatim into
  [`bonsai_forensics/reference.py`](../bonsai_forensics/reference.py) (dense QR rotation, their upper-Cholesky
  `Hinv`, per-row absmean, block 128); binary and ternary arms on
  `Qwen/Qwen3-1.7B`; fixed-passage metric vs held-out windows with real
  `32×2048` calibration Hessians.
- **Outcome:** Fixed-passage binary `rotate=in, damp=0.3` reproduces the
  published number exactly: **2.027 → 4.268 = 2.106×**. With real calibration
  and *held-out* evaluation the same configurations collapse: binary **1178×**,
  ternary `in/d0.1` **9184×**, `both/d0.1` **34877×** (fp32 held-out PPL 39.7).
  Our own T10 canary was **3514×** held-out. The public stack is the same order
  of broken (1.2e3–3.5e4×).
- **Verdict:** Falsified. The 2.1× exists only when calibration and evaluation
  are the same passage.
- **Evidence:** commit `ea089e2`; `artifacts/reference/*.json`;
  [`docs/RUN-CANARY.md`](RUN-CANARY.md); project records Round 6.
- **Status:** closed.
- **What it rules out:** PTQ tuning as a route to 27B parity; the published 2.1×
  as an off-calibration quality bar.
- **Cost to revisit:** low ($0, ~100 s per config) — but only meaningful if a
  genuinely new algorithm appears.
- **Worth re-assessing?** No.

## F2 — GPTQ/Hessian variants move codes away from Prism's

- **Category:** (a) falsified hypothesis (scientific).
- **Claim/hypothesis:** Prism's 8% trit residual (vs rotate+absmean RTN) is an
  error-compensation trick recoverable with GPTQ/OBQ in the correct basis.
- **Method:** Loaded embedding + layers 0–3 of the 27B only (55 tensors, no
  55 GB residency); captured real Shakespeare Hessians; layout-adjusted and
  rotated each weight per T30; swept GPTQ damping, act-order, `RᵀHR` vs raw
  `H`, and group scale search.
- **Outcome:** Agreement with Prism's trits — **RTN absmean 0.9145**;
  GPTQ `d0.01` **0.8086**, `d0.1` **0.8395**, `d0.3` **0.8497**; act-order
  **0.8065**; `RᵀHR` **0.8086**; raw `H` **0.7991**; group scale-search
  diagonal **0.8625**, block **0.8506**. Monotone in damping: suppressing
  compensation converges back to RTN. Every error-compensated assignment moves
  codes *away* from theirs.
- **Verdict:** Falsified. The quantizer reverse-engineering track is closed.
- **Evidence:** `artifacts/gate3/prefix27/sweep-layer0.json`;
  [`research/gate3-qat-verdict.md`](../research/gate3-qat-verdict.md); project records Round 12. (T32 was diagnostics-only;
  no source commit.)
- **Status:** closed.
- **What it rules out:** GPTQ/OBQ as the missing mechanism; a GPTQ rental for
  parity.
- **Cost to revisit:** low locally (prefix trick); no rental justified.
- **Worth re-assessing?** No.

## F3 — Stochastic Calibration Resonance falsified

- **Category:** (a) falsified hypothesis (scientific).
- **Claim/hypothesis:** Calibration noise/dithering (stochastic rounding) can
  explain Prism's assignment.
- **Method:** 1.7B GPTQ g128, damp 0.1, clean vs 5%-random-token-noise Hessians,
  compared against Prism's released 1.7B `PQ2_0` trits (byte-exact decode).
- **Outcome:** RTN 0.61196; GPTQ clean 0.61771; GPTQ noise5 **0.61774**;
  noise moved 4.28% of codes; agreement change **+0.003 pp**. No effect.
  Side finding: the `Q2_0_g64` (type 42) decoder in [`oracle.py`](../bonsai_forensics/oracle.py) decodes garbage
  (see [U3](#u3--oracledecode_q2_0_g64-decodes-garbage)); T7's 0.612 came from the F16 dequant, not this decoder.
- **Verdict:** Falsified.
- **Evidence:** `artifacts/gate3/scr-report.json`;
  [`research/gate3-qat-verdict.md`](../research/gate3-qat-verdict.md); project records Round 12.
- **Status:** closed.
- **What it rules out:** calibration noise, dithering and stochastic rounding as
  the source of the residual.
- **Cost to revisit:** low ($0).
- **Worth re-assessing?** No.

## F4 — No-rental rotate+absmean RTN parity is dead

- **Category:** (a) falsified hypothesis (scientific).
- **Claim/hypothesis:** The no-rental shortcut — rotate the public base into
  Prism's basis and pack absmean RTN — reproduces Prism's quality (their 1.44×).
- **Method:** Copied Prism's GGUF and rewrote **all 402** `PQ2_0` payloads in
  place with rotate+absmean RTN of the pinned public base (converter layouts per
  T30; embeddings input-rotated). Metadata, tokenizer, F32/BF16 exemptions and
  the tensor table stayed byte-identical; repack byte-exact; 402/402 post-patch
  SHA-256 verified. Eval: fork `llama-perplexity`, `-ngl 99 -c 512 --chunks 8`,
  tinyshakespeare, ROCm1, one process.
- **Outcome:** Prism `PQ2_0` **18.5851 ± 1.209**; ours (402 RTN payloads, same
  file) **23,606.19 ± 1231.6** = **1,270×**. Smoke with only layers 0+3 patched
  (2 chunks): 53.89 vs 20.41 = 2.6× from 13/402 tensors — the GDN recurrence
  amplifies small weight perturbations.
- **Verdict:** Falsified. 92% trit agreement is not a quality proxy; the missing
  8% carries the entire gap.
- **Evidence:** commit `5eac0c0` (T31); `artifacts/gate2/eval-report.json`;
  `artifacts/gate2/rtn-absmean.gguf`; [`research/gate2-rtn-quality.md`](../research/gate2-rtn-quality.md);
  project records Round 11.
- **Status:** closed.
- **What it rules out:** the no-rental rotate+RTN shortcut for parity; trit
  agreement as a standalone quality metric.
- **Cost to revisit:** low locally, but the direction is settled.
- **Worth re-assessing?** No.

## F5 — The last ~8% was localized to end-to-end QAT and deliberately not funded

- **Category:** (a)/(b) mixed — partial scientific result, **closed by scope
  decision** (not a methodological dead-end).
- **Claim/hypothesis:** Local QAT/KD (STE ternary training + KD) can close the
  quality gap and reach Prism's acceptance rate (≥97% retention, stretch 98.2%
  = Bonsai-2 parity) without a rental.
- **Method:** T28 [`bonsai_forensics/recover.py`](../bonsai_forensics/recover.py) — 196 attention/MLP linears
  wrapped as exact ternary (g128, half-away) with straight-through gradients;
  bf16 master weights; KD vs the frozen fp32 teacher; Adafactor + gradient
  checkpointing; 301k-token tinyshakespeare corpus. It ran *after* the
  elimination programme (Gates 1–3; F1–F4, F6, F7) had localized the gap.
- **Outcome:** Held-out PPL — run1 (1500 steps) 1.673×; run2 (5000 steps)
  student 42.988 vs teacher 38.966 = **1.103×** (90.6% retention); run3 (7000
  steps) 1.129× with held-out worsening while train loss fell (overfit).
  1.103× clears the 1.44× Prism bar but is **short of the ≥97% mission
  retention target** (stretch 98.2%). Corpus-limited at ~5k steps.
- **Why this is the last 8% (elimination chain):** Gate 2 ([F4](#f4--no-rental-rotateabsmean-rtn-parity-is-dead)) showed the ~8%
  trit residual carries the entire quality gap; Gate 3 ([F2](#f2--gptqhessian-variants-move-codes-away-from-prisms), [F3](#f3--stochastic-calibration-resonance-falsified)) showed that
  residual is *weight movement during training*, not any quantizer/calibration
  trick; and local per-layer KD cannot control global compounding ([F6](#f6--student-stream-block-wise-kd-dead-end), [F7](#f7--teacher-forced-block-wise-kd-dead-end)).
  With every non-training route eliminated, end-to-end QAT/KD is the only
  remaining route to Prism's exact acceptance. The gap is therefore *localized*,
  not mysterious.
- **Verdict:** **Deliberate stop, not a dead-end.** The fix is known and priced;
  the programme chose not to fund a 27B end-to-end proof-run because Track B
  already delivers the capability from Prism's public weights, and the
  proof-run has real cost and uncertain yield (the 1.7B QAT run was itself
  short of target).
- **Decision (2026-09-20, programme owner):** do not run the 27B end-to-end QAT
  proof. Rationale: (i) the mission non-goal is to reproduce the recipe — the
  target is the result; (ii) that result is public; (iii) the remaining
  confidence is a localization/inference, not a demonstrated replication.
- **Evidence:** commit `64ff7bc` (T28);
  `artifacts/recover/run1/recover-report-s5000.json` and
  `recover-report.json`; bar 1.44× from commit `8cd8038` (T7) / project records
  Round 7; project records Round 9.
- **Status:** closed by decision (deferred, not eliminated).
- **What it rules out:** local QAT/KD at 1.7B on a 301k-token corpus as a
  *complete* quality path. It does **not** rule out 27B end-to-end QAT.
- **Cost to revisit:** medium — a real 27B KD corpus plus either a 1×48–80 GB
  rental for full-master QAT or a substantially larger local LoRA/block-wise
  pilot, plus iteration.
- **Worth re-assessing?** Only if *capability ownership* (rather than using
  Prism's public weights) becomes a requirement.
- **[rev 2026-09-20] Seed discrepancy:** the handoff seeded F5 as
  "quality gate failed". project records Round 9 and the T28 task row record the gate
  as **PASSED** (1.103× < 1.44× bar). The defensible failure is at the *mission*
  level (90.6% < 97%), not the T28 gate. This entry records the verified
  reading; see "Seeded-entry verification notes" below.
- **[rev 2026-09-20b] Reframed per programme-owner clarification:** F5 was
  treated as a failure against the *mission acceptance rate* — we expected to
  match Prism's retention and set out to find the last 8% — not against the T28
  gate. The 8% trit residual (Gates 1–3) is where that quality lies, and
  end-to-end QAT is the only route to it. The route was priced and consciously
  skipped, because Prism's result is public and Track B already ships it. This
  is a budget/scope decision, not an inability to proceed.
- **[rev 2026-09-22] Held-out number retracted.** The "Outcome" figure
  (student 42.988 vs teacher 38.966 = 1.103×) was measured on windows that were
  part of the training pool: `build_batches` had no holdout and
  `evaluate_perplexity` read `[samples, samples + eval_windows)` windows. A
  clean multi-region holdout gives ~2.09× mean, ~48% retention (8 regions,
  5000 steps). See
  [F11](#f11--t28-holdout-leak-evaluation-on-training-windows). The mission-level
  shortfall is therefore larger than 90.6% implies; the scope decision above
  (do not fund the 27B proof-run) is unchanged.
- **[rev 2026-09-24] Precision wording:** the historical entry says “fp32
  teacher,” but the current model loader uses BF16. Future manifests must record
  the actual teacher/master/logit dtypes rather than inherit that label.

## F6 — Student-stream block-wise KD dead-end

- **Category:** (b) implementation dead-end (engineering).
- **Claim/hypothesis:** Training each transformer block against the frozen
  block's own output (student stream) can replace end-to-end QAT.
- **Method:** `artifacts/pilot/scripts/blockwise_kd.py`; 64 layers
  sequentially, 16 windows × 512 tokens, 300 STE steps/layer; then packed and
  evaluated under the Gate-2 protocol.
- **Outcome:** Per-layer relative error improved (e.g. layer 0 0.4225 → 0.1084),
  but the assembled model evaluated at **PPL 1,252,835 (1.25 M)** — worse than
  plain RTN. Error compounds across blocks.
- **Verdict:** Dead-end as a standalone route.
- **Evidence:** `artifacts/pilot/blockwise/blockwise-report.json`,
  `artifacts/pilot/blockwise/eval-bw.log`;
  `artifacts/CLEANUP.md`; project records Round 13.
- **Status:** closed.
- **What it rules out:** per-layer student-stream training as a substitute for
  end-to-end training.
- **Cost to revisit:** low locally, but only as a component of end-to-end QAT.
- **Worth re-assessing?** Only inside an end-to-end QAT loop.

## F7 — Teacher-forced block-wise KD dead-end

- **Category:** (b) implementation dead-end (engineering).
- **Claim/hypothesis:** Using the fp32 teacher's per-layer outputs as targets
  (teacher forcing) controls compounding.
- **Method:** Same sequential block trainer with teacher-forced targets
  (`artifacts/pilot/scripts/blockwise_kd.py`, `_tf` variant); 64 layers,
  16×512 windows, 300 steps/layer.
- **Outcome:** Assembled PPL **1,055,018 (1.06 M)** — better than the
  student-stream variant (F6) but still far worse than RTN. Local per-layer KD
  cannot control global compounding.
- **Verdict:** Dead-end. Only end-to-end QAT can.
- **Evidence:** `artifacts/pilot/blockwise_tf/blockwise-tf-report.json`,
  `artifacts/pilot/blockwise_tf/eval-bw.log`; project records Round 13.
- **Status:** closed.
- **What it rules out:** per-layer teacher-forced KD as a local substitute for
  end-to-end QAT.
- **Cost to revisit:** low locally; not recommended.
- **Worth re-assessing?** No.

## F8 — Entropy/excess-loss data selection lost to random

- **Category:** (a)/(b) mixed.
- **Claim/hypothesis:** Selecting KD windows by model surprisal / learning
  progress (top-20% by teacher CE and RTN-student CE) beats random sampling.
- **Method:** Wikitext-103 pool scored by teacher CE and RTN-student CE
  (`artifacts/pilot/`); top-20% (`run-selected`) vs random
  (`run-random`) KD arms, identical budget; held-out eval.
- **Outcome:** Selected **student PPL 172.28** vs random **115.91** (teacher
  22.83). The selected arm lost.
- **Verdict:** Falsified for this pilot. (Score report: selected excess mean
  14.22 vs random 12.50 — the ranking did not isolate better training windows.)
- **Evidence:** `artifacts/pilot/eval-selected.json`,
  `artifacts/pilot/eval-random.json`,
  `artifacts/pilot/score-report.json`; project records Round 13.
- **Status:** closed (pilot scale).
- **What it rules out:** this scoring function, at this pool/budget, as a
  data-selection win.
- **Cost to revisit:** low locally — a larger pool, multiple seeds, or a
  different selection metric could change the result.
- **Worth re-assessing?** Yes, if a real 27B KD corpus is built.

## F9 — 27B full-model CPU offload swap-thrashes

- **Category:** (b)/(c) implementation dead-end / incident.
- **Claim/hypothesis:** The 55.6 GB 27B base can be loaded with CPU offload for
  full-model Hessian capture.
- **Method:** Attempted full-model load/offload for 27B Hessians on a 2×20 GB
  card / 30 GiB host.
- **Outcome:** Swap-thrash; the job was killed and replaced by the **prefix
  trick** (load embedding + first N layers only), which avoids 55 GB residency
  and made the Gate-3 Hessians possible.
- **Verdict:** Full-model offload is not viable on this box.
- **Evidence:** project records Rounds 11–12; `project handoff` §6;
  `artifacts/base27/` (52 GiB on disk); `artifacts/CLEANUP.md`.
- **Status:** resolved (workaround adopted).
- **What it rules out:** full 27B residency/offload for local diagnostics.
- **Cost to revisit:** n/a unless a 1×48–80 GB machine is rented.
- **Worth re-assessing?** No.

## F10 — `-ngl 99` allocation failure on one 20 GB card

- **Category:** (c) resolved incident (post-mortem).
- **Claim/hypothesis:** The 27B `PQ2_0` (7.2 GB) fits one 20 GB card at
  `-ngl 99` with a large context.
- **Method:** Direct `llama-server` launch with `-ngl 99` and default context.
- **Outcome:** `alloc_tensor_range: failed to allocate ROCm0 buffer of size
  27233914880` → `llama_model_load: error loading model: unable to allocate
  ROCm0 buffer`. The requested buffer (~25.4 GB) exceeds the 20 GB card.
- **Verdict:** Resolved by unsloth-style auto-fit `ngl` (`fit_gpu_layers`,
  ~10% VRAM headroom minus the f16 KV cache) plus an OOM pre-flight guard.
- **Evidence:** `logs/llama_server_8090.log:2912–2913`; commit `a52aace`
  (`harness/models.py`, `fit_gpu_layers` at line 379); commit `abe76d3`
  (OOM guard); project records hardening series.
- **Status:** resolved.
- **What it rules out:** blind `-ngl 99` on a single card; naive context sizing.
- **Cost to revisit:** none.
- **Worth re-assessing?** No.

## F11 — T28 holdout leak (evaluation on training windows)

- **Category:** (a) correction — a previously reported result fails on
  re-measurement.
- **Claim/hypothesis (as published):** The T28 1.7B STE+KD run reached a
  held-out **1.103× / 90.6% retention** ([F5](#f5--the-last-8-was-localized-to-end-to-end-qat-and-deliberately-not-funded)).
- **Method:** Re-inspection of [`bonsai_forensics/recover.py`](../bonsai_forensics/recover.py)
  plus clean-holdout re-runs. `build_batches` sampled windows from
  `ids[:usable]` with **no exclusion**, while `evaluate_perplexity` read a fixed
  token region, and with the T28 config that region was inside the sampling
  pool. Over 5000 steps at batch 2, each eval window was sampled ~17 times.
- **Outcome:** The published 1.103× is a **train-set (memorized) figure**.
  The first fix was also wrong: the eval call passed a hardcoded `seq_len=2048`
  while the holdout assumed `seq_len=512`, so the excluded region
  (`[16384, 18432)`) was not the evaluated region (`[65536, 73728)`). That run's
  **1.219× / 82.0% is retracted** as well.
  The clean measurement is the `rmd_kd.py` multi-region run
  (`artifacts/rmd/ste-regions-8x8`): 8 disjoint regions spread across the
  corpus, all excluded from training, 8 windows each (32k held-out tokens), 5000
  steps. Mean ratio **2.0855×** (min 1.546, max 2.711, std 0.34), i.e. ~48%
  retention, range ~37-65%. It does **not** clear the 1.44× gate on this
  evaluation. A corrected T28-harness re-run with a single source of truth for
  the holdout (`eval_holdout_windows`) is in flight
  (`artifacts/recover/holdout-v2`).
  A region diagnostic (`/tmp/opencode/diag_regions.py`) shows absmean-STE
  quantization of the base is catastrophic at init (~1.4e5× on the regions
  tested), confirming the harness is not miscalculating.
- **Verdict:** **Reported result retracted twice.** Neither 90.6% nor 82.0%
  survives; the clean multi-region number is ~48% retention (2.09× mean). The
  scope decision in F5 is unchanged.
- **Evidence:** private record
  `hivebench/artifacts/ternary/recover/run1/recover-report-s5000.json`
  (config `samples 32, seq_len 512, batch 2, steps 5000`; student 42.988 vs
  teacher 38.966); contaminated first fix
  `artifacts/recover/holdout-baseline/recover-report.json` (retracted, leak 2);
  clean multi-region `artifacts/rmd/ste-regions-8x8/eval-regions.json`;
  corrected re-run pending at `artifacts/recover/holdout-v2`.
- **[rev 2026-09-22b]** The initial F11 correction removed the wrong window
  range (see Outcome). Kept as appended history rather than a rewrite: 1.103×
  was leak 1, 1.219× was leak 2, ~2.09× is the clean number.
- **Status:** open (correction propagated to
  [`docs/WHITEPAPER.md`](WHITEPAPER.md) sections 3, 4, 5 and 7; F5 rev note).
- **[rev 2026-09-24]** The local `artifacts/recover/holdout-v2` report is present
  and records a completed 5,000-step clean-holdout run, so the older “pending”
  wording above is stale. It is still a validation-selected legacy result, not
  an untouched final test-set claim; the next baseline must preserve that
  distinction.
- **What it rules out:** citing 1.103× / 90.6% as a held-out result, and the
  inference that the 1.7B pilot was close to the mission bar. It does **not**
  rule out STE+KD training as a route.
- **Cost to revisit:** low — the harness holdout fix
  (`build_batches(..., holdout_windows=...)`) plus a re-run are cheap. An exact
  re-evaluation of the original T28 checkpoint is impossible: no checkpoint was
  saved.
- **Worth re-assessing?** Yes — re-baseline 1.7B retention on a clean holdout
  before judging any further variant.

## U1 — Internal tooling drift blocked harness imports

- **Category:** (c)/(d) resolved incident + open durable fix — **unrelated to the white-paper findings**.
- **Claim/hypothesis:** The evaluation harness could import its tooling
  dependency against the dependency's current state.
- **Method:** Repo-wide collection against the dependency checkout.
- **Outcome:** Collection failed: a configuration class
  no longer existed after the dependency's
  internal reorganisation. **Resolved operationally** by pinning the harness
  to a detached worktree of the dependency at a pre-reorganisation tip
  (unit 561 passed, integration 50 passed + 3 skipped).
- **Verdict:** Mitigated; the durable fix is task **T23** (still open).
- **Evidence:** project records, Round 1 finding F6, Round 2, T23 row;
  project handoff §6; [`docs/BONSAI-RUNTIME.md`](BONSAI-RUNTIME.md).
- **Status:** resolved operationally; real fix pending.
- **What it rules out:** importing the dependency at its latest state without
  the pin.
- **Cost to revisit:** low–medium (T23 shim in the harness).
- **Worth re-assessing?** Yes — T23.

## U2 — Two ROCm contexts hang GPU1 at firmware level

- **Category:** (c)/(d) resolved incident + standing risk — **unrelated to the white-paper findings**.
- **Claim/hypothesis:** Two independent ROCm processes can run on the two cards
  concurrently.
- **Method:** Ran a concurrent scorer + block-trainer (then a batch-4 scoring
  relaunch) with independent contexts.
- **Outcome:** The system went down twice; forensics show GPU1 (`07:00.0`) SMU
  metrics failing, then PSP/TOC firmware init failure (`error -22`) on the next
  boot, recovered only after a second reboot. Not host OOM.
- **Verdict:** Resolved by policy: **one heavy ROCm process at a time**; two
  cards are used only via a single model-parallel process. All GPU work pinned
  to GPU1 (`HIP_VISIBLE_DEVICES=1`; HIP index = PCI order).
- **Evidence:** project records Round 13 (crash forensics, second crash);
  `project handoff` §6.
- **Status:** resolved (policy), root cause external to this repo.
- **What it rules out:** independent concurrent ROCm contexts on this driver
  stack.
- **Cost to revisit:** none unless the driver/firmware stack changes.
- **Worth re-assessing?** Only on a driver/ROCm upgrade.

## U3 — `oracle.decode_q2_0_g64` decodes garbage

- **Category:** (d) open bug/risk — **unrelated to the white-paper findings**.
- **Claim/hypothesis:** The type-42 `Q2_0_g64` decoder in [`oracle.py`](../bonsai_forensics/oracle.py) reads
  mainline 18-byte g64 blocks correctly.
- **Method:** Cross-checked decoded 1.7B `Q2_0_g64` tensors against the F16
  dequant / RTN agreement.
- **Outcome:** The decoder returns garbage. T7's reported 1.7B mean agreement
  of **0.612** came from the F16-dequant path, not from this decoder. The
  `PQ2_0` (type 142) decoder *is* byte-exact against Prism's own F16 dequant.
- **Verdict:** Open latent bug. Not on the 27B path (the 27B uses `PQ2_0`).
- **Evidence:** [`bonsai_forensics/oracle.py:190`](../bonsai_forensics/oracle.py) (`decode_q2_0_g64` reads `d`
  from bytes 16..17 and `qs` from 0..15); mainline ggml defines
  `block_q2_0 { ggml_half d; uint8_t qs[QK2_0/4]; }`, i.e. `d` first
  (`~/.unsloth/llama.cpp/ggml/src/ggml-common.h:194–199`). The offline test
  ([`tests/ternary/test_oracle.py:35`](../tests/ternary/test_oracle.py)) encodes the same wrong layout, so it is
  self-consistent but not a real validation.
  [`research/gate3-qat-verdict.md`](../research/gate3-qat-verdict.md) (lines 29–30, 82); `project handoff` §7;
  project records Round 12.
- **Status:** open.
- **What it rules out:** trusting `Q2_0_g64` agreement numbers from [`oracle.py`](../bonsai_forensics/oracle.py)
  until fixed; using this decoder for the 1.7B oracle comparisons.
- **Cost to revisit:** low (a few lines + one regression test in
  [`tests/ternary/test_oracle.py`](../tests/ternary/test_oracle.py)).
- **Worth re-assessing?** Yes — cheap correctness fix.

---

## Seeded-entry verification notes

All seeded entries F1–F10 and U1–U3 were checked against their sources before writing.
Results:

- **F5 — contradiction found, then resolved by owner clarification.** The seed
  says "quality gate failed", but project records Round 9 ("T28 COMPLETE — quality
  gate PASSED") and the T28 task row ("gate passed, best 1.103×") both record
  the opposite. Owner clarification: "failed" meant the **mission acceptance
  rate** (we expected to match Prism's retention and chased the last 8%), not
  the T28 1.44× gate. Both readings are correct against their respective bars.
  Recorded in the F5 entry as `[rev 2026-09-20]` and `[rev 2026-09-20b]`; the
  entry is now framed as a *deliberate scope stop*, not a methodological
  dead-end. This is the only seeded entry whose stated verdict did not survive
  verification unchanged.
- **F1 — artifact nuance (not a contradiction).** The 2.106× fixed-passage
  reproduction is attested in commit `ea089e2` and project records Round 6. The
  persisted `artifacts/reference/*.json` files are the *held-out*
  (real-calibration) runs and show the collapse (1178× / 9184× / 34877×); the
  fixed-passage run was not persisted under a matching filename. Both facts
  agree with the reports.
- **F9 — RAM figures.** The seed's "~20 GB RAM" matches the host's *available*
  RAM (~19 GiB of 30 GiB) and the 14 GB per-job cap used for GPU jobs;
  project records §1 lists the raw total as "30 GB RAM (+30 swap)". Not a
  contradiction, but note the distinction between total and usable RAM.
- **F2/F3** are recorded as "T32/Gate 3" in the seed; T32 was a diagnostics-only
  round with no source commit — evidence is the `artifacts/gate3/`
  reports plus the [`research/gate3-qat-verdict.md`](../research/gate3-qat-verdict.md) write-up and project records R12.
- **F6/F7/F8** PPL figures were verified directly from the saved eval logs and
  JSONs (1,252,835 / 1,055,018 / 172.28 vs 115.91).
- **F10** was verified against the actual server log
  (`logs/llama_server_8090.log`), not just the write-up.
- **U1/U2/U3** are recorded from project records / `project handoff`; U3 was
  additionally confirmed against the [`oracle.py`](../bonsai_forensics/oracle.py) source.

No other contradictions were found between the seeded entries and the reports.
