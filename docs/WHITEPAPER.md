# Ternary Bonsai Replication (TBR) — White Paper

> **Status:** source of truth for the TBR programme, as of 2026-09-20.
> **Companion:** [`docs/FAILURES.md`](FAILURES.md) — the failure register. Every
> falsified route named here is stated in one line and carries an `F<n>` pointer
> to its full entry (claim, method, evidence, cost to revisit).
> **Spec hash:** `0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b`
> **Runtime write-up:** [`docs/BONSAI-RUNTIME.md`](BONSAI-RUNTIME.md)

This document records what is *verified*, what end-state was chosen, and why.
It is deliberately brief on negative results: load-bearing falsifications are
stated inline (omitting them would make the paper misleading by omission) and
expanded only in the register.

---

## 1. Mission and acceptance gate

Replicate PrismML's **Bonsai-2 27B** achievement on `Qwen/Qwen3.8-27B`: true
end-to-end ternary weights (~2 bpw, `{−1,0,+1}`), runnable on 2× RX 7900 XT via
the Prism ROCm fork, evaluated inside hivebench.

The programme-level acceptance gate (HIVE-PLAN §0, T20) was:

1. GGUF ≤ 8.0 GB that loads on `gfx1100`; ≥60 tok/s tg128.
2. KLD ≤ 2× Bonsai-2 `PQ2_0`, same corpus/binary/day.
3. hivebench A/B: no category regression >2 points vs the released Prism arm;
   offline suite green.
4. Ablation report (A/B/C) with fixed-seed evidence.
5. Reproducible from pinned hashes; run-log records cost/GPU-hours/config hash.

**Non-goal:** reproducing Prism's proprietary recipe. The target is the
*result*; parity is measured, not assumed.

---

## 2. Verified ground truth

Everything in this section was measured against the released files and the
public base, not inferred from marketing.

| Fact | Value / source |
|---|---|
| Base model | `Qwen/Qwen3.8-27B`, HF `main` sha `1d4bf0f2…` (the pin) |
| Released artifact | `Ternary-Bonsai-2-27B-PQ2_0.gguf`, 7.2 GB, sha `3907dc16…` |
| Format | ternary g128, FP16 scale/group, blockwise Walsh–Hadamard rotation `R = (1/√n)·Hₙ·diag(S)`, n=1024 |
| Rotation convention | **input-axis, sign-then-FWHT, block 1024, no norm fold** (T30) |
| Layouts in the released file | `in_proj_qkv` and `in_proj_z` V rows **tiled**; `ssm_out` stays **grouped**; `ffn_*`/`attn_*` plain; embeddings input-rotated |
| Norm convention | GGUF norms are `1 + w` (HF's `(1+w)` RMSNorm parameterisation) |
| Full-precision exemptions | `in_proj_a/b`, `conv1d`, `A_log`, `dt_bias`, norms, `q_norm/k_norm` (~0.0976% of params) |
| Basis reproduction (Gate 1) | plain absmean RTN of the rotated base matches Prism's trits at **0.920 mean / 0.885 min** (n=26 tensors); chance is 0.33 |
| Runtime (T29) | Prism fork, ROCm: **pp512 399 t/s, tg128 39.6 t/s**; held-out PPL **17.86** (R8), **18.5851 ± 1.209** under the controlled Gate-2 protocol |
| Container is real | their F16 dequant of the 1.7B `PQ2_0` equals our byte-exact decode; packing is lossless |

**The decisive fact:** Prism's quality lives in their *trained* weights, not in a
public quantizer. Gate 3 proved no error-compensation variant of the public base
reproduces their codes (F2), stochastic-rounding noise has no effect (F3), and
their codes fit our activation Hessians *worse* than plain RTN (1.072×). The
only mechanism left standing is weight movement during training.

---

## 3. Chosen end-state: Track B (released weights via the fork)

Two tracks were considered:

- **Track A — build our own artifact** (PTQ → QAT/KD). Closed for parity:
  the quantizer reverse-engineering track is closed (F2, F3), local
  rotate+absmean RTN of the base collapses to 23,606 PPL vs their 18.5851
  (1,270×; F4), and the 1.7B QAT/KD proof reached 1.103× / 90.6% retention,
  short of the ≥97% mission target and corpus-limited (F5).
- **Track B — run Prism's released `PQ2_0` on the Prism ROCm fork and evaluate
  it in hivebench.** This is the pragmatic end-state and the shipped result.

Track B is the deliverable: the released 27B runs on **one** RX 7900 XT at
39.6 t/s, and the hivebench memory layer measurably improves context fidelity
over a FIFO baseline (§5). Track A is not abandoned as a *research* direction —
it is simply a training problem, not a quantization one, and no local per-layer
KD variant solved it (F6, F7).

**Why the last 8% was not chased.** Gate 2 showed the ~8% trit residual carries
the entire quality gap (F4); Gate 3 showed that residual is *trained weight
movement*, not a quantizer trick (F2, F3); local per-layer KD cannot control
global compounding (F6, F7). That leaves **end-to-end QAT/KD at 27B as the only
remaining route** to Prism's exact acceptance. The programme stopped there *by
decision*: the fix is now known and priced, the mission's non-goal is the
recipe, the result is public, and Track B already ships it. The defensible claim
this supports is **"the gap is localized and reachable"** — *not* "replication
is proven". Gate 3 established that *some* training moved their weights, not
that QAT/KD specifically reproduces their numbers, and the 1.7B QAT run was
itself short of target. See F5.

---

## 4. Load-bearing falsifications (one line each)

Full entries, evidence and revisit costs are in [`FAILURES.md`](FAILURES.md).

- **F1** Public PTQ at ~2 bpw does not survive off-calibration; the published
  2.1× is a calibration-passage artifact (our canary 3514×, ThakiCloud
  reference 1.2e3–3.5e4× held-out).
- **F2** Every GPTQ/Hessian variant moves codes *away* from Prism's; RTN 0.9145
  vs GPTQ 0.8086–0.8497 → the quantizer reverse-engineering track is closed.
- **F3** Stochastic Calibration Resonance (5% calibration noise/dithering) is
  falsified: 4.3% of codes moved, agreement changed by **+0.003 pp**.
- **F4** rotate+absmean RTN of the public base = **23,606 PPL** vs their
  **18.5851** (1,270×) in the same binary/corpus/settings → no-rental RTN parity
  is dead.
- **F5** The last ~8% was localized to end-to-end QAT — every cheaper route was
  eliminated (F1–F4, F6, F7) and the 1.7B proof reached **1.103× / 90.6%**
  retention, short of the ≥97% target; the 27B proof-run was priced and
  **deliberately not funded** because the result is public and Track B ships it.
- **F6** Student-stream block-wise KD = **1.25 M PPL**, worse than RTN
  (per-layer compounding).
- **F7** Teacher-forced block-wise KD = **1.06 M PPL**, still worse than RTN;
  local per-layer KD cannot control global compounding.
- **F8** Entropy/excess-loss data selection lost to random (172.3 vs 115.9 PPL).
- **F9** 27B full-model CPU offload swap-thrashes (55.6 GB weights vs 40 GB VRAM
  + ~20 GB usable RAM); the prefix trick (embedding + first N layers) avoids it.
- **F10** `-ngl 99` on one 20 GB card fails to allocate (`unable to allocate
  ROCm0 buffer`); **resolved** by unsloth-style auto-fit `ngl`.
- **F11** Sibling `strata-memory` drift (`cortex.config.StrataConfig` gone)
  blocked live harness import; **resolved** operationally via the F6 pin
  workaround (durable T23 fix still pending).
- **F12** Two independent ROCm contexts hang GPU1 at firmware level
  (SMU→PSP/TOC `error -22` on next boot); policy: **one heavy ROCm process at a
  time**, two cards only via one model-parallel process.
- **F13** `oracle.decode_q2_0_g64` (type 42) decodes garbage; the T7 1.7B 0.612
  agreement came from the F16-dequant path, not this decoder — **open**.

---

## 5. The evidence chain (Gates 1–3 + T28/T29)

| Gate | Question | Result |
|---|---|---|
| **T30 / Gate 1** | Does rotate+RTN in Prism's basis reproduce their trits? | Basis **cracked**: 0.920 mean / 0.885 min agreement. Residual is *structural* (non-monotone within groups, denser codes, ~10% lower weight-space error than RTN) — fingerprinted as error-compensated or QAT weights. |
| **T31 / Gate 2** | What does the residual cost? | A byte-controlled swap of all **402** `PQ2_0` payloads for rotate+absmean RTN of the public base collapses PPL **18.5851 → 23,606 (1,270×)**. Metadata/exemptions stay byte-identical; 402/402 post-patch SHA-256 verified. |
| **T32 / Gate 3** | Quantizer trick or trained weights? | **Trained weights.** SCR +0.003 pp (F3); GPTQ/H-variants all move codes away (RTN 0.9145 vs 0.8086–0.8497, F2); their codes' activation-weighted error is 1.072× RTN's. The reverse-engineering track is closed. |
| **T28** | Can local QAT/KD close the gap? | 1.7B reached **1.103×** held-out (bar 1.44×) — 90.6% retention; run3 overfit at 7k steps on a 301k-token corpus. This is the last lever: the ~8% residual is trained weights (Gates 2–3), so end-to-end QAT is the only route to Prism's exact acceptance — known, priced, deliberately not run (F5). |
| **T29** | Ship the released model. | Served on the Prism fork, smoke **5/5**, hivebench eval below. |

---

## 6. Hivebench evaluation (T29)

Released `Ternary-Bonsai-2-27B-PQ2_0` served by the Prism fork on GPU1
(`HIP_VISIBLE_DEVICES=1`), driver `bonsai_forensics_eval.py`, `--no-thinking`.
Smoke: **5/5 PASS** (coherent: `Paris`, `51`, `ternary.`), ~0.4–0.9 s/reply.

Paired A/B, **124 turns compared** (107 first-mention and 10 no-fact turns
excluded):

| Metric | hive | FIFO |
|---|---|---|
| answer recall | 72.6 | **77.4** |
| avg fact hit ratio | 0.731 | **0.789** |
| avg context fidelity | **0.306** | 0.205 |

- `fidelity_hive_gt_fifo_ratio` **69.4%**; `hive_ge_fifo_ratio` **89.0%**;
  `ctx_hive_ge_fifo_ratio` 98.2%.
- Outcome split: `hive_only` 4, `fifo_only` 10, `both_sufficient` 77,
  `neither_sufficient` 33.

**Reading:** the memory layer improves *context fidelity* over FIFO
(+0.101 absolute, strict win ratio 69.4%) while trailing on raw answer recall
(−4.8 points). It is a real, measured behavioural effect, not a headline
"beats FIFO" claim.

---

## 7. Conclusions

1. **The released weights are the product.** Track B is verified and shipped:
   7.2 GB, one RX 7900 XT, 39.6 t/s generation, PPL 18.5851 under the controlled
   protocol.
2. **Our own ternary artifact is a training problem — fully localized,
   deliberately not chased.** Every non-training route was eliminated
   (F1–F4, F6, F7); the ~8% residual that carries the quality is trained weight
   movement, so end-to-end QAT/KD at 27B is the only remaining lever. It is
   known and priced (a real corpus plus a 1×48–80 GB rental or a large local
   pilot) and was not funded, because the result it would reproduce is already
   public and shipped via Track B. What is proven is *where* the gap is, not
   that the recipe reproduces Prism's numbers (F5).
3. **The rotation basis and format are fully mapped** and reusable (F1–F4 do
   not invalidate the container facts in §2 — they invalidate the shortcut of
   filling that container with public PTQ weights).
4. **The memory layer is a modest, measurable win on context fidelity**, not a
   blanket quality win.
5. **Gate status is partial, and stated plainly.** The T20 gate in §1 was
   written for *our own* artifact and is only partly satisfiable by Track B:
   item 1 (size/load) passes but throughput is **39.6 t/s, below the 60 t/s
   target**; item 2 (KLD vs Bonsai) is satisfied trivially because the shipped
   model *is* Bonsai; item 3 (hivebench A/B vs the Prism arm) is **not run**
   (out of scope here); item 4 (A/B/C ablation) was never reached because the
   PTQ runs it depends on are blocked; item 5 (regenerate our own artifact) is
   not met. This paper does not claim the T20 gate was passed.

---

## 8. Provenance and reproduce

```sh
# serve + smoke (one heavy ROCm process at a time; pin the device if shared)
HIP_VISIBLE_DEVICES=1 python scripts/eval_llama_server.py smoke \
    --server-bin /path/to/prism-fork/llama-server \
    --gguf /path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf \
    --ngl 99 --ctx 4096 --no-thinking \
    --out artifacts/eval/bonsai27-smoke.json

# controlled perplexity protocol (Gate-2 settings)
HIP_VISIBLE_DEVICES=1 python scripts/eval_llama_server.py perplexity \
    --perplexity-bin /path/to/prism-fork/llama-perplexity \
    --gguf /path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf \
    --corpus artifacts/corpus/tinyshakespeare.txt \
    --ngl 99 --ctx 512 --chunks 8 \
    --out artifacts/eval/bonsai27-ppl.json
```

Key evidence (produced by the scripts here; the large artifacts are not
redistributed): `artifacts/gate1/gate1-report.json`,
`artifacts/gate2/{eval-report,patch-report}.json`,
`artifacts/gate3/{scr-report.json,prefix27/}`,
`artifacts/recover/run1/recover-report*.json`.

The gate work was originally done in the local `hivebench` repository; the
commit hashes cited throughout (`5eac0c0` T31, `cfea698` T30, `ea089e2` T27,
`8cd8038` T7, `64ff7bc` T28, `aaddd0b` T29 docs) are provenance from there.

---

## 9. Open items and risks

- **F13 (open bug):** `oracle.decode_q2_0_g64` (type 42) decodes garbage; fix
  when next touching `bonsai_forensics/oracle.py`. Not on the 27B path.
- **F11:** T23 durable sibling-compat fix still pending; live harness imports
  currently require the F6 pin.
- **F12:** ROCm single-context policy is a workaround, not a root fix.
- **Artifact dependency (Track B):** the shipped capability relies on Prism's
  released `PQ2_0` weights and their terms; if that artifact were withdrawn or
  restricted, the capability would have to be rebuilt. F5 is the map for that
  rebuild (end-to-end QAT/KD at 27B), but it has not been demonstrated.
- **Pending, out of scope here:** head-to-head A/B (`scripts/run_ab.sh`), the
  50-conversation eval, LoRA adaptation around the released weights, and the
  BF16 reference download.

See [`FAILURES.md`](FAILURES.md) for every entry's status and the cost to
revisit.
