# DSpark for Bonsai 2 — Path 1 plan (fork-compatible drafter)

**Goal.** Train the first DSpark speculative draft for Bonsai 2 (and the first
ternarizable drafter) whose weights load in the PrismML-Eng fork's `dspark`
runtime. This supersedes the smoke draft, which was topologically
incompatible (`DSPARK-EXPORT-AUDIT.md`).

Why this is the only route: the runtime needs the target's intermediate hidden
states, and capture is implemented only for `qwen35` / `deepseek4` /
`nemotron-h` targets (`src/models/*.cpp`). The public Qwen3-4B template's target
is `qwen3` and cannot be benchmarked. **Bonsai 2 is `qwen35`, so its target
supports capture.**

## Target & data (all local)

| item | value |
|---|---|
| target | Bonsai 2 27B (`qwen35`): hidden 5120, 64 layers, heads 24/4, head_dim 256, ff 17408, vocab 248320 |
| capture layers | `[5, 19, 33, 47, 61]` (`n_capture = 5`) |
| features | `artifacts/dspark/features/` — 288 sequences, per-position taps `[T, 25600]` fp16 (= 5×5120) |
| tokens | the same sequences' token ids (draft input / label stream) |

The feature width exactly equals the runtime's `fc` input `n_capture·n_embd`, so
the collected data already matches the fork's tap contract.

## Runtime topology to mirror (`src/models/dspark.cpp`)

- `dspark.fc`: `(n_embd, n_capture·n_embd)`, then `dspark.hidden_norm`
  (RMSNorm `n_embd`) → the **static target context**, one row per captured
  target position. Computed once per call.
- Trunk, `N` layers; each layer:
  1. `attn_norm` on the draft residual `x` (`[T_draft, n_embd]`);
  2. `seq = concat( target_ctx, norm(x) )` along the **sequence** axis;
  3. `q/k/v = proj(seq)` (q `n_head·head_dim`, k/v `n_head_kv·head_dim`);
  4. RMSNorm `q_norm`/`k_norm` over `head_dim`, then RoPE;
  5. **non-causal** attention (all rows attend to all), `o_proj`, keep the
     trailing `T_draft` rows;
  6. residual, then `ffn_norm` → SwiGLU (`gate`,`up`,`down`) → residual.
- `output_norm` → `output` (lm_head). Draft ships its own `token_embd`
  (`dspark` arch requires it); the output head may be shared
  (`LLAMA_DSPARK_SHARED_HEAD=1` → `dspark_head_source`).
- Markov head `markov_head_a` (prev-token embed, `[vocab, rank]`) ×
  `markov_head_b` (`[rank, vocab]`): the semi-autoregressive logit bias. With
  `hidden_correction=0` (the template default) it is applied host-side by the
  runtime (`common_speculative_impl_draft_dspark`), not in the graph.
- Optional: hidden correction (host-side GIDD loop) and log-SNR conditioning.
  Start without both, matching the template.

Chosen draft config: hidden 5120, 5 layers, heads 32/8, head_dim 128,
ff 9728, `markov_rank` 256, `block_size` 7, mask token = last id. Trunk ≈ 1.1 B;
`token_embd` 1.27 B; output head shared or 1.27 B.

## Objective

Block-diffusion drafter: over each `block_size=7` window,
`input = mask_token_id` for the block, `context = target taps of the preceding
positions`, `label = the block's real tokens`; next-token CE per slot, plus the
Markov bias chained on the previous (anchor/decoded) token. Stage 2 (optional):
on-policy passes against the target, as DeepSpec does.

## Compute budget (2×20 GiB, gfx1100)

- **Card convention:** HIP device **0** drives the desktop
  (Xwayland/plasmashell) — always run training/eval on the **other** card via
  `HIP_VISIBLE_DEVICES=1`.  The launchers default to this; `--device cuda:0`
  then maps to the free card.

- Freeze `token_embd` (init from the target) → trainable ≈ trunk (1.1 B).
  Adafactor + bf16: weights 2.2 GB, grads ~4.4 GB, state ~4.4 GB ≈ 11 GB →
  fits one card, leaving the second free for eval.
- Context rows dominate activations: train on windows (e.g. `T_ctx ≤ 512`,
  `T_draft = 7`) and accumulate; do not backprop a full 4096-row context.
- Features are fp16 on disk (13 GB); stream per sample (the lazy loader
  pattern in `scripts/pilot/dspark_train.py`).

## Milestones

1. **Fork-exact module** — `bonsai_forensics/dspark_prod.py`: parameter
   inventory + `fork_spec()` so `scripts/pilot/dspark_export_check.py` passes.
   ✅ **Done.** `DSparkProdDraft` reproduces the real
   `deepseek-ai/dspark_qwen3_4b_block7` template's **64 tensors name- and
   shape-for-name** (config `hidden 2560 / 5 layers / 32-8 heads / head_dim 128
   / ff 9728 / vocab 151936`), and the tiny-config inventory passes the fork
   spec (`tests/ternary/test_dspark_prod.py`).
2. **Forward parity** — the PyTorch trunk reproduces the runtime's per-layer
   concat/qk-norm/non-causal attention.
   ✅ **Done.** `scripts/pilot/dspark_forward_parity.py` builds a tiny
   `dspark`-arch GGUF from a `DSparkProdDraft`, writes the fork's `ref.bin` from
   our forward, and runs `tests/test-dspark-forward --tier2`: **argmax 4/4,
   top-5 5/5, max abs diff 4.2e-3** (≈6e-5 relative on ~70-scale logits), no
   non-finite. RoPE is NeoX, matching `LLAMA_ARCH_DSPARK`.
3. **Smoke train** — validate the block-diffusion batching/loss/optimizer on
   **synthetic** taps at tiny dims (CPU).
   ✅ **Done.** `scripts/pilot/dspark_prod_train.py` (with `make_block` unit
   tests) runs finite gradients; synthetic targets are uncorrelated noise, so
   the loss sits at `ln(vocab)` by construction — it validates mechanics only.
   Note: the real 25600-wide taps force the draft `n_embd` to 5120 (the
   runtime's `fc` maps `n_capture·n_embd → n_embd`), so a genuinely tiny run
   cannot consume real features; the real run is inherently full-width and needs
   the target's `token_embd`/head initialised to learn anything.
4. **Full train** — hidden 5120, 5 layers, frozen `token_embd`, Adafactor,
   windowed context; save an HF-format checkpoint with
   `architectures:["Qwen3DSparkModel"]`.
   ◐ **Started.** `scripts/pilot/dspark_target_init.py` recovers the target's
   deployed `token_embd` + `output` (`prism_loader.recover`, `W_hf = D @ R`) to
   `artifacts/dspark/bonsai2-init.pt` (5.1 GB bf16); the draft initialises from
   it and freezes both, so the trunk trains against a head that can already
   predict.
   - End-to-end on the real taps: 2.88 B params, 342 M trainable (1 layer).
   - **Overfit sanity passes:** one fixed block, 300 steps, Adam 1e-3 →
     loss **12.63 → 0.0015**, so the objective/loading/init are sound.
   - 50 steps at lr 1e-4 was flat (~13.0 = `ln 248320`) — too short/small, not
     a bug; a 1500-step run at Adam 2e-4 generalises: **loss 13.19 → 6.60**
     (mean-last-5 6.55) across the 288 real samples, well below `ln 248320`.
5. **Export + bench** — `scripts/pilot/convert_dspark_draft.py` →
   `dspark`-arch GGUF → `scripts/pilot/run_dspark_bench.sh` against the Bonsai 2
   target (capture supported); report accept rate / `tau` / tok-s.
   ◐ **Started.** `scripts/pilot/export_dspark_hf.py` writes the trained
   checkpoint as a `Qwen3DSparkModel` HF dir (18 tensors, fp16), converted to
   `artifacts/dspark/bonsai-draft-1l.gguf` (5.77 GB, arch `dspark`,
   `target_layers=[5,19,33,47,61]`, `block_size=7`, `markov_rank=256`).
   ✅ **Ran end-to-end on the released `Ternary-Bonsai-2-27B-PQ2_0` target**
   (24 prompts, block 7): AR parity PASSED; **accept 0.0030, tau 1.0208**
   (depth-1 0.0208); `ar_tok_s=35.4` vs `sp_tok_s=2.9` (**speedup 0.083**).
   The 1-layer trainer is far too weak to accept, and the host-side Markov
   correction dominates wall time (`correction_calls=768,
   correction_wall_us=173 s`) — the "drafting is cheap, verification/correction
   dominates" caveat, quantified.
   **5-layer run (6000 steps):** accept **0.0203** (1-layer 0.0030), tau 1.1422
   (0.1045 at depth 1); speedup still 0.10.  `scripts/pilot/dspark_prod_eval.py`
   shows held-in block **top-1 0.0996 / top-5 0.243** — i.e. the runtime's
   depth-1 accept ≈ the draft's top-1 accuracy, so acceptance is **quality-bound**,
   not a loop bug.  (The 6k run trained through the `test_sft` split; the trainer
   now has `--exclude-prefix` for a real hold-out.)  Next: raise top-1 and cut
   the correction cost.

## Training-efficiency finding (2026-09-24)

The 6k run was **badly under-trained, not capacity-bound**: one block per step ×
6000 steps = 6000 blocks, versus ≈42k distinct blocks in the 288-sequence
feature set — **~0.14 epochs**.  The trainer now batches `--batch-blocks N`
blocks per step (fixed context width so they stack); a hold-out run at 5 layers
/ ff 4096 / 8 blocks × 5000 steps ≈ 1.1 epochs is the current lever on top-1
(and therefore acceptance).  Data scale (more `dspark_collect` sequences) is the
next lever if top-1 plateaus.
6. **Optional novelty** — export the draft as PQ2_0 (ternarized drafter).

## Risks

- **Context length vs memory**: the honest use is long target context; training
  windows must approximate it without distribution shift.
- **Markov correctness**: the semi-AR chaining must match the runtime's
  host-side resample or acceptance suffers.
- **Parity**: any deviation from `src/models/dspark.cpp` (rope layout, qk-norm
  placement, concat axis) silently lowers acceptance; milestone 2's parity test
  is the guard.
- **Not small**: the draft is hidden-5120 by construction (`fc` maps
  `n_capture·n_embd → n_embd` with `n_embd` the target width), so this is a GPU
  project, unlike the original track-plan framing.
