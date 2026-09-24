# DSpark export audit — the smoke draft is not fork-loadable

**Status:** blocker found (2026-09-24). The locally trained smoke draft
(`artifacts/dspark/draft-medium.pt`) **cannot** be exported to a GGUF the
PrismML-Eng fork will load. Export + benchmark via the fork requires a draft
with the production `Qwen3DSparkModel` topology, which the smoke model is not.

Reproduce:

```sh
# ROCm python (has torch); audits the checkpoint's tensor inventory
~/.unsloth/studio/unsloth_studio/bin/python scripts/pilot/dspark_export_check.py \
    --checkpoint artifacts/dspark/draft-medium.pt --hidden-correction --confidence-head
```

## What the fork requires

`src/models/dspark.cpp::load_arch_tensors` creates a fixed inventory. Tensors
with flag `0` are **required**: a missing one aborts `llama_model_load`. The
layout (numpy `(out, in)` order) is:

| tensor | shape | required |
|---|---|---|
| `token_embd.weight` | `(vocab, n_embd)` | yes |
| `output_norm.weight` | `(n_embd,)` | yes |
| `output.weight` | `(vocab, n_embd)` | optional (else duplicated from `token_embd`) |
| `dspark.fc.weight` | `(n_embd, n_capture·n_embd)` | yes |
| `dspark.hidden_norm.weight` | `(n_embd,)` | yes |
| `dspark.correction_{hidden_norm,embed_norm}.weight` | `(n_embd,)` | if correction |
| `dspark.correction_{gate,up}.weight` | `(correction_size, 2·n_embd)` | if correction |
| `dspark.correction_down.weight` | `(n_embd, correction_size)` | if correction |
| `dspark.markov_head_{a,b}.weight` | `(vocab, markov_rank)` | if `markov_rank>0` |
| `dspark.confidence_head.weight/.bias` | `(1, n_embd(+rank))` / `(1,)` | if enabled |
| `dspark.log_snr_fc1/fc2.{weight,bias}` | `128`/`n_embd` | if `log_snr_conditioning` |
| `blk.i.attn_norm.weight` | `(n_embd,)` | yes |
| `blk.i.attn_{q,k,v}.weight` | `(n_head·head_dim, n_embd)` etc. | yes |
| `blk.i.attn_output.weight` | `(n_embd, n_head·head_dim)` | yes |
| `blk.i.attn_{q,k}_norm.weight` | `(head_dim,)` | **yes** |
| `blk.i.ffn_norm.weight` | `(n_embd,)` | yes |
| `blk.i.ffn_{gate,up}.weight` | `(intermediate, n_embd)` | yes |
| `blk.i.ffn_down.weight` | `(n_embd, intermediate)` | yes |

The forward is a **cross-attention** drafter (`graph::graph`): the raw
concatenated target taps `[n_capture · n_embd]` are projected once by
`dspark.fc` into `n_embd`, RMS-normalized (`dspark.hidden_norm`), and that
static context is concatenated along the *sequence* axis with each layer's
normed draft residual before that layer's q/k/v projection. The draft embedding
is the plain `token_embd` (no pre-fc concat). `hidden_correction` is a
deferred/auxiliary head, not part of this graph.

The production template `deepseek-ai/dspark_qwen3_4b_block7` (config cached
locally) is `architectures: ["Qwen3DSparkModel"]`: hidden 2560, 5 layers, GQA
`32/8` heads, `head_dim` 128, `markov_rank` 256, `target_layer_ids` of length 5,
no hidden correction, no log-SNR. That is the layout the fork + converter are
built for.

## What the smoke draft supplies

`artifacts/dspark/draft-medium.pt` (912 M params): hidden 1024, 4 layers,
`heads/kv = 16/4`, `head_dim` 64, intermediate 4096, vocab 248320, `markov_rank`
256, a **hidden-correction MLP** over the concatenated 5 target taps, and a
`fc` that consumes `concat(embed, correction)`.

Audit result: **51 actual vs 58 expected** tensors.

- **missing (9):** `blk.{0..3}.attn_{q,k}_norm.weight`, `dspark.hidden_norm.weight`.
  The per-layer qk-norms are required, so the loader aborts immediately.
- **extra (2):** `pre_fc_norm_embedding.weight`, `pre_fc_norm_hidden.weight` —
  no counterpart in the fork; `conversion/dspark.py` would fail to map them.
- **shape mismatch (5):** `dspark.fc.weight` `[1024,2048]` vs `[1024,5120]`,
  `correction_embed_norm` `[25600]` vs `[1024]`,
  `correction_gate`/`correction_up` `[4096,25600]` vs `[1024,2048]`,
  `correction_down` `[1024,4096]` vs `[1024,1024]`.

Root cause: `bonsai_forensics/dspark.py` is a deliberately reduced proof of
concept ("this module is for smoke training only" — its own docstring), not the
DeepSpec `Qwen3DSparkModel` wiring. The difference is topological (where the
target context enters; whether the draft concatenates its embedding), not a
rename or transpose, so no converter can bridge it.

## Converter note

`convert_hf_to_gguf.py --dspark` is hard-restricted to `DeepseekV4ForCausalLM`
(`if model_architecture != "DeepseekV4ForCausalLM": error`). The general
`DSparkModel` path (`conversion/dspark.py`) is registered for
`Qwen3DSparkModel` / `DSparkForCausalLM` / `DsparkSpeculator` / `DSparkModel`.
Our draft must therefore be exported with `architectures:["Qwen3DSparkModel"]`
and the exact inventory above, not via `--dspark`.

## Paths forward

1. **Train to the fork layout (the real path to a Bonsai-2 drafter).** Rewrite
   the draft to mirror `Qwen3DSparkModel` (qk-norm per layer, `fc` over raw
   `n_capture·n_embd` taps + standalone `hidden_norm`, plain `token_embd`
   embedding, no pre-fc norms), then retrain on the existing Bonsai-2 feature
   set. The collected features already match the fork's `fc` input exactly:
   5 layers × 5120 = 25600 = `n_capture · n_embd`. Cost: for a 27B target
   (`n_embd` 5120) the draft is hidden-5120, so it is materially bigger than the
   2560-hidden Qwen3-4B template; this is a GPU run, not the "small, local,
   completable" object the track plan assumed.
2. **Convert a real DeepSpec/Qwen3DSparkModel checkpoint** to validate the fork
   runtime + benchmark harness. This exercises the pipeline end to end but the
   target is the template's own (Qwen3-4B), not Bonsai 2 — useful as a baseline,
   not the novelty.

## Path 2 recipe (conversion validated, harness runs)

The real template converts and loads; only its weights were missing locally.

```sh
# 1. fetch the 2.8 GB template
huggingface-cli download deepseek-ai/dspark_qwen3_4b_block7 \
    config.json model.safetensors --local-dir artifacts/dspark/template

# 2. convert the DRAFT through the dspark converter (see below for why not the
#    stock convert_hf_to_gguf.py), and the target as usual
python scripts/pilot/convert_dspark_draft.py artifacts/dspark/template \
    --outfile artifacts/dspark/template-4b-block7-dspark-f16.gguf --outtype f16
python ~/llama.cpp/convert_hf_to_gguf.py <Qwen3-4B-snapshot> \
    --outfile artifacts/dspark/qwen3-4b-f16.gguf --outtype f16

# 3. benchmark with the fork's real-eval harness
scripts/pilot/run_dspark_bench.sh artifacts/dspark/qwen3-4b-f16.gguf \
    artifacts/dspark/template-4b-block7-dspark-f16.gguf
```

**Converter trap.** `conversion/__init__.py::TEXT_MODEL_MAP` lists
`"Qwen3DSparkModel"` **twice** — `"dspark"` and later `"qwen"` — so the stock
`convert_hf_to_gguf.py` resolves it to `conversion.qwen.DSparkModel`, a
`DFlashModel` subclass that emits a `dflash`-arch GGUF (`dflash.target_layers`,
no `dspark.*` KVs).  The DSpark runtime
(`common_speculative_impl_draft_dspark`, `llama_model_dspark_get_meta`) and
`tests/test-dspark-real-eval` require arch `dspark` with `dspark.dspark.*` KVs,
which only `conversion.dspark.DSparkModel` writes.  `scripts/pilot/convert_dspark_draft.py`
imports `conversion.qwen` then `conversion.dspark` so the DSpark registration
wins, and runs the stock converter.  Verified output: arch `dspark`, 64 tensors,
`dspark.dspark.{block_size=7, markov_rank=256, mask_token_id=151669,
target_layers=[1,9,17,25,33], confidence_head=true,
confidence_head_with_markov=true, hidden_correction=false}`.

**Harness trap.** The stock examples cannot drive a DSpark draft:
`llama-cli`/`llama-speculative` segfault because dspark's `process()` needs the
target's multi-layer tap captured via `llama_set_capture_layers` with
`common_speculative_process()` on every target batch.  Only
`test-dspark-real-eval` wires that up, so `run_dspark_bench.sh` uses it.

**Outcome (2026-09-24): path 2 is a dead end for the public template.** The
template converts to a valid `dspark` GGUF and the harness reads its meta
correctly (`n_capture=5, block_size=7, markov_rank=256, mask_token_id=151669`,
`target_layers=[1,9,17,25,33]`), but it cannot run: the draft needs the target's
intermediate hidden states, and `set_capture_layers` is implemented **only** for
`qwen35` (`src/models/qwen35.cpp`), `deepseek4`, and `nemotron-h` targets.  The
template's target is Qwen3-4B (`qwen3`), whose graph produces no capture tensor,
so the harness fails with `target graph did not produce requested capture
layers`.  Bonsai 2 is `qwen35`, so **path 1's target does support capture** —
which is precisely why the DSpark track can only be finished on Bonsai 2, not by
benchmarking the public Qwen3 template.

Until path 1 lands, the DSpark track is **blocked on a fork-compatible draft
architecture**; path 2 validated the converter/runtime plumbing but cannot
produce acceptance numbers.
