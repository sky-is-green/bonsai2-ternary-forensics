# Reference: `ProCreations/Ternary-Bonsai-2-27B-MTP`

**Third-party work (Apache-2.0), reviewed 2026-09-24.** Files mirrored locally
under `artifacts/reference/procreations-mtp/` (gitignored; not redistributed).
This note records what it confirms and what we reuse — not a claim about our
recipe.

## What it is

A **trained MTP head added to the released Bonsai 2 27B**. It is *not* a new
ternarisation:

- the base's **851 tensor payloads are byte-identical** to Prism's
  `Ternary-Bonsai-2-27B-PQ2_0.gguf`;
- only the ~**424.7M-param Qwen MTP head** is trained (15 BF16 tensors sourced
  from `Qwen/Qwen3.8-27B` @ `1d4bf0f2…`, verified byte-for-byte against the
  official release);
- the head ships as **Q8_0**; normalization weights stay floating-point.

## What it confirms about the format

1. **The embedding is stored Hadamard-latent.** `runtime/bonsai-mtp-embedding.patch`
   shows the MTP path must apply `llama_mul_mat_hadamard` + signs to the token
   embedding before use — i.e. the embedding table is stored in the *rotated*
   basis and inverse-rotated at lookup. This is the "402nd tensor" handling we
   flagged as a P0.
2. **The MTP head is outside the 851 base tensors** — matching our
   `targets.py` `mtp` exclusion.
3. **Numerical parity is achievable to ~1e-5.** `reports/numerical-parity.json`:
   cosine **0.99997–0.99999**, top-1 equal, max hidden-state error ~0.4. That is
   the bar for our exporter.

## What we reuse

| asset | use |
|---|---|
| `runtime/bonsai-mtp-embedding.patch` | runtime-side embedding inverse-rotation (our P0) |
| prebuilt patched runtimes + `build-runtime.sh` | MTP-capable Prism fork, no build-from-scratch |
| `training/parity.py`, `reports/numerical-parity*.json` | template for our code/scale + packed-artifact parity |
| `reports/donor-provenance.json` | the 15 MTP tensors; `unsloth/Qwen3.8-27B-NVFP4` retains them in BF16 |
| `training/train.py` + method | hidden-state **forward-KL** distillation of an MTP head |
| provenance discipline | byte-identical base check, donor/data provenance — matches our audit checklist |

## What it is *not*

- Not a ternarisation of a new model → no information about Prism's training
  recipe.
- Not a 2B drafter (it is the 27B base + head).
- The head is Q8, not ternary.
- Not evidence of recipe identity.

## Attribution

Upstream: <https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP>
(Apache-2.0). The patch and scripts are mirrored for local reference only; any
reuse in our public repo must carry this attribution and the upstream license.
