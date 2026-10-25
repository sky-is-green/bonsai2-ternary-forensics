# Prior art around Bonsai 2 — what's settled, what's open

Surveyed 2026-09-24.  This is the map that re-scopes our contribution: the
community has closed **format, interop, drafters and post-hoc refinement**; the
**training recipe** is the only unclaimed ground — and it is not locally runnable
at 27B.

## The ecosystem

| repo | base | contribution | notable |
|---|---|---|---|
| `prism-ml/Ternary-Bonsai-2-27B-gguf` | Qwen3.8-27B | the target: PQ2_0 ternary, ~7.2 GB | 852 tensors, g128 |
| `PrismML-Eng/llama.cpp` | — | type-142 runtime + MTP/DFlash/DSpark spec decode | fork present locally |
| `ProCreations/…-MTP` | Bonsai 2 (byte-identical) | trained MTP head (Q8), full training code, parity reports | 851 base tensors unchanged |
| `sudoingx/…-PTQ1_0-MTP-GGUF` | Bonsai 2 **PTQ1_0** (dense 1.75-bpw packing of the same ternary weights as PQ2_0) | MTP graft as block 64 | verification-cost wall; kernel PR #218 |
| `decent-jawfish/bonsai-2-27b-mtp` | Bonsai 2 PQ2_0 | MTP graft + **`graft_mtp.py`** + 8-line embedding patch | +44% decode, holds under concurrency |
| `BoldingBuilds/…-Abliterated-PQ2_0-MTP-GGUF` | Bonsai 2 PQ2_0 | **abliterated base + re-export to PQ2_0** | proves base mod + ternary re-export |
| `ProCreations/…-DFlash2` | Bonsai 2 (unchanged) | **separate DFlash2 draft** (Q8, 2.06 GB, 5 layers) | donor `z-lab/Qwen3.8-27B-DFlash2` |
| `ProCreations/bonsai-2-27b-gsq-rco-gguf` | Bonsai 2 PTQ1_0 | **GSQ + RCO** post-hoc refinement | WikiText-2 PPL 8.669 → 8.163 (−5.84%) |
| `OsaurusAI/Bonsai-2-27B-Ternary-JANG` | Prism MLX ternary | lossless repack (JANG affine) | logit parity vs reference PASS |
| `deepseek-ai/dspark_qwen3_4b_block7` | Qwen3-4B | DSpark template (Markov head) | not for Bonsai |

## What is settled

- **Format.** PQ2_0 (type 142), g128 ternary `{−s,0,+s}`; embedding, untied head,
  all attention/GDN/MLP matrices ternary; norms/recurrent-state float32; vision
  6-bit affine.
- **Packings.** Bonsai 2's PTQ1_0 (type 143, 1.75 bpw) and PQ2_0 (type 142,
  2.125 bpw) are two containers for the **same ternary weights**.  The full
  402-tensor parity check is in [`PACKING-PARITY.md`](PACKING-PARITY.md);
  throughput, not quality, is the reason to choose between them.
- **Basis.** Blockwise Hadamard (block 1024, explicit signs), **all 402 tensors
  rotate the last/ne0 axis** (`prism_loader.recover`: `W_hf = D @ R`).
- **Embedding.** Stored Hadamard-latent; every runtime read must inverse-rotate
  it (three independent patches confirm; ~8-line fix).
- **Interop.** Multiple grafted heads + a JANG repack load with logit parity.
- **Post-hoc quality.** GSQ (teacher-guided code refinement, Gumbel) + RCO
  (rate-constrained bit allocation) improves released Bonsai 2 by ~5.8% PPL.

## Key numbers

| model | WikiText-2 PPL | ratio vs BF16 |
|---|---:|---:|
| Qwen3.8-27B BF16 | 6.5137 | 1.000 |
| Bonsai 2 PTQ1_0 (dense ternary packing) | 8.6690 | 1.331 |
| Bonsai 2 + GSQ/RCO | 8.1627 | 1.253 |
| our Qwen3-1.7B ternary | — | 1.573 |

Prism's 94.6–98.2% is a **benchmark** retention; the measured **PPL** ratio for
Bonsai 2's dense ternary packing is 1.331.  Do not mix the two rulers.

The PTQ1_0/PQ2_0 files are not a 1-bit-vs-2-bit quality comparison: the
ternary codes and scales are identical across all 402 ternary tensors
([`PACKING-PARITY.md`](PACKING-PARITY.md)).  The older 1-bit Bonsai 27B
(`Q1_0`, Qwen3.6 base) is a different model and is not covered by this table.

## What is open

1. **The training recipe** — nobody retrains the ternary weights.  Ours matches
   the *class* but is 25+ points short on retention; a 27B run is infeasible
   locally (student + teacher ≈ 111 GB).
2. **Ternary drafters** — every published draft is Q8.  See
   [`DSPARK-TRACK.md`](DSPARK-TRACK.md).
3. **Post-hoc refinement of *our* models** — GSQ/RCO has only been applied to
   Bonsai 2; a teacher-guided refinement of our 2B/4B is cheap and local.
4. **A same-ruler comparison** — trained-ternary vs PTQ at matched size, one
   metric.

## Attribution

All third-party repos above are Apache-2.0 (except `z-lab`/`deepseek-ai`, which
are their own licenses).  Reference bundles are mirrored locally under
`artifacts/reference/` (gitignored) and are **not** redistributed.  Reuse in
this public repo must carry the upstream attribution and license.
