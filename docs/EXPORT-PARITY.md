# Export parity — rmd student → PQ2_0 → dequant

`scripts/pilot/export_roundtrip.py` validates the rmd→PQ2_0 bridge on a real
trained checkpoint.  It rebuilds the wrapped model, un-absorbs each trained
master to its primal HF weight, recomputes the **deployed absorbed weight in the
export basis**, quantizes it with the independent `ternary_ste` codec, and
compares that to the artifact's dequantized tensor read back from the run
directory.  It also checks, on the same tensors, that `unabsorb_linear` inverts
the trainer's absorption (STE off: `RotatedLinear(x)` vs `F.linear(x, W)`).

Run it (outputs must be on a **disk** path — the per-tensor checkpoints are
multi-GB and `/tmp` is a RAM-backed tmpfs; the script refuses tmpfs unless
`--allow-tmpfs`):

```sh
PY=$HOME/.unsloth/studio/unsloth_studio/bin/python
PYTHONPATH=. HF_HUB_OFFLINE=1 $PY scripts/pilot/export_roundtrip.py \
    --checkpoint artifacts/rmd/wiki-conv-0.6B/student-best.pt \
    --out /home/$USER/rt/parity-0.6b/model.gguf
```

## Results (2026-09-24)

| checkpoint | tensors | cosine mean | cosine min | MAE mean | unabsorb fwd rel |
|---|---|---|---|---|---|
| `rmd/wiki-conv-0.6B` | 196 | 0.99999998 | 0.99999861 | 2.5e-09 | 3.5e-03 |
| `rmd/ladder-1.7B` | 196 | 0.99999998 | 0.99999951 | 3.2e-09 | 3.5e-03 |

Both clear the ProCreations-style bar (`cosine ≥ 0.9999`); the residual max
error (~0.2) is a handful of groups where a single trit flips.  The MAE of
~1e-9 confirms the bridge is bit-consistent, and the ~3e-3 forward error is
bfloat16 rounding.

## Why the reference is computed in the export basis

The comparison must account for two transforms the canonical exporter applies
that the trainer does not:

1. **Hidden-norm fold.** `run_quant` folds each `input_layernorm` /
   `post_attention_layernorm` γ into its q/k/v/gate/up consumers and stores the
   norms as ones (`norm_fold_map`).  The trained model keeps the real RMSNorm.
   Comparing an unfolded master against the folded artifact cannot reach parity.
2. **Rotation-axis convention.** The rmd trainer's `rotation_mode="residual"`
   rotates an output projection's **output** axis (`o_proj`, `down_proj`),
   while the released PQ2_0 convention (`rotation.axis="last"`) rotates every
   tensor's **input/last** axis.  The exporter re-absorbs in the released
   convention, so output projections are quantized in a different basis than
   they were trained in.

Both are expected and function-preserving: the artifact's recovered weight
(`W_hf = D @ R`, per `prism_loader.recover`) is the (folded) primal weight
regardless of which axis was used (the recovered function is unchanged; only
the quantization grid moves for output projections).  Parity is therefore
measured against the export's own basis, which is what the deployed artifact
actually carries.  A first version compared the artifact against
`ternary_ste(absorbed master)` directly and reported cosine ~0.60 — that was a
wrong-basis artifact, not an export bug.

## Note on scale

The 0.6 B and 1.7 B checkpoints round-trip on CPU.  A **4 B** round-trip is
memory-bound on this 30 GiB host: the 8.8 GB checkpoint plus the 8 GB bf16 model
plus the multi-GB checkpoint stream (which must go to a disk path, not tmpfs)
does not leave headroom.  The lazy `RotatedModelTensorSource` removes the
whole-state float64 materialization but not the model+checkpoint residency; run
4 B parity on a larger host or via `--out` on disk with no other load.
