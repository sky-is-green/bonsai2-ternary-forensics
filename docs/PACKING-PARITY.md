# PTQ1_0 vs PQ2_0 — packing parity on Bonsai 2 27B

**Status: measured 2026-10-25.** This settles what “Q1 is close to Q2” means for
Bonsai 2. It is a **packing** comparison, not a separate 1-bit quality run.

## Artifacts

| pack | file | size | bytes/weight | sha256 |
|---|---:|---:|---:|---|
| PTQ1_0 | `Ternary-Bonsai-2-27B-PTQ1_0.gguf` | 5,946,648,928 B | 1.75 bpw | `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3` |
| PQ2_0 | `Ternary-Bonsai-2-27B-PQ2_0.gguf` | 7,206,168,928 B | 2.125 bpw | `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1` |

Both files have 851 tensors and the same tensor names and shapes. The 402
ternary tensors are stored as ggml type 143 (`PTQ1_0`) and type 142 (`PQ2_0`);
the non-ternary tensors are unchanged.

## Result

`scripts/packing_parity.py` decoded every ternary tensor from both files and
compared the ternary codes and fp16 group scales:

```text
checked 402 tensors / 4396032 rows; bad_codes=0; worst_scale_diff=0.000e+00
```

Both packings therefore carry the **same ternary model**. The only differences
are the on-disk container and the unpacking cost:

- PTQ1_0 stores the trits densely: 28 bytes per 128 weights.
- PQ2_0 stores each trit in a 2-bit slot: 34 bytes per 128 weights.
- PTQ1_0 is 1.26 GB smaller.
- PQ2_0 can be faster to unpack on some GPUs; PTQ1_0 can be faster when memory
  bandwidth is the limit. That is a runtime measurement, not a quality
  difference.

This is why the published model-card throughput swaps between the two packs by
hardware class (for example, PTQ1_0 faster on an RTX 4090; PQ2_0 faster on an
RTX 5090). It does **not** imply a Q1-vs-Q2 quality gap.

## Reproduce

```sh
python3 scripts/packing_parity.py \
  --ptq1 /path/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --pq2  /path/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --all-tensors --rows 0
```

The script is CPU-only and decodes the PTQ1_0 blocks with the same rule as the
runtime's `dequantize_row_ptq1_0`; the PQ2_0 decoder mirrors
`bonsai_forensics/pq2_0.py`. Use `--rows 8` for a quick sample.

## What this does *not* claim

- It does not measure runtime throughput. The local 7900 XT comparison is still
  unmeasured.
- It does not make the **older 1-bit Bonsai 27B** (`prism-ml/Bonsai-27B-gguf`,
  Qwen3.6-based, `Q1_0`) equivalent to Bonsai 2. That is a different base model
  and a different weight set; the older published retention numbers in
  `RETENTION-VS-SCALE.md` are not a Bonsai 2 PTQ1_0-vs-PQ2_0 result.
- It does not replace a same-ruler logit/PPL check. It proves the decoded
  ternary weights are identical; runtime arithmetic can still differ by the
  kernel path.

## Consequence for our work

For a trained DSpark or other ternary sidecar, choose the packing by
**deployment**, not by quality:

- PTQ1_0 for the smaller file and memory-bound cards.
- PQ2_0 where the unpacking path is faster on the target GPU.

Both should be exported from the same trained ternary weights and measured on
the actual engine.
