# Bonsai Runtime (T29) — released Bonsai-2 27B `PQ2_0` on the Prism ROCm fork

Plan-B end-state: run Prism's released ternary artifact as-is and evaluate it
with our harness. This is the pragmatic route to the user's goal — the quality
lives in their trained weights (Gate 3), so no local training is required.

## Artifact

| Item | Path |
|---|---|
| Model | `artifacts/oracle/bonsai27/Ternary-Bonsai-2-27B-PQ2_0.gguf` (7.2 GB, sha `3907dc16…`) |
| Fork binary | `artifacts/oracle/prism-fork/bin/llama-prism-b10709-9a9394a/llama-server` |
| Driver | `scripts/eval_llama_server.py` |

## Serve / evaluate

```sh
HIP_VISIBLE_DEVICES=1 python scripts/eval_llama_server.py smoke \
    --server-bin /path/to/prism-fork/llama-server \
    --gguf /path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf \
    --ngl 99 --ctx 4096 --no-thinking \
    --out artifacts/eval/bonsai27-smoke.json
```

- `--no-thinking` passes `--chat-template-kwargs '{"enable_thinking": false}'` to
  `llama-server`; without it the reasoning template returns empty visible content.
- `HIP_VISIBLE_DEVICES=1` keeps the desktop GPU0 free; the 27B `PQ2_0` fits one
  RX 7900 XT at `-ngl 99`, so both cards are not needed.
- The original run used a private harness driver; this repository ships the
  standalone replacement (`scripts/eval_llama_server.py`) with no harness
  dependency.

## Results (2026-09-20)

PPL under the Gate-2 protocol (tinyshakespeare, ctx 512, 8 chunks): **18.5851**.

Harness evaluation, `--max-convs 10`, 124 turns compared:

| Metric | hive | FIFO |
|---|---|---|
| answer recall | 72.6 | 77.4 |
| avg fact hit ratio | 0.731 | 0.789 |
| avg context fidelity | **0.306** | 0.205 |

- 5/5 smoke prompts PASS (coherent: `Paris`, `51`, `ternary.`); ~0.5–0.9 s/reply.
- `fidelity_hive_gt_fifo_ratio` 69.4%, `hive_ge_fifo_ratio` 89.0% — the memory
  layer improves context fidelity over FIFO while trailing slightly on raw answer
  recall. `hive_only` 4, `fifo_only` 10, `both_sufficient` 77.

## Status

T29 runtime + eval verified. Remaining (optional): LoRA adaptation around the
released weights, and a 2-card throughput pass if serving latency matters.
