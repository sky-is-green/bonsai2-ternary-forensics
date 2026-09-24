#!/usr/bin/env python
"""Forward-parity check: our PyTorch DSpark trunk vs the fork's C++ graph.

Milestone 2 of `docs/DSPARK-PATH1-PLAN.md`.  Builds a tiny `dspark`-arch GGUF
from a `DSparkProdDraft`'s own weights, runs our forward on fixed tap features /
draft tokens / positions, writes the fork's `ref.bin` format, then runs
`tests/test-dspark-forward --tier2` and parses the diff.

`ref.bin` (little-endian): int32 n_ctx_rows, n_embd_cap, block_size, vocab_size;
int32[n_ctx_rows] ctx_pos; f32[n_ctx_rows*n_embd_cap] ctx_feat;
int32[block_size] draft_tokens; int32[block_size] draft_pos;
f32[block_size*vocab_size] logits.
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HB))

from bonsai_forensics.dspark_prod import DSparkProdConfig, DSparkProdDraft  # noqa: E402

TINY = DSparkProdConfig(
    hidden_size=64, vocab_size=256, num_layers=2, num_heads=2, num_kv_heads=1,
    head_dim=32, intermediate_size=96, n_capture=2, markov_rank=0,
    block_size=4, mask_token_id=255, rms_norm_eps=1e-6, rope_theta=10000.0,
    confidence_head=False, tie_word_embeddings=True,
)
TARGET_LAYERS = [1, 16]
N_CTX, CTX_POS = 6, list(range(6))
DRAFT_TOKENS = [10, 255, 255, 255]
DRAFT_POS = [6, 7, 8, 9]


def _write_gguf(path: Path, model: DSparkProdDraft) -> None:
    from gguf import GGUFWriter, GGUFValueType

    c = model.config
    w = GGUFWriter(str(path), "dspark")
    w.add_name("dspark-parity-tiny")
    w.add_file_type(0)  # all f32
    w.add_block_count(c.num_layers)
    w.add_context_length(512)
    w.add_embedding_length(c.hidden_size)
    w.add_feed_forward_length(c.intermediate_size)
    w.add_head_count(c.num_heads)
    w.add_head_count_kv(c.num_kv_heads)
    w.add_layer_norm_rms_eps(c.rms_norm_eps)
    w.add_rope_freq_base(c.rope_theta)
    w.add_uint32("dspark.dspark.block_size", c.block_size)
    w.add_uint32("dspark.dspark.mask_token_id", c.mask_token_id)
    w.add_key_value("dspark.dspark.target_layers", TARGET_LAYERS,
                    GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT32)
    w.add_tokenizer_model("none")
    w.add_vocab_size(c.vocab_size)

    state = model.state_dict()
    for name, tensor in state.items():
        arr = tensor.detach().to(torch.float32).cpu().numpy()
        gguf_name = None
        if name == "embed_tokens.weight":
            gguf_name = "token_embd.weight"
        elif name == "norm.weight":
            gguf_name = "output_norm.weight"
        elif name == "lm_head.weight":
            continue  # tied -> runtime duplicates token_embd
        elif name in ("fc.weight", "hidden_norm.weight"):
            gguf_name = "dspark." + name
        elif name.startswith("layers."):
            _, i, suffix = name.split(".", 2)
            gguf_name = f"blk.{i}." + {
                "input_layernorm.weight": "attn_norm.weight",
                "post_attention_layernorm.weight": "ffn_norm.weight",
                "self_attn.q_proj.weight": "attn_q.weight",
                "self_attn.k_proj.weight": "attn_k.weight",
                "self_attn.v_proj.weight": "attn_v.weight",
                "self_attn.o_proj.weight": "attn_output.weight",
                "self_attn.q_norm.weight": "attn_q_norm.weight",
                "self_attn.k_norm.weight": "attn_k_norm.weight",
                "mlp.gate_proj.weight": "ffn_gate.weight",
                "mlp.up_proj.weight": "ffn_up.weight",
                "mlp.down_proj.weight": "ffn_down.weight",
            }[suffix]
        if gguf_name is None:
            raise ValueError(f"unmapped parameter {name}")
        w.add_tensor(gguf_name, np.ascontiguousarray(arr))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _write_ref(path: Path, model: DSparkProdDraft) -> np.ndarray:
    c = model.config
    rng = np.random.default_rng(0)
    ctx_feat = rng.standard_normal((N_CTX, c.n_embd_cap)).astype(np.float32) * 0.5
    tokens = torch.tensor([DRAFT_TOKENS], dtype=torch.long)
    taps = torch.from_numpy(ctx_feat).unsqueeze(0)
    pos = torch.tensor([CTX_POS + DRAFT_POS], dtype=torch.long)
    with torch.no_grad():
        logits, _ = model(tokens, taps, position_ids=pos)
    ref = logits[0].to(torch.float32).numpy()
    with path.open("wb") as f:
        f.write(struct.pack("<4i", N_CTX, c.n_embd_cap, c.block_size, c.vocab_size))
        f.write(np.asarray(CTX_POS, dtype="<i4").tobytes())
        f.write(ctx_feat.tobytes())
        f.write(np.asarray(DRAFT_TOKENS, dtype="<i4").tobytes())
        f.write(np.asarray(DRAFT_POS, dtype="<i4").tobytes())
        f.write(ref.tobytes())
    return ref


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llama-cpp", default=str(Path.home() / "llama.cpp"))
    ap.add_argument("--work-dir", default="/tmp/opencode/dspark-parity")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args(argv)

    llama = Path(args.llama_cpp)
    harness = llama / "build" / "bin" / "test-dspark-forward"
    if not harness.is_file():
        raise SystemExit(f"harness not built: {harness}")
    sys.path.insert(0, str(llama / "gguf-py"))

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    model = DSparkProdDraft(TINY).eval()

    gguf_path = work / "tiny-dspark-f32.gguf"
    ref_path = work / "ref.bin"
    _write_gguf(gguf_path, model)
    _write_ref(ref_path, model)
    print(f"[parity] wrote {gguf_path} and {ref_path}", flush=True)

    proc = subprocess.run([str(harness), str(gguf_path), "--tier2", str(ref_path)],
                          capture_output=True, text=True,
                          env={**os.environ, "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES", "1")})
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        return proc.returncode

    summary = {}
    for line in proc.stdout.splitlines():
        if line.startswith("mean_abs_diff="):
            for kv in line.split():
                k, _, v = kv.partition("=")
                summary[k] = float(v)
    max_abs = summary.get("max_abs_diff", float("inf"))
    print(f"[parity] max_abs_diff={max_abs:.3e}")
    ok = max_abs < 1e-2
    print(f"[parity] {'PASS' if ok else 'FAIL'} (threshold 1e-2)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
