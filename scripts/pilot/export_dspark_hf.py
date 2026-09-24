#!/usr/bin/env python
"""Write a trained `DSparkProdDraft` checkpoint as an HF dir for conversion.

Emits the `Qwen3DSparkModel` config schema + `model.safetensors` with the
template's parameter names, so `scripts/pilot/convert_dspark_draft.py` (the
`conversion.dspark` path) can produce a `dspark`-arch GGUF.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HB))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    ap.add_argument("--target-layers", default="5,19,33,47,61")
    ap.add_argument("--num-target-layers", type=int, default=64)
    args = ap.parse_args(argv)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    c = ck["config"]
    state = ck["model"]
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    target_layers = [int(x) for x in args.target_layers.split(",") if x]
    hf_config = {
        "architectures": ["Qwen3DSparkModel"],
        "model_type": "qwen3",
        "hidden_size": int(c["hidden"]),
        "num_hidden_layers": int(c["layers_count"]),
        "num_attention_heads": int(c["heads"]),
        "num_key_value_heads": int(c["kv_heads"]),
        "head_dim": int(c["head_dim"]),
        "intermediate_size": int(c["intermediate"]),
        "vocab_size": int(c["vocab"]),
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 40960,
        "tie_word_embeddings": False,
        "markov_rank": int(c["markov_rank"]),
        "markov_head_type": "vanilla",
        "block_size": int(c["block_size"]),
        "mask_token_id": int(c["vocab"]) - 1,
        "target_layer_ids": target_layers,
        "num_target_layers": int(args.num_target_layers),
        "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    save_file({k: v.to(dtype).contiguous() for k, v in state.items()},
              str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps(hf_config, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[export] wrote {out}/model.safetensors ({len(state)} tensors) and config.json")
    print("[export] layers:", hf_config["num_hidden_layers"], "target_layers:", target_layers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
