"""Export trained correction branches as a llama.cpp LoRA adapter GGUF.

For ``--branch-target attn_out`` checkpoints, each branch becomes a LoRA on
the attention output projection, matching the TAARDIS adapter convention:

    blk.N.attn_output.weight.lora_a   (gguf dims [in, rank]  <- down.weight.T)
    blk.N.attn_output.weight.lora_b   (gguf dims [rank, out] <- up.weight.T)

and the metadata:

    general.type = adapter
    general.architecture = <arch>
    adapter.type = lora            (--taardis writes taardis-lora)
    adapter.lora.alpha = 0.0       (scale 1.0, as in the reference adapters)

The ``moe_out`` placement has no linear tensor to attach to and is refused:
that placement needs a runtime op rather than a LoRA.

Usage:
  PYTHONPATH=<fork>/gguf-py python export_branches_lora.py \
      --load $MOE_ARTIFACTS/olmoe/olmoe-corr-r512-g128-attnout-step8000.pt \
      --arch olmoe --target attn_out --out branches-attnout.lora.gguf
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter


def layer_index(key: str) -> int:
    return int(re.search(r"layers\.(\d+)\.", key).group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="olmoe")
    ap.add_argument("--target", choices=["attn_out", "moe_out"], default="attn_out")
    ap.add_argument("--taardis", action="store_true",
                    help="declare taardis-lora instead of plain lora")
    ap.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    args = ap.parse_args()

    if args.target == "moe_out":
        raise SystemExit("moe_out branches are not LoRA-mappable; export is not possible")

    sd = torch.load(args.load, map_location="cpu")
    downs = {}
    ups = {}
    for key, tensor in sd.items():
        if key.endswith(".branch.down.weight"):
            downs[layer_index(key)] = tensor.float().numpy()
        elif key.endswith(".branch.up.weight"):
            ups[layer_index(key)] = tensor.float().numpy()
    layers = sorted(set(downs) & set(ups))
    if not layers:
        raise SystemExit(f"no branch tensors found in {args.load}")
    rank = downs[layers[0]].shape[0]
    print(f"exporting {len(layers)} branches, rank {rank}")

    w = GGUFWriter(args.out, arch=args.arch)
    w.add_type("adapter")
    w.add_string("adapter.type", "taardis-lora" if args.taardis else "lora")
    w.add_float32("adapter.lora.alpha", 0.0)

    data_dtype = np.float16 if args.dtype == "f16" else np.float32
    for i in layers:
        down = downs[i]        # [rank, in]
        up = ups[i]            # [out, rank]
        # gguf dims are reversed relative to PyTorch; the reference adapters
        # store lora_a as [in, rank] and lora_b as [rank, out].
        w.add_tensor(f"blk.{i}.attn_output.weight.lora_a",
                     np.ascontiguousarray(down.T).astype(data_dtype))
        w.add_tensor(f"blk.{i}.attn_output.weight.lora_b",
                     np.ascontiguousarray(up.T).astype(data_dtype))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    out = Path(args.out)
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
