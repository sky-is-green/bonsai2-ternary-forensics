"""Export trained correction branches as a llama.cpp LoRA adapter GGUF.

For ``--branch-target attn_out`` checkpoints, each branch becomes a LoRA on
the attention output projection, matching the TAARDIS adapter convention:

    blk.N.attn_output.weight.lora_a   (gguf dims [in, rank]  <- down.weight [rank, in])
    blk.N.attn_output.weight.lora_b   (gguf dims [rank, out] <- up.weight [out, rank])

and the metadata:

    general.type = adapter
    general.architecture = <arch>
    adapter.type = lora            (--taardis writes taardis-lora)
    adapter.lora.alpha = 0.0       (scale 1.0, as in the reference adapters)

The exported factors are ternarised with the same deployed quantizer used in
training/eval (``--deploy-quant`` / ``--branch-quant``, mirroring
``CorrectionBranch._weights``), so the adapter reproduces the eval numbers.

The ``moe_out`` placement has no linear tensor to attach to and is refused:
that placement needs a runtime op rather than a LoRA.

Usage:
  PYTHONPATH=<fork>/gguf-py python export_branches_lora.py \
      --load $MOE_ARTIFACTS/olmoe/olmoe-corr-r512-g128-attnoutlloyd-step8000.pt \
      --arch olmoe --target attn_out --out branches-attnout.lora.gguf
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moe_proxy import ternary_absmean, ternary_lloyd  # noqa: E402


def layer_index(key: str) -> int:
    return int(re.search(r"layers\.(\d+)\.", key).group(1))


def deploy_weights(down: torch.Tensor, up: torch.Tensor, quant: str,
                   kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce ``CorrectionBranch._weights`` for the deployed factors."""
    if quant == "none":
        return down, up
    fn = ternary_lloyd if kind == "lloyd" else ternary_absmean
    if quant == "g128":
        return fn(down, 128), fn(up, 128)
    # rank-component scales, folded from the down factor into the up factor
    s = down.abs().mean(dim=1).clamp_min(1e-8)          # [rank]
    qd = torch.clamp(torch.round(down / s[:, None]), -1, 1)
    t = up.abs().mean(dim=0).clamp_min(1e-8)            # [rank]
    qu = torch.clamp(torch.round(up / t[None, :]), -1, 1)
    return qd, qu * (s * t)[None, :]


def load_base_routers(base_model: str) -> dict:
    """Load ``model.layers.N.mlp.gate.weight`` from an HF safetensors dir or .pt file."""
    path = Path(base_model)
    out = {}
    if path.is_dir():
        from safetensors import safe_open
        for f in sorted(path.glob("*.safetensors")):
            with safe_open(f, framework="pt") as sf:
                for k in sf.keys():
                    if k.endswith(".mlp.gate.weight"):
                        out[k] = sf.get_tensor(k).float()
    else:
        sd = torch.load(path, map_location="cpu")
        out = {k: v.float() for k, v in sd.items() if k.endswith(".mlp.gate.weight")}
    if not out:
        raise SystemExit(f"no router weights found under {base_model}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="olmoe")
    ap.add_argument("--target", choices=["attn_out", "moe_out"], default="attn_out")
    ap.add_argument("--taardis", action="store_true",
                    help="declare taardis-lora instead of plain lora")
    ap.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    ap.add_argument("--deploy-quant", choices=["lloyd", "absmean"], default="lloyd",
                    help="scale rule for ternarising the factors (default: the deployed rule)")
    ap.add_argument("--branch-quant", choices=["g128", "rank", "none"], default="g128",
                    help="deployed branch format the checkpoint was trained with "
                         "('none' exports the raw fp32 masters)")
    ap.add_argument("--routers", action="store_true",
                    help="also export the trained router deltas as exact rank<=64 LoRA "
                         "pairs on ffn_gate_inp (needs --base-model)")
    ap.add_argument("--base-model", default="",
                    help="HF dir (safetensors) holding the frozen-body routers the "
                         "checkpoint was trained against")
    args = ap.parse_args()

    if args.target == "moe_out":
        raise SystemExit("moe_out branches are not LoRA-mappable; export is not possible")
    if args.routers and not args.base_model:
        raise SystemExit("--routers needs --base-model")

    sd = torch.load(args.load, map_location="cpu")
    sd = {k.replace(".doctor.", ".branch."): v for k, v in sd.items()}
    downs = {}
    ups = {}
    for key, tensor in sd.items():
        if key.endswith(".branch.down.weight"):
            downs[layer_index(key)] = tensor
        elif key.endswith(".branch.up.weight"):
            ups[layer_index(key)] = tensor
    layers = sorted(set(downs) & set(ups))
    if not layers:
        raise SystemExit(f"no branch tensors found in {args.load}")
    rank = downs[layers[0]].shape[0]
    print(f"exporting {len(layers)} branches, rank {rank}, "
          f"deploy-quant {args.deploy_quant}/{args.branch_quant}")

    w = GGUFWriter(args.out, arch=args.arch)
    w.add_type("adapter")
    w.add_string("adapter.type", "taardis-lora" if args.taardis else "lora")
    w.add_float32("adapter.lora.alpha", 0.0)

    data_dtype = np.float16 if args.dtype == "f16" else np.float32
    for i in layers:
        down, up = deploy_weights(downs[i], ups[i], args.branch_quant, args.deploy_quant)
        down = down.float().numpy()    # [rank, in]
        up = up.float().numpy()        # [out, rank]
        # gguf-py reverses dims on write: passing [rank, in] / [out, rank]
        # stores ne [in, rank] / [rank, out], matching the reference adapters.
        w.add_tensor(f"blk.{i}.attn_output.weight.lora_a",
                     np.ascontiguousarray(down).astype(data_dtype))
        w.add_tensor(f"blk.{i}.attn_output.weight.lora_b",
                     np.ascontiguousarray(up).astype(data_dtype))

    if args.routers:
        base = load_base_routers(args.base_model)
        n_routers = 0
        for key, trained in sd.items():
            if not key.endswith(".mlp.gate.weight") or key not in base:
                continue
            delta = trained.float() - base[key]          # [out, in]
            u, s, vh = torch.linalg.svd(delta, full_matrices=False)
            a = vh.numpy()                               # [rank, in]
            b = (u * s[None, :]).numpy()                 # [out, rank]
            i = layer_index(key)
            w.add_tensor(f"blk.{i}.ffn_gate_inp.weight.lora_a",
                         np.ascontiguousarray(a).astype(data_dtype))
            w.add_tensor(f"blk.{i}.ffn_gate_inp.weight.lora_b",
                         np.ascontiguousarray(b).astype(data_dtype))
            n_routers += 1
        print(f"exported {n_routers} router deltas (rank {a.shape[0]})")

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    out = Path(args.out)
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
