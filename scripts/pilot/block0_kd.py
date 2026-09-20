"""Block-wise KD smoke (layer 0): can STE training recover a ternary block's output?

Emulates the Prism runtime: quantized linears store absorbed weights `R(W)` and
rotate their input activations at runtime; unquantized params stay FP. Target is
the FP block's own output on real hidden states (self-distillation), so no 55 GB
teacher residency is needed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HB))

from bonsai_forensics import rotation

BASE = HB / "artifacts/base27"
MANIFEST = HB / "artifacts/oracle/bonsai27/hadamard-manifest.json"
POOL = HB / "artifacts/pilot/pool_windows.npy"
OUT = HB / "artifacts/pilot/block0"
GROUP = 128


def fwht_torch(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    x = x.clone()
    h = 1
    while h < n:
        x = x.reshape(*x.shape[:-1], n // (2 * h), 2, h)
        a = x[..., 0, :].clone()
        b = x[..., 1, :].clone()
        x[..., 0, :] = a + b
        x[..., 1, :] = a - b
        x = x.reshape(*x.shape[:-3], n)
        h *= 2
    return x


def rotate_input(x: torch.Tensor, signs: list[np.ndarray]) -> torch.Tensor:
    g = len(signs[0])
    out = torch.empty_like(x)
    for k, s in enumerate(signs):
        blk = x[..., k * g:(k + 1) * g]
        st = torch.as_tensor(s, dtype=x.dtype, device=x.device)
        out[..., k * g:(k + 1) * g] = fwht_torch(blk * st) / math.sqrt(g)
    return out


class TernaryLinear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, signs: list[np.ndarray], group: int = GROUP):
        super().__init__()
        w = rotation.absorb_input(weight.float().numpy(), signs).astype(np.float32)
        self.master = torch.nn.Parameter(torch.from_numpy(w))
        self.signs = signs
        self.group = group

    def quantized(self) -> torch.Tensor:
        w = self.master
        out_f, in_f = w.shape
        g = w.reshape(out_f, -1, self.group)
        s = g.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        t = torch.clamp(torch.sign(g / s) * torch.floor((g / s).abs() + 0.5), -1.0, 1.0)
        wq = (t * s).reshape(out_f, in_f)
        return w + (wq - w).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(rotate_input(x, self.signs), self.quantized())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    cfg = AutoConfig.from_pretrained(BASE)
    tcfg = cfg.text_config
    index = json.loads((BASE / "model.safetensors.index.json").read_text())
    sign_sets = rotation.load_sign_manifest(MANIFEST)

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    pool = np.load(POOL)
    ids = torch.tensor(pool.reshape(-1)[: args.windows * args.seq], dtype=torch.long).reshape(args.windows, args.seq)

    def load_tensor(name):
        shard = index["weight_map"][name]
        with safe_open(BASE / shard, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    embed = load_tensor("model.language_model.embed_tokens.weight").to("cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        h0 = F.embedding(ids.to("cuda"), embed).float()
    del embed
    torch.cuda.empty_cache()
    print(f"[block] h0 {tuple(h0.shape)}", flush=True)

    prefix = f"model.language_model.layers.{args.layer}."
    state = {
        key[len(prefix):]: load_tensor(key)
        for key in index["weight_map"]
        if key.startswith(prefix)
    }
    fp_layer = Qwen3_5DecoderLayer(tcfg, args.layer).to("cuda", dtype=torch.bfloat16)
    missing, unexpected = fp_layer.load_state_dict(state, strict=False)
    print(f"[block] fp layer loaded missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    fp_layer.eval()
    cos = torch.ones(1, args.seq, 64, device="cuda", dtype=torch.bfloat16)
    sin = torch.zeros(1, args.seq, 64, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        target = fp_layer(h0.to(torch.bfloat16), (cos, sin)).float()
    print(f"[block] target {tuple(target.shape)}", flush=True)

    student = Qwen3_5DecoderLayer(tcfg, args.layer).to("cuda", dtype=torch.float32)
    student.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
    quantized = 0
    for module_name, module in list(student.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        suffix = module_name.split(".")[-1]
        if suffix not in ("in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"):
            continue
        parent_name, _, child = module_name.rpartition(".")
        parent = student.get_submodule(parent_name)
        signs = sign_sets[str(module.in_features)]
        setattr(parent, child, TernaryLinear(module.weight.detach().cpu(), signs))
        quantized += 1
    student.to("cuda")
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    masters = []
    for module in student.modules():
        if isinstance(module, TernaryLinear):
            module.master.requires_grad_(True)
            masters.append(module.master)
    print(f"[block] student: {quantized} ternary linears, {len(masters)} masters, "
          f"{sum(m.numel() for m in masters) / 1e6:.1f}M params", flush=True)

    def evaluate():
        with torch.no_grad():
            out = student(h0.to(torch.float32), (cos.float(), sin.float()))
            mse = F.mse_loss(out, target).item()
            rel = (out - target).norm().item() / target.norm().item()
        return mse, rel

    before = evaluate()
    print(f"[block] before: mse={before[0]:.6f} rel={before[1]:.4f}", flush=True)

    optimizer = torch.optim.Adam(masters, lr=args.lr)
    batches = torch.randperm(args.windows, generator=torch.Generator().manual_seed(0))
    history = []
    started = time.time()
    student.train()
    step = 0
    while step < args.steps:
        order = batches[step % len(batches): step % len(batches) + args.batch]
        if len(order) < args.batch:
            order = torch.cat([order, batches[: args.batch - len(order)]])
        x = h0[order].to(torch.float32)
        y = target[order]
        out = student(x, (cos.float(), sin.float()))
        if step == 0:
            print(f"[block] debug out {tuple(out.shape)} y {tuple(y.shape)}", flush=True)
        loss = F.mse_loss(out, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(masters, 1.0)
        optimizer.step()
        step += 1
        if step % 20 == 0 or step == 1:
            history.append({"step": step, "loss": round(loss.item(), 6)})
            print(f"[block] step {step} loss {loss.item():.6f}", flush=True)

    after = evaluate()
    print(f"[block] after: mse={after[0]:.6f} rel={after[1]:.4f} seconds={time.time() - started:.0f}", flush=True)
    (OUT / f"block{args.layer}-report.json").write_text(json.dumps({
        "layer": args.layer, "windows": args.windows, "seq": args.seq, "steps": args.steps,
        "before": {"mse": before[0], "rel": before[1]},
        "after": {"mse": after[0], "rel": after[1]},
        "history": history,
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
