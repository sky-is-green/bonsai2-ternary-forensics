"""Sequential block-wise KD for 27B: train every layer against its FP self, carrying the student stream.

For layer k: FP block k produces the target from the current hidden stream H_k;
the student block (absorbed ternary masters, runtime-style input rotation) is
trained by STE to match it; then the student block produces H_{k+1}. Masters are
saved per layer for the packing step. No 55 GB residency, no external teacher.
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
OUT = HB / "artifacts/pilot/blockwise"
GROUP = 128
QUANT_SUFFIXES = (
    "in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def log(msg):
    print(f"[bw] {msg}", flush=True)


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


def make_student(fp_layer, sign_sets):
    import copy

    student = copy.deepcopy(fp_layer).float()
    count = 0
    for module_name, module in list(student.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        if module_name.split(".")[-1] not in QUANT_SUFFIXES:
            continue
        parent_name, _, child = module_name.rpartition(".")
        parent = student.get_submodule(parent_name)
        setattr(parent, child, TernaryLinear(module.weight.detach().cpu(), sign_sets[str(module.in_features)]))
        count += 1
    student.to("cuda")
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    masters = []
    for module in student.modules():
        if isinstance(module, TernaryLinear):
            module.master.requires_grad_(True)
            masters.append(module.master)
    return student, masters, count


def train_block(student, masters, h, target, cos, sin, steps, batch, lr):
    optimizer = torch.optim.Adam(masters, lr=lr)
    n = h.shape[0]
    generator = torch.Generator().manual_seed(0)
    student.train()
    for step in range(steps):
        order = torch.randint(0, n, (batch,), generator=generator)
        x = h[order].float()
        y = target[order]
        out = student(x, (cos[order].float(), sin[order].float()))
        loss = F.mse_loss(out, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(masters, 1.0)
        optimizer.step()
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            log(f"  step {step} loss {loss.item():.6f}")
    student.eval()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-layer", type=int, default=0)
    ap.add_argument("--end-layer", type=int, default=64)
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

    cfg = AutoConfig.from_pretrained(BASE)
    tcfg = cfg.text_config
    index = json.loads((BASE / "model.safetensors.index.json").read_text())
    sign_sets = rotation.load_sign_manifest(MANIFEST)

    def load_tensor(name):
        with safe_open(BASE / index["weight_map"][name], framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    pool = np.load(POOL)
    ids = torch.tensor(pool.reshape(-1)[: args.windows * args.seq], dtype=torch.long).reshape(args.windows, args.seq)
    rotary = Qwen3_5TextRotaryEmbedding(tcfg).to("cuda")
    position_ids = torch.arange(args.seq, device="cuda").unsqueeze(0).expand(args.windows, -1)

    embed = load_tensor("model.language_model.embed_tokens.weight").to("cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        h = F.embedding(ids.to("cuda"), embed).float()
    del embed
    torch.cuda.empty_cache()
    log(f"h0 {tuple(h.shape)}")

    report = {"layers": [], "windows": args.windows, "seq": args.seq, "steps": args.steps}
    started = time.time()
    for layer in range(args.start_layer, args.end_layer):
        prefix = f"model.language_model.layers.{layer}."
        state = {key[len(prefix):]: load_tensor(key) for key in index["weight_map"] if key.startswith(prefix)}
        fp_layer = Qwen3_5DecoderLayer(tcfg, layer).to("cuda", dtype=torch.bfloat16)
        fp_layer.load_state_dict(state, strict=False)
        fp_layer.eval()
        with torch.no_grad():
            cos, sin = rotary(h.to(torch.bfloat16), position_ids)
            target = fp_layer(h.to(torch.bfloat16), (cos, sin)).float()

        student, masters, count = make_student(fp_layer, sign_sets)
        with torch.no_grad():
            before = student(h.float(), (cos.float(), sin.float()))
            rel_before = (before - target).norm().item() / target.norm().item()
        train_block(student, masters, h, target, cos, sin, args.steps, args.batch, args.lr)
        with torch.no_grad():
            after = student(h.float(), (cos.float(), sin.float()))
            rel_after = (after - target).norm().item() / target.norm().item()
            h_next = after.to(torch.bfloat16).float()
        log(f"layer {layer}: {count} ternary, rel {rel_before:.4f} -> {rel_after:.4f}, "
            f"{time.time() - started:.0f}s elapsed")
        report["layers"].append({
            "layer": layer, "ternary_linears": count,
            "rel_before": rel_before, "rel_after": rel_after,
            "seconds": time.time() - started,
        })
        if args.save:
            payload = {name: module.master.detach().to(torch.bfloat16).cpu() for name, module in student.named_modules()
                       if isinstance(module, TernaryLinear)}
            torch.save(payload, OUT / f"layer{layer:02d}-masters.pt")
        (OUT / "blockwise-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        h = h_next
        del fp_layer, student, masters, target
        torch.cuda.empty_cache()

    report["seconds"] = time.time() - started
    (OUT / "blockwise-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"done: {report['seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
