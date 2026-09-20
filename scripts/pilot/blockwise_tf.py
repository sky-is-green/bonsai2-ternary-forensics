"""Teacher-forced block-wise KD for 27B, sharded across two GPUs in one process.

Fixes the compounding bug in `blockwise_kd.py`: instead of carrying the *student*
stream, run the full-precision stack once and save its hidden stream h_0..h_64
(`--phase prep`), then train each layer as an independent `h_k -> h_{k+1}`
distillation (`--phase train`). Independent blocks are sharded across both ROCm
devices within a single process (the sanctioned model-parallel shape).

Quantized linears store absorbed weights `W Rᵀ` and rotate their input at
runtime, matching the Prism/llama.cpp convention. Masters are saved per layer
for `pack_blockwise.py`.
"""

from __future__ import annotations

import argparse
import copy
import gc
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
OUT = HB / "artifacts/pilot/blockwise_tf"
TARGETS = OUT / "targets"
GROUP = 128
N_LAYERS = 64
QUANT_SUFFIXES = (
    "in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def log(msg):
    print(f"[tf] {msg}", flush=True)


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
        return (w + ((t * s).reshape(out_f, in_f) - w).detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(rotate_input(x, self.signs), self.quantized())


def load_index():
    return json.loads((BASE / "model.safetensors.index.json").read_text())


def load_tensor(index, name, framework="pt"):
    from safetensors import safe_open
    with safe_open(BASE / index["weight_map"][name], framework=framework, device="cpu") as handle:
        return handle.get_tensor(name)


def get_rotary(tcfg, h, position_ids):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    rotary = Qwen3_5TextRotaryEmbedding(tcfg).to(h.device)
    return rotary(h.to(torch.bfloat16), position_ids)


def build_student(index, tcfg, layer, sign_sets, device):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
    prefix = f"model.language_model.layers.{layer}."
    state = {key[len(prefix):]: load_tensor(index, key) for key in index["weight_map"] if key.startswith(prefix)}
    fp = Qwen3_5DecoderLayer(tcfg, layer)
    fp.load_state_dict(state, strict=False)
    student = copy.deepcopy(fp).float()
    del fp, state
    gc.collect()
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
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    masters = []
    for module in student.modules():
        if isinstance(module, TernaryLinear):
            module.master.requires_grad_(True)
            masters.append(module.master)
    student.to(device)
    return student, masters, count


def phase_prep(args, index, tcfg):
    TARGETS.mkdir(parents=True, exist_ok=True)
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    pool = np.load(POOL)
    ids = torch.tensor(pool[: args.windows, : args.seq], dtype=torch.long)
    dev = f"cuda:{args.prep_device}"
    embed = load_tensor(index, "model.language_model.embed_tokens.weight").to(dev, dtype=torch.bfloat16)
    with torch.no_grad():
        h = F.embedding(ids.to(dev), embed).float()
    del embed
    torch.cuda.empty_cache()
    position_ids = torch.arange(args.seq, device=dev).unsqueeze(0).expand(args.windows, -1)
    torch.save(h.to(torch.bfloat16).cpu(), TARGETS / "h00.pt")
    log(f"h0 {tuple(h.shape)}")
    started = time.time()
    for layer in range(N_LAYERS):
        prefix = f"model.language_model.layers.{layer}."
        state = {key[len(prefix):]: load_tensor(index, key) for key in index["weight_map"] if key.startswith(prefix)}
        fp = Qwen3_5DecoderLayer(tcfg, layer).to(dev, dtype=torch.bfloat16)
        fp.load_state_dict(state, strict=False)
        fp.eval()
        with torch.no_grad():
            cos, sin = get_rotary(tcfg, h, position_ids)
            h = fp(h.to(torch.bfloat16), (cos, sin)).float()
        torch.save(h.to(torch.bfloat16).cpu(), TARGETS / f"h{layer + 1:02d}.pt")
        if layer == 0:
            torch.save(cos.cpu(), TARGETS / "cos.pt")
            torch.save(sin.cpu(), TARGETS / "sin.pt")
        del fp, state, cos, sin
        torch.cuda.empty_cache()
        if layer % 8 == 7 or layer == N_LAYERS - 1:
            log(f"prep layer {layer}: {time.time() - started:.0f}s")
    log(f"prep done {time.time() - started:.0f}s")


def phase_train(args, index, tcfg):
    TARGETS.mkdir(parents=True, exist_ok=True)
    sign_sets = rotation.load_sign_manifest(MANIFEST)
    cos_full = torch.load(TARGETS / "cos.pt")
    sin_full = torch.load(TARGETS / "sin.pt")
    n = args.windows
    report = {"layers": [], "windows": args.windows, "seq": args.seq, "steps": args.steps}
    started = time.time()

    def slot_for(layer):
        device = f"cuda:{args.devices[layer % len(args.devices)]}"
        student, masters, count = build_student(index, tcfg, layer, sign_sets, device)
        h_in = torch.load(TARGETS / f"h{layer:02d}.pt").to(device).float()
        tgt = torch.load(TARGETS / f"h{layer + 1:02d}.pt").to(device).float()
        with torch.no_grad():
            before = student(h_in, (cos_full.to(device).float(), sin_full.to(device).float()))
            rel_before = (before - tgt).norm().item() / tgt.norm().item()
        return {
            "layer": layer, "device": device, "student": student, "masters": masters, "count": count,
            "h": h_in, "tgt": tgt, "opt": torch.optim.Adam(masters, lr=args.lr),
            "gen": torch.Generator().manual_seed(1000 + layer), "rel_before": rel_before,
            "cos": cos_full.to(device), "sin": sin_full.to(device),
        }

    def step(sl):
        idx = torch.randint(0, n, (args.batch,), generator=sl["gen"])
        x = sl["h"][idx]
        y = sl["tgt"][idx]
        out = sl["student"](x, (sl["cos"][idx].float(), sl["sin"][idx].float()))
        loss = F.mse_loss(out, y)
        sl["opt"].zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sl["masters"], 1.0)
        sl["opt"].step()
        return loss.detach()

    for start in range(args.start_layer, args.end_layer, 2):
        pair = [layer for layer in (start, start + 1) if layer < args.end_layer]
        slots = [slot_for(layer) for layer in pair]
        for sl in slots:
            sl["student"].train()
        losses = {sl["layer"]: 0.0 for sl in slots}
        for step_i in range(args.steps):
            for sl in slots:
                sl["loss_t"] = step(sl)
            if step_i % 50 == 0 or step_i == args.steps - 1:
                for sl in slots:
                    losses[sl["layer"]] = sl["loss_t"].item()
        for sl in slots:
            sl["student"].eval()
            with torch.no_grad():
                after = sl["student"](sl["h"], (sl["cos"].float(), sl["sin"].float()))
                rel_after = (after - sl["tgt"]).norm().item() / sl["tgt"].norm().item()
            payload = {name: module.master.detach().to(torch.bfloat16).cpu()
                       for name, module in sl["student"].named_modules() if isinstance(module, TernaryLinear)}
            torch.save(payload, OUT / f"layer{sl['layer']:02d}-masters.pt")
            report["layers"].append({
                "layer": sl["layer"], "device": sl["device"], "ternary_linears": sl["count"],
                "rel_before": sl["rel_before"], "rel_after": rel_after,
                "loss": losses[sl["layer"]], "seconds": time.time() - started,
            })
            log(f"layer {sl['layer']:2d} [{sl['device']}]: {sl['count']} ternary, "
                f"rel {sl['rel_before']:.4f} -> {rel_after:.4f}, {time.time() - started:.0f}s")
        (OUT / "blockwise-tf-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        for sl in slots:
            del sl["student"], sl["masters"], sl["opt"], sl["h"], sl["tgt"], sl["cos"], sl["sin"]
        del slots
        gc.collect()
        torch.cuda.empty_cache()

    report["seconds"] = time.time() - started
    (OUT / "blockwise-tf-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"train done {report['seconds']:.0f}s")


def main() -> int:
    global OUT, TARGETS
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("prep", "train", "all"), default="all")
    ap.add_argument("--start-layer", type=int, default=0)
    ap.add_argument("--end-layer", type=int, default=N_LAYERS)
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--prep-device", type=int, default=1)
    ap.add_argument("--devices", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    OUT = Path(args.out)
    TARGETS = OUT / "targets"
    OUT.mkdir(parents=True, exist_ok=True)

    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(BASE)
    tcfg = cfg.text_config
    index = load_index()

    if args.phase in ("prep", "all"):
        phase_prep(args, index, tcfg)
    if args.phase in ("train", "all"):
        phase_train(args, index, tcfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
