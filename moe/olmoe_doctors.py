"""TAARDIS-style correction branches for a ternary MoE (OLMoE).

Freeze the RTN-ternary expert banks, add trainable low-rank correction branches
("Doctors") on each MoE block output plus trainable routers, and train them
jointly end-to-end.  The router corrections are the MoE-specific addition:
they let routing track the teacher once the hidden states are repaired.

Loss = LM + output KD (top-50 teacher logits) + router KD (teacher top-8).

Usage:
  # training: both cards via device_map auto (teacher + student coexist briefly)
  HIP_VISIBLE_DEVICES=0,1 python olmoe_doctors.py train \
      --device-map auto --steps 2000 --rank 64
  # single-card stages (eval, cache): pin the free card, not the display card
  HIP_VISIBLE_DEVICES=1 python olmoe_doctors.py eval --rank 64 --load <ckpt>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from moe_proxy import ternary_absmean  # noqa: E402
from olmoe_proxy import (CACHE, MODEL, OUT, gate_hook, load_model,  # noqa: E402
                         parse_layers, windows)


@torch.no_grad()
def quantize_bank_inplace(p: torch.Tensor, group: int = 128, chunk: int = 8) -> None:
    """RTN the frozen expert bank in place, chunked to bound GPU memory."""
    for s in range(0, p.shape[0], chunk):
        part = p[s:s + chunk]
        q = ternary_absmean(part, group)
        part.copy_(q)
        del q
    torch.cuda.empty_cache()


class Doctor(nn.Module):
    """Low-rank correction branch, zero-initialised on the output side.

    Master weights stay fp32 (adapter-scale updates survive), the matmuls are
    done in fp32 and cast back to the activation dtype.

    ``quant`` controls the deployed branch format (STE during training):
      - ``fp32``  : no quantisation (reference)
      - ``g128``  : ternary codes + fp16 group scales (our expert format)
      - ``rank``  : TAARDIS V3-style, one ternary scale per rank component,
                    folded from the down factor into the up factor
    """

    def __init__(self, hidden: int, rank: int, quant: str = "fp32"):
        super().__init__()
        self.down = nn.Linear(hidden, rank, bias=False)
        self.up = nn.Linear(rank, hidden, bias=False)
        self.quant = quant
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def _weights(self):
        wd, wu = self.down.weight, self.up.weight
        if self.quant == "fp32":
            return wd, wu
        with torch.no_grad():
            if self.quant == "g128":
                wdq = ternary_absmean(wd, 128)
                wuq = ternary_absmean(wu, 128)
            else:                                    # rank component scales
                s = wd.abs().mean(dim=1).clamp_min(1e-8)       # [rank]
                qd = torch.clamp(torch.round(wd / s[:, None]), -1, 1)
                t = wu.abs().mean(dim=0).clamp_min(1e-8)       # [rank]
                qu = torch.clamp(torch.round(wu / t[None, :]), -1, 1)
                wdq = qd                                       # A is pure ternary
                wuq = qu * (s * t)[None, :]                    # B carries the A*B fold
        # STE: only the fp32 master carries grad
        return wdq + (wd - wd.detach()), wuq + (wu - wu.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wd, wu = self._weights()
        h = F.linear(x.float(), wd)
        return F.linear(h, wu).to(x.dtype)


class MoEWithDoctor(nn.Module):
    def __init__(self, mlp: nn.Module, hidden: int, rank: int, quant: str = "fp32"):
        super().__init__()
        self.mlp = mlp
        self.doctor = Doctor(hidden, rank, quant)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x) + self.doctor(x)


def _fp16_layers(args) -> set:
    return {int(x) for x in str(getattr(args, "branch_quant_fp16_layers", "")).split(",")
            if x.strip()}


def build(args):
    mm = {0: "14GiB", 1: "19GiB"} if args.device_map == "auto" else None
    model, tok = load_model(args.device_map, mm)
    hidden = model.config.hidden_size
    fp16 = _fp16_layers(args)
    for i, layer in enumerate(model.model.layers):
        experts = layer.mlp.experts
        # freeze the body in its deploy format: RTN once, no on-the-fly work
        quantize_bank_inplace(experts.gate_up_proj, args.group)
        quantize_bank_inplace(experts.down_proj, args.group)
        dev = next(layer.mlp.parameters()).device
        quant = "fp32" if i in fp16 else args.branch_quant
        layer.mlp = MoEWithDoctor(layer.mlp, hidden, args.rank, quant).to(dev)
    if fp16:
        print(f"mixed sidecar: fp16 branches on layers {sorted(fp16)}", flush=True)
    for name, p in model.named_parameters():
        p.requires_grad_(("doctor." in name) or (".gate." in name))
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {n_tr/1e6:.2f}M (doctors + routers)", flush=True)
    return model, tok


def save(model, args, step):
    sd = {k: v for k, v in model.state_dict().items()
          if ".doctor." in k or ".gate." in k}
    tag = "" if args.branch_quant == "fp32" else f"-{args.branch_quant}"
    fp16 = _fp16_layers(args)
    if fp16:
        tag += "-mixed"
    tag += f"-{args.tag}" if args.tag else ""
    p = OUT / f"olmoe-doctors-r{args.rank}{tag}-step{step}.pt"
    torch.save(sd, p)
    if args.branch_quant == "fp32":
        b = sum(v.numel() for k, v in sd.items()
                if k.endswith("down.weight") or k.endswith("up.weight")) * 4
        n_br = b / 4
    else:
        n_br = n_fp = 0
        for k, v in sd.items():
            if not (k.endswith("down.weight") or k.endswith("up.weight")):
                continue
            li = int(k.split("layers.")[1].split(".")[0]) if ".layers." in k else -1
            n_br += v.numel()
            n_fp += v.numel() if li in fp16 else 0
        n_tern = n_br - n_fp
        b = n_fp * 4 + n_tern / 4                        # 2-bit packed ternary codes
        if args.branch_quant == "g128":
            b += n_tern / 128 * 2                        # fp16 scale per 128 group
        else:
            b += args.rank * 2 * (len(model.model.layers) - len(fp16))
    print(f"saved {p} [deployed branches {b/1e6:.1f} MB, {b*8/n_br:.3f} bpw]",
          flush=True)


@torch.no_grad()
def quick_eval(model, data, args, ref=None):
    model.eval()
    total, ntok = 0.0, 0
    agree = []
    for i in range(len(data)):
        ids = data[i:i + 1].to(args.device)
        store = {}
        hs = [layer.mlp.mlp.gate.register_forward_hook(gate_hook(store, j))
              for j, layer in enumerate(model.model.layers)]
        logits = model(ids).logits
        for h in hs:
            h.remove()
        total += F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                 ids[:, 1:].reshape(-1), reduction="sum").item()
        ntok += ids[:, 1:].numel()
        if ref is not None:
            for j in store:
                a = ref[i][j].to(store[j][2].device)
                b = store[j][2]
                agree.append(float((a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).float().mean()))
    model.train()
    ppl = math.exp(total / ntok)
    return ppl, (sum(agree) / len(agree) if agree else None)


def train(args):
    model, tok = build(args)
    cache = torch.load(CACHE, map_location="cpu")
    data = windows(tok, args.windows, args.seq, args.seed,
                   max_chars=args.corpus_chars)
    # teacher router reference for the agreement metric
    ref = None
    if args.eval_every:
        from transformers import AutoModelForCausalLM
        mm = {0: "14GiB", 1: "19GiB"} if args.device_map == "auto" else None
        teacher, _ = load_model(args.device_map, mm)
        ev = windows(tok, 2, args.seq, 999, "wikitext")
        ref = {}
        for i in range(len(ev)):
            ids = ev[i:i + 1].to(args.device)
            store = {}
            hs = [layer.mlp.gate.register_forward_hook(gate_hook(store, j))
                  for j, layer in enumerate(teacher.model.layers)]
            teacher(ids)
            for h in hs:
                h.remove()
            ref[i] = {j: store[j][2].cpu() for j in store}
        del teacher
        torch.cuda.empty_cache()
    else:
        ev = None

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adafactor(params, lr=args.lr, weight_decay=0.0)
    model.train()
    step = 0
    for epoch in range(args.epochs):
        for rec in cache:
            ids = data[step % len(data):step % len(data) + 1].to(args.device)
            store = {}
            hs = [layer.mlp.mlp.gate.register_forward_hook(gate_hook(store, j))
                  for j, layer in enumerate(model.model.layers)]
            logits = model(ids).logits
            for h in hs:
                h.remove()
            lm = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                 ids[:, 1:].reshape(-1))
            ti = rec["idx"].to(logits.device)
            tv = rec["val"].to(logits.device).float()
            s_sel = logits[:, :-1].gather(-1, ti).reshape(-1, ti.shape[-1])
            kd = F.kl_div(F.log_softmax(s_sel.float() / args.temp, dim=-1),
                          F.log_softmax(tv.reshape(-1, tv.shape[-1]) / args.temp, dim=-1),
                          log_target=True, reduction="batchmean") * (args.temp ** 2)
            rkd = torch.zeros((), device=args.device)
            if args.router_weight > 0:
                for i, layer in enumerate(model.model.layers):
                    s_logits, _, _ = store[i]
                    dev = s_logits.device
                    t_idx = rec["router"][i][0].to(dev).long()
                    t_p = rec["router"][i][1].to(dev).float()
                    s_p = F.softmax(s_logits.float(), dim=-1).gather(-1, t_idx)
                    s_p = s_p / s_p.sum(-1, keepdim=True).clamp_min(1e-9)
                    t_p = t_p / t_p.sum(-1, keepdim=True).clamp_min(1e-9)
                    rkd = rkd + F.kl_div(s_p.clamp_min(1e-9).log(), t_p,
                                         reduction="batchmean").to(rkd.device)
                rkd = rkd / len(model.model.layers)
            loss = lm + args.kd_weight * kd + args.router_weight * rkd
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if (args.lr_half_every and step >= args.lr_decay_start
                    and step % args.lr_half_every == 0):
                for g in opt.param_groups:
                    g["lr"] *= 0.5
                print(f"step {step}: lr -> {opt.param_groups[0]['lr']:.3e}", flush=True)
            if step % args.log_every == 0:
                print(f"step {step} lm {lm.item():.4f} kd {kd.item():.4f} "
                      f"rkd {float(rkd):.4f} total {loss.item():.4f}", flush=True)
            if args.eval_every and step % args.eval_every == 0 and ev is not None:
                ppl, ag = quick_eval(model, ev, args, ref)
                print(f"  [eval] step {step} ppl {ppl:.2f} "
                      f"router_agree {ag:.4f}" if ag else f"  [eval] step {step} ppl {ppl:.2f}",
                      flush=True)
            if step % args.ckpt_every == 0:
                save(model, args, step)
            if args.steps and step >= args.steps:
                break
        if args.steps and step >= args.steps:
            break
    save(model, args, step)
    print("training done", flush=True)


@torch.no_grad()
def evaluate(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    data = windows(tok, args.eval_windows, args.seq, 999, "wikitext")

    # teacher
    teacher, _ = load_model(args.device)
    ref, t_total, t_ntok = {}, 0.0, 0
    for i in range(len(data)):
        ids = data[i:i + 1].to(args.device)
        store = {}
        hs = [layer.mlp.gate.register_forward_hook(gate_hook(store, j))
              for j, layer in enumerate(teacher.model.layers)]
        logits = teacher(ids).logits
        for h in hs:
            h.remove()
        ref[i] = {j: store[j][2].cpu() for j in store}
        t_total += F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                   ids[:, 1:].reshape(-1), reduction="sum").item()
        t_ntok += ids[:, 1:].numel()
    print(f"teacher ppl {math.exp(t_total/t_ntok):.4f}", flush=True)
    del teacher
    torch.cuda.empty_cache()

    model, _ = build(args)
    model.eval()

    def run(tag):
        total, ntok, agree = 0.0, 0, []
        for i in range(len(data)):
            ids = data[i:i + 1].to(args.device)
            store = {}
            hs = [layer.mlp.mlp.gate.register_forward_hook(gate_hook(store, j))
                  for j, layer in enumerate(model.model.layers)]
            logits = model(ids).logits
            for h in hs:
                h.remove()
            total += F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                     ids[:, 1:].reshape(-1), reduction="sum").item()
            ntok += ids[:, 1:].numel()
            for j in store:
                a = ref[i][j].to(store[j][2].device)
                b = store[j][2]
                agree.append(float((a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).float().mean()))
        ppl = math.exp(total / ntok)
        print(f"{tag}: ppl {ppl:.4f} router_agree {sum(agree)/len(agree):.4f}", flush=True)
        return {"ppl": round(ppl, 4), "router_agree": round(sum(agree) / len(agree), 4)}

    res = {"rtn_no_doctors": run("rtn_no_doctors")}
    if args.load:
        sd = torch.load(args.load, map_location="cpu")
        model.load_state_dict(sd, strict=False)
        res["trained_doctors"] = run("trained_doctors")
    (OUT / "doctors-eval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "eval"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--windows", type=int, default=512)
    ap.add_argument("--corpus-chars", type=int, default=10_000_000,
                    help="fineweb character buffer; must match the cache build")
    ap.add_argument("--eval-windows", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--branch-quant", choices=["fp32", "g128", "rank"], default="fp32",
                    help="deployed branch format; STE-trained when != fp32")
    ap.add_argument("--branch-quant-fp16-layers", default="",
                    help="comma-separated layer indices kept fp32 in a mixed sidecar")
    ap.add_argument("--tag", default="", help="optional run tag for checkpoint names")
    ap.add_argument("--lr-half-every", type=int, default=0,
                    help="halve the LR every N steps (0=off)")
    ap.add_argument("--lr-decay-start", type=int, default=0,
                    help="step at which LR halving begins")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--kd-weight", type=float, default=0.5)
    ap.add_argument("--router-weight", type=float, default=0.5)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--load", default="")
    args = ap.parse_args()
    if args.stage == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
