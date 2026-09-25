"""Qwen3.5-MoE (35B-A3B) ternary proxy: fused expert banks + correction branches.

Port of ``olmoe_proxy.py`` / ``olmoe_corrections.py`` to the ``qwen3_5_moe``
architecture:

  - experts are fused parameters (``experts.gate_up_proj`` /
    ``experts.down_proj``), ternarised in place (STE), 256 experts top-8;
  - the router is ``layer.mlp.gate`` (not an ``nn.Linear``); the shared expert
    and the attention/GDN projections stay FP for the first cut;
  - 40 layers (30 Gated-DeltaNet, 10 full attention), hidden 2048;
  - corrections are residual-stream branches on each ``layer.mlp`` output plus
    trainable routers; loss = LM + output KD (router KD optional, default 0).

Stages:
  smoke : local prefix check (embedding + N layers from a partial checkpoint)
          of the patched forward and a few correction steps; no full model.
  cache : FP teacher pass -> per-window top-50 logits + per-layer router top-8.
  train : frozen ternary body + correction branches + routers.
  eval  : held-out PPL and per-layer router agreement.

Ops: training needs the FP teacher and ternary student resident together, so it
needs a card that holds both (or two cards via ``--device-map auto`` with
``--max-memory``); single-card stages should pin the free card
(``HIP_VISIBLE_DEVICES=1``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from moe_proxy import ternary_absmean  # noqa: E402
from olmoe_corrections import (CorrectionBranch, MoEWithCorrection,  # noqa: E402
                               load_branch_state, quantize_bank_inplace)
from olmoe_proxy import gate_hook, ternary_ste, windows  # noqa: E402

ART = Path(os.environ.get("MOE_ARTIFACTS", HERE / "artifacts"))
MODEL = ART / "empero-hf"          # full or partial qwen3_5_moe checkpoint
OUT = ART / "qwen35"
CACHE = OUT / "teacher-cache.pt"


# ------------------------------------------------------------------- patch ---

def patch_experts(group: int, model=None):
    """Make ``Qwen3_5MoeExperts`` ternarise its fused banks on the fly (STE).

    ``device_map="auto"`` binds the pre-patch forward onto each experts
    instance, shadowing the class patch; pass the loaded ``model`` to rebind
    those instances too.
    """
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts

    def ternary_forward(self, hidden_states, top_k_index, top_k_weights):
        gu = ternary_ste(self.gate_up_proj, group)
        dn = ternary_ste(self.down_proj, group)
        dt = gu.dtype
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index,
                                                      num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx].to(dt)
            gate, up = F.linear(current_state, gu[expert_idx]).chunk(2, dim=-1)
            h = self.act_fn(gate) * up
            h = F.linear(h, dn[expert_idx])
            h = h * top_k_weights[token_idx, top_k_pos, None].to(dt)
            final_hidden_states.index_add_(0, token_idx, h.to(final_hidden_states.dtype))
        return final_hidden_states

    original = Qwen3_5MoeExperts.forward

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if not getattr(self, "_ternary", False):
            return original(self, hidden_states, top_k_index, top_k_weights)
        return ternary_forward(self, hidden_states, top_k_index, top_k_weights)

    Qwen3_5MoeExperts.forward = forward

    if model is not None:
        for m in model.modules():
            if not isinstance(m, Qwen3_5MoeExperts):
                continue
            inner = m.__dict__.get("forward")

            def make(inner):
                def f(self, *a, **k):
                    if not getattr(self, "_ternary", False):
                        return inner(*a, **k)
                    return ternary_forward(self, *a, **k)
                return f

            m.forward = (make(inner).__get__(m, type(m)) if inner is not None
                         else forward.__get__(m, type(m)))


def text_layers(model):
    """Return the decoder-layer list for text-model and conditional wrappers."""
    root = model
    if hasattr(root, "language_model"):
        return root.language_model.layers
    if hasattr(root, "layers"):
        return root.layers
    return root.model.language_model.layers


# ---------------------------------------------------------------- prefix -----

def load_prefix(n_layers: int, device: str = "cuda:0", dtype=torch.bfloat16,
                model_dir: Path | None = None):
    """Load embedding + N decoder layers (+ norm/lm_head when present).

    Enough for the smoke test; reads only the local shards the index maps.
    """
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel
    from safetensors import safe_open

    model_dir = Path(model_dir or MODEL)
    cfg = AutoConfig.from_pretrained(model_dir)
    tcfg = cfg.text_config
    tcfg.num_hidden_layers = n_layers
    tcfg.layer_types = list(tcfg.layer_types)[:n_layers]
    if hasattr(tcfg, "mtp_num_hidden_layers"):
        tcfg.mtp_num_hidden_layers = 0
    model = Qwen3_5MoeTextModel(tcfg)
    tok = AutoTokenizer.from_pretrained(model_dir)

    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    handles, state, skipped = {}, {}, set()
    for key, shard in idx.items():
        if key.startswith("model.language_model."):
            stripped = key[len("model.language_model."):]
            if stripped.startswith("layers."):
                if int(stripped.split(".")[1]) >= n_layers:
                    continue
            elif stripped not in ("embed_tokens.weight", "norm.weight"):
                continue
        elif key == "lm_head.weight":
            stripped = "lm_head.weight"
        else:
            continue
        path = model_dir / shard
        if not path.exists():
            skipped.add(shard)
            continue
        h = handles.get(shard)
        if h is None:
            h = handles[shard] = safe_open(path, framework="pt", device="cpu")
        state[stripped] = h.get_tensor(key)
    if skipped:
        print(f"warning: {len(skipped)} shard(s) not on disk; skipped "
              f"({len(state)} tensors loaded)", flush=True)
    if "lm_head.weight" in state:
        model.lm_head = torch.nn.Linear(tcfg.hidden_size,
                                        state["lm_head.weight"].shape[0], bias=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # A skipped shard can leave the final norm at its zero init; make it
    # identity so the smoke's hidden states are meaningful.
    final_norm = getattr(model, "norm", None)
    if final_norm is not None and final_norm.weight.abs().sum().item() == 0:
        with torch.no_grad():
            final_norm.weight.fill_(1.0)
    model.to(dtype=dtype, device=device)
    model.eval()
    return model, tok, missing, unexpected


# ------------------------------------------------------------------- smoke ---

def smoke(args):
    patch_experts(args.group)
    model, tok, missing, unexpected = load_prefix(args.layers, args.device)
    print(f"prefix: {args.layers} layers; missing={len(missing)} "
          f"unexpected={len(unexpected)}", flush=True)
    if not hasattr(model, "lm_head"):
        raise SystemExit("prefix has no lm_head; cannot run the LM-loss smoke")

    data = windows(tok, 2, args.seq, 999, "wikitext")
    ids = data[0:1].to(args.device)

    def capture():
        routes = {}
        handles = [layer.mlp.gate.register_forward_hook(gate_hook(routes, i))
                   for i, layer in enumerate(model.layers)]
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        for h in handles:
            h.remove()
        return out, routes

    experts = [layer.mlp.experts for layer in model.layers]
    for e in experts:
        e._ternary = False
    fp_out, fp_route = capture()
    for e in experts:
        e._ternary = True
    tern_out, tern_route = capture()
    drift = float((tern_out.last_hidden_state - fp_out.last_hidden_state).norm()
                  / (fp_out.last_hidden_state.norm() + 1e-12))
    agree = float(torch.stack([
        (fp_route[i][2].unsqueeze(-1) == tern_route[i][2].unsqueeze(-2)).any(-1).float().mean()
        for i in fp_route]).mean())
    print(f"FP vs ternary: hidden drift {drift:.4f}, router top-8 agreement {agree:.4f}",
          flush=True)

    # correction branches on each MoE block output + trainable routers
    hidden = model.config.hidden_size
    for layer in model.layers:
        layer.mlp = MoEWithCorrection(layer.mlp, hidden, args.rank, args.branch_quant).to(args.device)
    for name, p in model.named_parameters():
        p.requires_grad_((".branch." in name) or ("mlp.gate." in name))
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {n_tr/1e6:.2f}M (branches + routers)", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adafactor(params, lr=args.lr, weight_decay=0.0)
    model.train()
    for step in range(1, args.steps + 1):
        hidden = model(input_ids=ids, use_cache=False).last_hidden_state
        logits = model.lm_head(hidden)
        lm = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                             ids[:, 1:].reshape(-1))
        lm.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 5 == 0 or step == 1:
            print(f"smoke step {step}: lm {lm.item():.4f}", flush=True)
    print("smoke ok", flush=True)


# ------------------------------------------------------------------- cache ---

@torch.no_grad()
def stage_cache(args):
    model, tok = load_full(args)
    data = windows(tok, args.windows, args.seq, args.seed)
    recs = []
    for w in range(len(data)):
        ids = data[w:w + 1].to(args.device)
        store = {}
        handles = [layer.mlp.gate.register_forward_hook(gate_hook(store, i))
                   for i, layer in enumerate(text_layers(model))]
        logits = model(ids).logits
        for h in handles:
            h.remove()
        t_top = logits[:, :-1].topk(args.top_logits, dim=-1)
        router = {i: (store[i][2].cpu().to(torch.int16),
                      store[i][1].cpu().to(torch.float16)) for i in store}
        recs.append({"idx": t_top.indices.cpu().to(torch.int32),
                     "val": t_top.values.cpu().to(torch.float16),
                     "router": router})
        if w % 100 == 0:
            print(f"cached {w}/{len(data)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(recs, CACHE)
    print(f"wrote {CACHE} ({CACHE.stat().st_size/1e9:.2f} GB)", flush=True)


# ------------------------------------------------------------------- train ---

def load_full(args):
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    mm = None
    if args.max_memory:
        mm = {int(k): v for k, v in (part.split(":") for part in args.max_memory.split(","))}
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=args.device_map, max_memory=mm)
    return model, tok


def stage_train(args):
    patch_experts(args.group)
    model, tok = load_full(args)
    layers = text_layers(model)
    with torch.no_grad():
        for layer in layers:
            quantize_bank_inplace(layer.mlp.experts.gate_up_proj, args.group)
            quantize_bank_inplace(layer.mlp.experts.down_proj, args.group)
            layer.mlp.experts._ternary = False          # banks already quantised
    hidden = getattr(model.config, "hidden_size", None) or model.config.text_config.hidden_size
    for layer in layers:
        dev = next(layer.mlp.parameters()).device
        layer.mlp = MoEWithCorrection(layer.mlp, hidden, args.rank, args.branch_quant).to(dev)
    for name, p in model.named_parameters():
        p.requires_grad_((".branch." in name) or (".mlp.gate." in name))
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {n_tr/1e6:.2f}M (branches + routers)", flush=True)

    cache = torch.load(CACHE, map_location="cpu")
    data = windows(tok, args.windows, args.seq, args.seed)
    opt = torch.optim.Adafactor([p for p in model.parameters() if p.requires_grad],
                                lr=args.lr, weight_decay=0.0)
    model.train()
    step = 0
    for epoch in range(args.epochs):
        for rec in cache:
            ids = data[step % len(data):step % len(data) + 1].to(args.device)
            logits = model(ids).logits
            lm = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                 ids[:, 1:].reshape(-1))
            ti = rec["idx"].to(logits.device)
            tv = rec["val"].to(logits.device).float()
            s_sel = logits[:, :-1].gather(-1, ti).reshape(-1, ti.shape[-1])
            kd = F.kl_div(F.log_softmax(s_sel.float() / args.temp, dim=-1),
                          F.log_softmax(tv.reshape(-1, tv.shape[-1]) / args.temp, dim=-1),
                          log_target=True, reduction="batchmean") * (args.temp ** 2)
            loss = lm + args.kd_weight * kd
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                print(f"step {step} lm {lm.item():.4f} kd {kd.item():.4f}", flush=True)
            if args.ckpt_every and step % args.ckpt_every == 0:
                save(model, args, step)
            if args.steps and step >= args.steps:
                break
        if args.steps and step >= args.steps:
            break
    save(model, args, step)
    print("training done", flush=True)


def save(model, args, step):
    sd = {k: v for k, v in model.state_dict().items()
          if ".branch." in k or ".mlp.gate." in k}
    tag = "" if args.branch_quant == "fp32" else f"-{args.branch_quant}"
    p = OUT / f"qwen35-corr-r{args.rank}{tag}-step{step}.pt"
    torch.save(sd, p)
    print(f"saved {p}", flush=True)


# -------------------------------------------------------------------- eval ---

@torch.no_grad()
def stage_eval(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    data = windows(tok, args.eval_windows, args.seq, 999, "wikitext")

    patch_experts(args.group)
    model, _ = load_full(args)
    layers = text_layers(model)
    with torch.no_grad():
        for layer in layers:
            quantize_bank_inplace(layer.mlp.experts.gate_up_proj, args.group)
            quantize_bank_inplace(layer.mlp.experts.down_proj, args.group)
            layer.mlp.experts._ternary = False
    hidden = getattr(model.config, "hidden_size", None) or model.config.text_config.hidden_size
    for layer in layers:
        dev = next(layer.mlp.parameters()).device
        layer.mlp = MoEWithCorrection(layer.mlp, hidden, args.rank, args.branch_quant).to(dev)
    if args.load:
        missing, unexpected = load_branch_state(model, args.load)
        print(f"loaded {args.load}: missing={len(missing)} unexpected={len(unexpected)}",
              flush=True)
    model.eval()

    total, ntok = 0.0, 0
    for i in range(len(data)):
        ids = data[i:i + 1].to(args.device)
        logits = model(ids).logits
        total += F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                 ids[:, 1:].reshape(-1), reduction="sum").item()
        ntok += ids[:, 1:].numel()
    ppl = math.exp(total / ntok)
    res = {"ppl": round(ppl, 4), "checkpoint": args.load, "rank": args.rank,
           "branch_quant": args.branch_quant}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "qwen35-eval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["smoke", "cache", "train", "eval"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--max-memory", default="", help="e.g. '0:39GiB,1:39GiB'")
    ap.add_argument("--layers", type=int, default=4, help="smoke prefix depth")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument("--branch-quant", choices=["fp32", "g128", "rank"], default="fp32")
    ap.add_argument("--windows", type=int, default=4096)
    ap.add_argument("--eval-windows", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-logits", type=int, default=50)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--kd-weight", type=float, default=0.5)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--load", default="")
    args = ap.parse_args()
    if args.stage == "smoke":
        smoke(args)
    elif args.stage == "cache":
        stage_cache(args)
    elif args.stage == "train":
        stage_train(args)
    else:
        stage_eval(args)


if __name__ == "__main__":
    main()
