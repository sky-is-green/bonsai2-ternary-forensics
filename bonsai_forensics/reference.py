"""T27 — third-party reference bar: ThakiCloud QuIP, adapted for ternary.

Ported from `ThakiCloud/bonsai-1bit-repro` `scripts/quip.py` (Apache-2.0) so we
can (a) reproduce their published binary result on Qwen3-1.7B and (b) run the
same rotation + GPTQ machinery in ternary mode as a quality bar our pipeline
must meet or beat before any 27B rental.

What it does per Linear layer: rotate `W` (and the calibration Hessian) into an
incoherent basis with a dense random orthogonal matrix, run GPTQ error
compensation there, then fold the rotation back. The final weights are dense
fp32 — this is a *quality reference*, not a deployable format; our absorbed,
packable pipeline is compared against it.

Their construction is intentionally kept verbatim (`Hn = 2H/N`, upper-Cholesky
of `Hinv`, per-row absmean scale, block 128); the ternary op is the only
adaptation (`clip(round(w/s), -1, 1)` instead of `sign(w)`).

Usage::

    python -m bonsai_forensics.reference --model-dir artifacts/canary/hf \
        --device cuda --bits binary --rotate in --damp 0.3
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

G = 128
FIXED_TEXT = (
    "The transformer architecture relies on self-attention to model long-range "
    "dependencies. Incoherence processing rotates weights so outliers spread into "
    "near-Gaussian values, and error compensation then propagates residual quantization "
    "error to remaining weights to minimize the layer output error. Together these keep "
    "pure one-bit models from collapsing on long reasoning. Perplexity on a fixed passage "
    "is a deterministic signal; lower is better and a large jump means collapse."
) * 8

_ORTH: dict[tuple[int, int], torch.Tensor] = {}


def rand_orth(d: int, seed: int = 0) -> torch.Tensor:
    key = (d, seed)
    if key not in _ORTH:
        generator = torch.Generator().manual_seed(seed + d)
        q, _ = torch.linalg.qr(torch.randn(d, d, generator=generator))
        _ORTH[key] = q.float()
    return _ORTH[key]


def _perplexity(model, ids) -> float:
    model.eval()
    with torch.no_grad():
        return math.exp(min(model(ids, labels=ids).loss.item(), 20.0))


@torch.no_grad()
def gptq_reference(W, H, *, bits: str = "binary", damping: float = 0.3, blocksize: int = 128,
                   salient_frac: float = 0.0, group_size: int = G):
    """ThakiCloud's GPTQ loop verbatim; `bits` selects sign vs ternary codec."""
    W = W.float().clone()
    in_dim = W.shape[1]
    idx = torch.arange(in_dim, device=W.device)
    diagmean = torch.diag(H).mean().clamp(min=1e-6)
    H = H.clone()
    H[idx, idx] += damping * diagmean
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    salient = torch.zeros_like(W, dtype=torch.bool)
    if salient_frac > 0:
        hdiag = torch.diag(H).clamp(min=0)
        sal = (W ** 2) * hdiag.unsqueeze(0)
        k = max(1, int(salient_frac * sal.numel()))
        threshold = torch.kthvalue(sal.reshape(-1), sal.numel() - k + 1).values
        salient = sal >= threshold
    try:
        lower = torch.linalg.cholesky(H)
        h_inv = torch.linalg.cholesky(torch.cholesky_inverse(lower), upper=True)
    except Exception:
        h_inv = torch.eye(in_dim, device=W.device)

    def code(w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if bits == "binary":
            return scale.squeeze(1) * torch.sign(w + (w == 0).float())
        return scale.squeeze(1) * torch.clamp(torch.round(w / scale.squeeze(1)), -1, 1)

    Q = torch.zeros_like(W)
    for i1 in range(0, in_dim, blocksize):
        i2 = min(i1 + blocksize, in_dim)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        h_inv1 = h_inv[i1:i2, i1:i2]
        salient1 = salient[:, i1:i2]
        scale = W1.abs().mean(dim=1, keepdim=True).clamp(min=1e-9)
        for j in range(i2 - i1):
            w = W1[:, j]
            q = code(w, scale)
            if salient_frac > 0:
                q = torch.where(salient1[:, j], w, q)
            Q1[:, j] = q
            e = (w - q) / h_inv1[j, j]
            W1[:, j:] -= e.unsqueeze(1) * h_inv1[j, j:].unsqueeze(0)
            E1[:, j] = e
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ h_inv[i1:i2, i2:]
    return Q


def _heldout_windows(tokenizer, corpus: Path, samples: int, seq: int, eval_windows: int):
    ids = tokenizer(corpus.read_text(encoding="utf-8"), return_tensors="np")["input_ids"].reshape(-1)
    start = samples * seq
    return ids[start : start + eval_windows * seq].reshape(eval_windows, seq)

def run(args) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.float32).to(args.device)
    fixed = tokenizer(FIXED_TEXT, return_tensors="pt")["input_ids"].to(args.device)
    ppl_before_fixed = _perplexity(model, fixed)

    targets = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and (args.include_embed_head or ("lm_head" not in name and "embed" not in name))
    }
    hessians = {n: torch.zeros(m.weight.shape[1], m.weight.shape[1]) for n, m in targets.items()}
    counts = {n: 0 for n in targets}
    handles = []

    def hook(name):
        def forward_pre(module, inputs):
            x = inputs[0].reshape(-1, inputs[0].shape[-1]).float()
            hessians[name] += (x.t() @ x).cpu()  # keep the 7.5 GB of Hessians off the GPU
            counts[name] += x.shape[0]
        return forward_pre

    for name, module in targets.items():
        handles.append(module.register_forward_pre_hook(hook(name)))
    with torch.no_grad():
        if args.calib_corpus:
            calib_ids = tokenizer(
                Path(args.calib_corpus).read_text(encoding="utf-8"), return_tensors="np"
            )["input_ids"].reshape(-1)
            windows = calib_ids[: args.calib_samples * args.calib_seq_len].reshape(
                args.calib_samples, args.calib_seq_len
            )
            for window in windows:
                ids = torch.tensor(window, dtype=torch.long, device=args.device).unsqueeze(0)
                model(ids)
        else:
            model(fixed)
    for handle in handles:
        handle.remove()

    heldout_ref: list[float] | None = None
    if args.eval_corpus:
        eval_windows = _heldout_windows(tokenizer, Path(args.eval_corpus), args.samples, args.seq_len, args.eval_windows)
        with torch.no_grad():
            heldout_ref = [
                _perplexity(model, torch.tensor(w, dtype=torch.long, device=args.device).unsqueeze(0))
                for w in eval_windows
            ]

    with torch.no_grad():
        for name, module in targets.items():
            H = (2 * hessians[name] / max(counts[name], 1)).to(module.weight.device)
            W = module.weight.data.float()
            rotations_in = rotations_out = None
            if args.rotate in ("in", "both"):
                rotations_in = rand_orth(W.shape[1], args.seed).to(W.device)
                W = W @ rotations_in.t()
                H = rotations_in @ H @ rotations_in.t()
            if args.rotate == "both":
                rotations_out = rand_orth(W.shape[0], args.seed).to(W.device)
                W = rotations_out @ W
            Q = gptq_reference(W, H, bits=args.bits, damping=args.damp, salient_frac=args.salient_frac)
            if rotations_out is not None:
                Q = rotations_out.t() @ Q
            if rotations_in is not None:
                Q = Q @ rotations_in
            module.weight.data = Q.to(module.weight.dtype)
            hessians[name] = None

    ppl_after_fixed = _perplexity(model, fixed)
    result = {
        "model": args.model_dir,
        "bits": args.bits,
        "rotate": args.rotate,
        "damp": args.damp,
        "salient_frac": args.salient_frac,
        "include_embed_head": args.include_embed_head,
        "fixed_passage": {
            "ppl_fp32": round(ppl_before_fixed, 3),
            "ppl_quant": round(ppl_after_fixed, 3),
            "ratio": round(ppl_after_fixed / ppl_before_fixed, 3),
        },
        "seconds": round(time.time() - started, 1),
    }
    if args.eval_corpus:
        windows = _heldout_windows(tokenizer, Path(args.eval_corpus), args.samples, args.seq_len, args.eval_windows)
        ratios = []
        with torch.no_grad():
            for window in windows:
                ids = torch.tensor(window, dtype=torch.long, device=args.device).unsqueeze(0)
                ratios.append(_perplexity(model, ids))
        ref_mean = sum(heldout_ref or ratios) / len(ratios)
        quant_mean = sum(ratios) / len(ratios)
        result["heldout"] = {
            "ppl_fp32_mean": round(ref_mean, 3),
            "ppl_quant_mean": round(quant_mean, 3),
            "ratio": round(quant_mean / ref_mean, 3),
            "collapsed": quant_mean >= 1e8,  # _perplexity clips loss at 20
            "ppl_quant": [round(r, 2) for r in ratios],
        }
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--bits", choices=("binary", "ternary"), default="binary")
    parser.add_argument("--rotate", choices=("none", "in", "both"), default="in")
    parser.add_argument("--damp", type=float, default=0.3)
    parser.add_argument("--salient-frac", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--include-embed-head", action="store_true")
    parser.add_argument("--eval-corpus", default="")
    parser.add_argument("--calib-corpus", default="", help="capture Hessians on real windows instead of FIXED_TEXT")
    parser.add_argument("--calib-samples", type=int, default=32)
    parser.add_argument("--calib-seq-len", type=int, default=2048)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--eval-windows", type=int, default=4)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    result = run(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
