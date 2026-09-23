"""KLD vs the FP base — llama.cpp's convention, computed directly in PyTorch.

Why not the llama.cpp wrapper: our student is a `RotatedLinear` model (ternary
codes in the rotated basis + the absorbed rotation). Packing it to GGUF would
drop the rotation and measure a different model. This computes the same quantity
llama-perplexity does, on the true deployed forward.

llama.cpp's definition (tools/perplexity/perplexity.cpp, log_softmax()):
    kld(token) = sum_i p_base[i] * (log p_base[i] - log p_model[i]),
                 over i with log p_base[i] > -16
    also reported: same-top-token rate, mean/RMS delta_p (p_model - p_base at the
    true token), and PPL ratio.

    python scripts/pilot/kld_eval.py --model-dir <base> --checkpoint <student.pt> \
        --corpus <txt> --chunks 50 --ctx 2048 --device cuda:0 --out out.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HB / "scripts" / "pilot"))

from rmd_kd import wrap_rotated  # noqa: E402

LOG_BASE_FLOOR = -16.0


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--chunks", type=int, default=50)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--rot-block", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    ids = tok(Path(args.corpus).read_text(encoding="utf-8"), return_tensors="np")["input_ids"].reshape(-1)
    n = (len(ids) // args.ctx) * args.ctx
    ids = torch.tensor(ids[:n], dtype=torch.long).reshape(-1, args.ctx)
    ids = ids[: args.chunks]
    print(f"[kld] {ids.shape[0]} chunks x {args.ctx} tokens", flush=True)

    teacher = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16).to(device)
    teacher.eval()
    student = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16).to(device)
    wrap_rotated(student, seed=args.seed, ste=True, learn_scale=False, block=args.rot_block)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = student.load_state_dict(payload["state"], strict=False)
    print(f"[kld] loaded step={payload.get('step')} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    student.eval()

    kld_sum = nll_s = nll_t = 0.0
    same_top = 0
    dp_sum = dp2_sum = 0.0
    count = 0
    SLICE = 256
    with torch.inference_mode():
        for i in range(ids.shape[0]):
            batch = ids[i].unsqueeze(0).to(device)
            tl = teacher(batch).logits[0, :-1]      # bf16, keep as-is
            sl = student(batch).logits[0, :-1]
            targets = batch[0, 1:]
            T = tl.shape[0]
            for t0 in range(0, T, SLICE):
                tb = tl[t0:t0 + SLICE].float()
                sb = sl[t0:t0 + SLICE].float()
                tg = targets[t0:t0 + SLICE]
                logpb = F.log_softmax(tb, dim=-1)
                logpm = F.log_softmax(sb, dim=-1)
                mask = logpb > LOG_BASE_FLOOR
                pb = logpb.exp() * mask
                kld_sum += float((pb * (logpb - logpm)).sum())
                nll_t += float((-logpb.gather(-1, tg.unsqueeze(-1)).squeeze(-1)).sum())
                nll_s += float((-logpm.gather(-1, tg.unsqueeze(-1)).squeeze(-1)).sum())
                same_top += int((tb.argmax(-1) == sb.argmax(-1)).sum())
                dp = (logpm.gather(-1, tg.unsqueeze(-1)).exp()
                      - logpb.gather(-1, tg.unsqueeze(-1)).exp()).squeeze(-1)
                dp_sum += float(dp.sum())
                dp2_sum += float((dp ** 2).sum())
                count += tg.numel()
                del tb, sb, tg, logpb, logpm, mask, pb, dp
            del tl, sl, targets
            print(f"[kld] chunk {i+1}/{ids.shape[0]} running mean KLD {kld_sum/count:.5f}", flush=True)

    report = {
        "checkpoint": args.checkpoint,
        "chunks": int(ids.shape[0]),
        "ctx": args.ctx,
        "tokens": count,
        "kld_mean": kld_sum / count,
        "ppl_base": math.exp(nll_t / count),
        "ppl_model": math.exp(nll_s / count),
        "ppl_ratio": math.exp(nll_s / count - nll_t / count),
        "same_top_p": same_top / count,
        "mean_delta_p": dp_sum / count,
        "rms_delta_p": math.sqrt(dp2_sum / count),
    }
    print("\n[kld] " + json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                                   for k, v in report.items()}, indent=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"[kld] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
