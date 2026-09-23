"""Minimal multiple-choice benchmark — accuracy retention, Prism's ruler.

Prism's published retention is *benchmark accuracy* (quantized / FP16), not a
PPL ratio. To compare our recipe on the same ruler we need an accuracy number,
so this runs a few standard MC tasks and reports accuracy. The absolute numbers
are a light harness (not lm-eval), but retention = quantized/FP uses the *same*
harness for both arms, so the ratio is meaningful even where the absolute score
is not.

Tasks are context+continuation scored by mean log-prob of the continuation:
  arc_easy   allenai/ai2_arc (ARC-Easy)
  hellaswag  Rowan/hellaswag
  piqa       ybisk/piqa

    python scripts/pilot/mc_bench.py --model-dir <base> [--checkpoint student-best.pt] \
        --tasks arc_easy hellaswag piqa --limit 400 --device cuda:0 --out out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HB / "scripts" / "pilot"))

from rmd_kd import wrap_rotated  # noqa: E402


def load_task(name: str, limit: int | None):
    from datasets import load_dataset

    rows = []
    if name == "arc_easy":
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
        for r in ds:
            labels = r["choices"]["label"]
            texts = r["choices"]["text"]
            key = r["answerKey"]
            if key not in labels:
                continue
            rows.append((f"Question: {r['question']}\nAnswer:", texts, labels.index(key)))
    elif name == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")
        for r in ds:
            rows.append((r["ctx"], r["endings"], int(r["label"])))
    elif name == "piqa":
        ds = load_dataset("ybisk/piqa", split="validation")
        for r in ds:
            rows.append((f"Question: {r['goal']}\nAnswer:", [r["sol1"], r["sol2"]], int(r["label"])))
    else:
        raise SystemExit(f"unknown task {name}")
    return rows[:limit] if limit else rows


@torch.no_grad()
def score_choice(model, tok, prompt: str, choice: str, device) -> float:
    ctx = tok(prompt, return_tensors="pt").input_ids.to(device)
    full = tok(prompt + " " + choice, return_tensors="pt").input_ids.to(device)
    n_ctx = ctx.shape[1]
    if full.shape[1] <= n_ctx:
        return float("-inf")
    logits = model(full).logits[0, :-1].float()
    targets = full[0, 1:]
    logp = F.log_softmax(logits, dim=-1)
    choice_logp = logp[torch.arange(n_ctx - 1, full.shape[1] - 1, device=device),
                       targets[n_ctx - 1:]]
    return float(choice_logp.mean())


def evaluate(model, tok, task: str, rows, device) -> dict:
    correct = 0
    items = []
    for prompt, choices, gold in rows:
        scores = [score_choice(model, tok, prompt, c, device) for c in choices]
        pred = max(range(len(scores)), key=lambda i: scores[i])
        correct += int(pred == gold)
        items.append({"task": task, "gold": gold, "pred": pred, "scores": scores})
    n = len(rows)
    return {"task": task, "n": n, "acc": correct / n if n else 0.0, "items": items}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="optional rmd_kd student.pt; loads the rotated ternary model")
    ap.add_argument("--rot-block", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--tasks", nargs="+", default=["arc_easy", "hellaswag", "piqa"])
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16).to(device)
    if args.checkpoint:
        wrap_rotated(model, seed=args.seed, ste=True, learn_scale=False, block=args.rot_block)
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state"], strict=False)
        print(f"[mc] loaded checkpoint step={payload.get('step')} (rotated, STE ternary)")
    model.eval()

    results = []
    for t in args.tasks:
        rows = load_task(t, args.limit)
        r = evaluate(model, tok, t, rows, device)
        print(f"[mc] {t}: {r['acc']*100:.1f}%  (n={r['n']})", flush=True)
        results.append(r)

    report = {
        "model_dir": args.model_dir,
        "checkpoint": args.checkpoint,
        "limit": args.limit,
        "tasks": results,
        "mean_acc": sum(r["acc"] for r in results) / len(results),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"[mc] mean acc {report['mean_acc']*100:.1f}% -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
