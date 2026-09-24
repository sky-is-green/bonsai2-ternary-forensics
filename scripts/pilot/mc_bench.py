"""Minimal multiple-choice benchmark — a screening proxy, not Prism's ruler.

Prism publishes benchmark retention, but this lightweight ARC/HellaSwag/PIQA
harness is not lm-eval and does not reproduce Prism's exact suite, prompts,
versions, or uncertainty protocol.  It is useful for a matched FP-vs-student
diagnostic when both arms use this same harness; do not label its ratio as a
like-for-like Prism comparison.

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
sys.path.insert(0, str(HB))

from bonsai_forensics.modeling import load_text_causal_lm  # noqa: E402
from bonsai_forensics.rotation import load_sign_manifest  # noqa: E402
from bonsai_forensics.targets import infer_profile  # noqa: E402
from scripts.pilot.rmd_kd import wrap_rotated  # noqa: E402


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
        # `ybisk/piqa` ships only a loading script, and datasets>=3 removed
        # script support. `lighteval/piqa` (plain_text) is the same data as
        # parquet, with the same goal/sol1/sol2/label schema.
        ds = load_dataset("lighteval/piqa", "plain_text", split="validation")
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
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--checkpoint", default=None,
                    help="optional rmd_kd student.pt; loads the rotated ternary model")
    ap.add_argument("--rot-block", type=int, default=None)
    ap.add_argument("--rot-seed", type=int, default=None)
    ap.add_argument("--target-profile", default=None)
    ap.add_argument("--target-suffixes", default=None)
    ap.add_argument("--include-lm-head", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--rotation-mode", choices=("residual", "input"), default=None)
    ap.add_argument("--signs-manifest", default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--tasks", nargs="+", default=["arc_easy", "hellaswag", "piqa"])
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    device = torch.device(args.device)
    payload = None
    checkpoint_config = {}
    checkpoint_manifest = {}
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        checkpoint_config = dict(payload.get("config") or {})
        checkpoint_manifest = payload.get("manifest") or {}
    revision = args.model_revision or checkpoint_manifest.get("model", {}).get("revision")
    tok = AutoTokenizer.from_pretrained(args.model_dir, revision=revision)
    model = load_text_causal_lm(
        args.model_dir, revision=revision, dtype=torch.bfloat16, device=device)
    if payload is not None:
        profile = infer_profile(
            model, args.target_profile or checkpoint_config.get("target_profile", "auto"))
        suffix_value = (args.target_suffixes if args.target_suffixes is not None
                        else checkpoint_config.get("target_suffixes"))
        suffixes = tuple(s.strip() for s in suffix_value.split(",") if s.strip()) if suffix_value else None
        include_head = (args.include_lm_head if args.include_lm_head is not None
                        else checkpoint_config.get("include_lm_head"))
        rotation_mode = args.rotation_mode or checkpoint_config.get("rotation_mode", "residual")
        rot_seed = args.rot_seed if args.rot_seed is not None else checkpoint_config.get("rot_seed")
        if rot_seed is None or (not checkpoint_manifest and rot_seed == 0):
            # Legacy checkpoints stored parser default 0, which meant "follow
            # seed" in the original runner.
            rot_seed = checkpoint_config.get("seed", args.seed)
        block = args.rot_block if args.rot_block is not None else checkpoint_config.get("rot_block")
        signs_path = args.signs_manifest or checkpoint_config.get("signs_manifest")
        wrap_rotated(
            model, seed=int(rot_seed), ste=True, learn_scale=False, block=block,
            suffixes=suffixes, profile=profile, include_lm_head=include_head,
            rotation_mode=rotation_mode,
            sign_sets=load_sign_manifest(signs_path) if signs_path else None)
        model.load_state_dict(payload["state"], strict=False)
        print(f"[mc] loaded checkpoint step={payload.get('step')} (manifest recipe)")
    model.eval()

    results = []
    for t in args.tasks:
        rows = load_task(t, args.limit)
        r = evaluate(model, tok, t, rows, device)
        print(f"[mc] {t}: {r['acc']*100:.1f}%  (n={r['n']})", flush=True)
        results.append(r)

    report = {
        "model_dir": args.model_dir,
        "model_revision": revision,
        "checkpoint": args.checkpoint,
        "checkpoint_manifest_present": bool(checkpoint_manifest),
        "harness": "arc_easy+hellaswag+piqa-screening-proxy",
        "datasets": {
            "arc_easy": {"path": "allenai/ai2_arc", "config": "ARC-Easy", "split": "test"},
            "hellaswag": {"path": "Rowan/hellaswag", "split": "validation"},
            "piqa": {"path": "lighteval/piqa", "config": "plain_text", "split": "validation"},
        },
        "comparable_to_prism": False,
        "limit": args.limit,
        "tasks": results,
        "mean_acc": sum(r["acc"] for r in results) / len(results),
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"[mc] mean acc {report['mean_acc']*100:.1f}% -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
