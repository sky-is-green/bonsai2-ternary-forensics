"""Held-out evaluation for a T28 recover checkpoint: PPL + KL vs the FP teacher."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HB))

from bonsai_forensics.recover import freeze_non_ternary, wrap_ternary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model-dir", default=str(HB / "artifacts/canary/hf"))
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    ids = tokenizer(Path(args.corpus).read_text(encoding="utf-8"), return_tensors="np")["input_ids"].reshape(-1)
    windows = ids[: args.windows * args.seq].reshape(args.windows, args.seq)

    teacher = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16).to(args.device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16).to(args.device)
    wrap_ternary(student)
    freeze_non_ternary(student)
    payload = torch.load(args.checkpoint, map_location="cpu")
    student.load_state_dict(payload["state"])
    student.eval()

    rows = []
    with torch.no_grad():
        for window in windows:
            batch = torch.tensor(window, dtype=torch.long, device=args.device).unsqueeze(0)
            teacher_logits = teacher(batch).logits.float()
            student_logits = student(batch).logits.float()
            teacher_loss = torch.nn.functional.cross_entropy(
                teacher_logits[:, :-1].reshape(-1, teacher_logits.size(-1)), batch[:, 1:].reshape(-1)
            ).item()
            student_loss = torch.nn.functional.cross_entropy(
                student_logits[:, :-1].reshape(-1, student_logits.size(-1)), batch[:, 1:].reshape(-1)
            ).item()
            kl = torch.nn.functional.kl_div(
                torch.log_softmax(student_logits[:, :-1] / 2.0, dim=-1),
                torch.softmax(teacher_logits[:, :-1] / 2.0, dim=-1),
                reduction="batchmean",
            ).item() * 4.0
            rows.append({
                "teacher_ppl": math.exp(teacher_loss),
                "student_ppl": math.exp(student_loss),
                "kl": kl,
            })
    report = {
        "checkpoint": args.checkpoint,
        "corpus": args.corpus,
        "windows": args.windows,
        "seq": args.seq,
        "teacher_ppl": float(np.mean([r["teacher_ppl"] for r in rows])),
        "student_ppl": float(np.mean([r["student_ppl"] for r in rows])),
        "kl": float(np.mean([r["kl"] for r in rows])),
        "per_window": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("teacher_ppl", "student_ppl", "kl")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
