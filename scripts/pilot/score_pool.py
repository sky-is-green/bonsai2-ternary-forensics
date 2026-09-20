"""Score pool windows by teacher CE and RTN-student CE; select top-20% + random control."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HB))

from bonsai_forensics.recover import wrap_ternary

OUT = HB / "artifacts/pilot"
MODEL_DIR = HB / "artifacts/canary/hf"
BATCH = 2


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[score] device {torch.cuda.get_device_name(0)}", flush=True)
    pool = np.load(OUT / "pool_windows.npy")
    evals = np.load(OUT / "eval_windows.npy")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    @torch.no_grad()
    def ce_per_window(windows, tag):
        path = OUT / f"{tag}_partial.npy"
        out = np.full(len(windows), np.nan, dtype=np.float32)
        start = 0
        if path.is_file():
            previous = np.load(path)
            out[: len(previous)] = previous
            start = len(previous)
            print(f"[score] {tag} resumed at {start}", flush=True)
        for i in range(start, len(windows), BATCH):
            chunk = torch.tensor(windows[i:i + BATCH], dtype=torch.long, device="cuda")
            logits = model(chunk).logits.float()[:, :-1, :]
            targets = chunk[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
            ).reshape(chunk.size(0), -1).mean(dim=1)
            out[i:i + BATCH] = loss.cpu().numpy()
            if (i // BATCH) % 250 == 0:
                np.save(path, out[: i + BATCH])
                print(f"[score] {tag} {i}/{len(windows)}", flush=True)
        np.save(path, out)
        return out

    started = time.time()
    if (OUT / "teacher_ce.npy").is_file():
        teacher = np.load(OUT / "teacher_ce.npy")
        print("[score] teacher scores resumed from disk", flush=True)
    else:
        teacher = ce_per_window(pool, "teacher")
        np.save(OUT / "teacher_ce.npy", teacher)
        print(f"[score] teacher done in {time.time() - started:.0f}s", flush=True)
    if (OUT / "student_ce.npy").is_file():
        student = np.load(OUT / "student_ce.npy")
        print("[score] student scores resumed from disk", flush=True)
    else:
        wrap_ternary(model)
        student = ce_per_window(pool, "student")
        np.save(OUT / "student_ce.npy", student)
        print(f"[score] student done in {time.time() - started:.0f}s", flush=True)
    excess = student - teacher
    np.savez(OUT / "scores.npz", teacher=teacher, student=student, excess=excess)

    k = len(pool) // 5
    order = np.argsort(-excess)
    top = np.sort(order[:k])
    rng = np.random.default_rng(0)
    rand = np.sort(rng.choice(len(pool), size=k, replace=False))

    def decode(idx):
        return "\n\n".join(tokenizer.decode(pool[i], skip_special_tokens=False) for i in idx)

    (OUT / "selected.txt").write_text(decode(top), encoding="utf-8")
    (OUT / "random.txt").write_text(decode(rand), encoding="utf-8")
    (OUT / "eval.txt").write_text(
        "\n\n".join(tokenizer.decode(w, skip_special_tokens=False) for w in evals), encoding="utf-8"
    )
    report = {
        "teacher_mean_ce": float(teacher.mean()),
        "student_mean_ce": float(student.mean()),
        "excess_mean": float(excess.mean()),
        "selected_windows": int(k),
        "selected_excess_mean": float(excess[top].mean()),
        "random_excess_mean": float(excess[rand].mean()),
        "selected_teacher_mean": float(teacher[top].mean()),
        "random_teacher_mean": float(teacher[rand].mean()),
        "seconds": time.time() - started,
    }
    (OUT / "score-report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("[score] " + json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
