"""Build a Wikitext-103 window pool + held-out eval set for the selection pilot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

HB = Path(__file__).resolve().parents[2]
OUT = HB / "artifacts/pilot"
WIN = 2048
POOL = 6000
EVAL = 16


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(HB / "artifacts/canary/hf")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    text = "\n".join(dataset["text"][:400000])
    ids = tokenizer(text, return_tensors="np")["input_ids"].reshape(-1)
    print(f"[prep] {len(dataset)} rows, {len(text)} chars, {ids.size} tokens", flush=True)
    needed = (POOL + EVAL) * WIN
    if ids.size < needed:
        raise SystemExit(f"need {needed} tokens, have {ids.size}")
    pool = ids[: POOL * WIN].reshape(POOL, WIN)
    evals = ids[POOL * WIN : needed].reshape(EVAL, WIN)
    np.save(OUT / "pool_windows.npy", pool)
    np.save(OUT / "eval_windows.npy", evals)
    print(f"[prep] pool {pool.shape}, eval {evals.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
