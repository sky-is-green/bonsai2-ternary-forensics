"""Decode Wikitext-103 pool windows into text corpus files for recover.py.

recover.py consumes a plain-text corpus (it re-tokenizes on load). The pilot
pool holds token windows, so this decodes a slice of the pool back to text,
matching the existing selected.txt/random.txt convention. The held-out eval
corpus is decoded from eval_windows.npy (same slices score_pool used).

Usage::

    python scripts/pilot/prep_corpus_text.py --windows 3000 \
        --pool artifacts/ternary/pilot/pool_windows.npy \
        --eval artifacts/ternary/pilot/eval_windows.npy \
        --tokenizer artifacts/ternary/canary/hf \
        --out artifacts/ternary/pilot/wikitext_3000.txt \
        --out-eval artifacts/ternary/pilot/eval.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="pool_windows.npy from prep_pool.py")
    ap.add_argument("--eval", default="", help="eval_windows.npy; decoded to --out-eval")
    ap.add_argument("--tokenizer", required=True, help="local HF tokenizer dir")
    ap.add_argument("--windows", type=int, default=3000, help="pool windows to decode")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-eval", default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    pool = np.load(args.pool)
    n = min(args.windows, len(pool))
    chunks = [tokenizer.decode(pool[i], skip_special_tokens=False) for i in range(n)]
    text = "\n\n".join(chunks)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"[prep-corpus] {n} windows -> {args.out} ({len(text)} chars, {len(text.split())} words)", flush=True)

    if args.out_eval and args.eval:
        evals = np.load(args.eval)
        etext = "\n\n".join(
            tokenizer.decode(evals[i], skip_special_tokens=False) for i in range(len(evals))
        )
        Path(args.out_eval).write_text(etext, encoding="utf-8")
        print(f"[prep-corpus] {len(evals)} eval windows -> {args.out_eval}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())