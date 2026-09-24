"""Decode pinned Wikitext pool windows into text corpus files.

The pilot consumes plain text and re-tokenizes it on load.  This helper keeps
that convenience while recording tokenizer identity and output hashes beside
the generated corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HB))
from bonsai_forensics.provenance import file_fingerprint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True, help="pool_windows.npy from prep_pool.py")
    parser.add_argument("--eval", default="", help="eval_windows.npy; decoded to --out-eval")
    parser.add_argument("--tokenizer", required=True, help="local HF tokenizer dir or Hub ID")
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--windows", type=int, default=3000, help="pool windows to decode")
    parser.add_argument("--out", required=True)
    parser.add_argument("--out-eval", default="")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, revision=args.tokenizer_revision)
    pool = np.load(args.pool)
    n = min(args.windows, len(pool))
    chunks = [tokenizer.decode(pool[i], skip_special_tokens=False) for i in range(n)]
    text = "\n\n".join(chunks)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")

    eval_output = None
    if args.out_eval and args.eval:
        evals = np.load(args.eval)
        etext = "\n\n".join(
            tokenizer.decode(evals[i], skip_special_tokens=False) for i in range(len(evals))
        )
        eval_output = Path(args.out_eval)
        eval_output.parent.mkdir(parents=True, exist_ok=True)
        eval_output.write_text(etext, encoding="utf-8")

    manifest = {
        "schema": "bonsai-ternary-text-corpus-manifest-v1",
        "pool": file_fingerprint(args.pool),
        "eval_pool": file_fingerprint(args.eval) if args.eval else None,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "windows": n,
        "output": file_fingerprint(out),
        "eval_output": file_fingerprint(eval_output) if eval_output else None,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    out.with_suffix(out.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[prep-corpus] {n} windows -> {out} ({len(text)} chars)", flush=True)
    if eval_output:
        print(f"[prep-corpus] eval -> {eval_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
