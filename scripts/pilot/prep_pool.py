"""Build a pinned Wikitext-103 window pool + held-out eval set.

The original pilot script hard-coded mutable dataset/tokenizer inputs and wrote
only bare ``.npy`` files.  This version records the source revisions, token
stream hash, and output fingerprints in a sidecar manifest.
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
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--dataset-revision", default=None,
                        help="immutable dataset revision/commit")
    parser.add_argument("--tokenizer", default=str(HB / "artifacts/canary/hf"))
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--out-dir", default=str(HB / "artifacts/ternary/pilot"))
    parser.add_argument("--rows", type=int, default=400000)
    parser.add_argument("--pool", type=int, default=6000)
    parser.add_argument("--eval", type=int, default=16)
    parser.add_argument("--win", type=int, default=2048)
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, revision=args.tokenizer_revision)
    dataset = load_dataset(
        args.dataset, args.dataset_config, split="train",
        revision=args.dataset_revision)
    text = "\n".join(dataset["text"][:args.rows])
    ids = tokenizer(text, return_tensors="np")["input_ids"].reshape(-1)
    needed = (args.pool + args.eval) * args.win
    if ids.size < needed:
        raise SystemExit(f"need {needed} tokens, have {ids.size}")
    pool = ids[: args.pool * args.win].reshape(args.pool, args.win)
    evals = ids[args.pool * args.win: needed].reshape(args.eval, args.win)
    pool_path = out_dir / "pool_windows.npy"
    eval_path = out_dir / "eval_windows.npy"
    np.save(pool_path, pool)
    np.save(eval_path, evals)
    manifest = {
        "schema": "bonsai-ternary-pool-manifest-v1",
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "dataset_revision": args.dataset_revision,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "rows": args.rows,
        "window": args.win,
        "pool_shape": list(pool.shape),
        "eval_shape": list(evals.shape),
        "token_count": int(ids.size),
        "token_ids_sha256": hashlib.sha256(
            ids.astype("<i8", copy=False).tobytes()).hexdigest(),
        "outputs": {
            "pool": file_fingerprint(pool_path),
            "eval": file_fingerprint(eval_path),
        },
    }
    (out_dir / "pool_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[prep] {len(dataset)} rows, {len(text)} chars, {ids.size} tokens", flush=True)
    print(f"[prep] pool {pool.shape}, eval {evals.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
