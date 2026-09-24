#!/usr/bin/env python
"""Build a draft-training text set from UltraChat 200k (pinned revision).

Mirrors ProCreations' `prepare_data.py`: first long-enough unique chats in the
official `train_sft` / `test_sft` splits, rendered with the Qwen chat template,
plus a provenance manifest with per-text SHA-256.

    python scripts/pilot/prep_dspark_texts.py \
        --out artifacts/dspark/texts.jsonl --train 256 --validation 32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = "HuggingFaceH4/ultrachat_200k"
REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"


def render(messages: list[dict]) -> str:
    return "".join(
        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--train", type=int, default=256)
    ap.add_argument("--validation", type=int, default=32)
    ap.add_argument("--min-chars", type=int, default=2500)
    ap.add_argument("--revision", default=REVISION)
    args = ap.parse_args(argv)

    from datasets import load_dataset

    rows: list[dict] = []
    seen: set[str] = set()
    for split, count, label in (("train_sft", args.train, "train"),
                                ("test_sft", args.validation, "validation")):
        stream = load_dataset(REPO, revision=args.revision, split=split, streaming=True)
        kept = 0
        for index, sample in enumerate(stream):
            text = render(sample["messages"])
            if len(text) < args.min_chars:
                continue
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            rows.append({"id": f"{split}-{index:06d}", "split": label,
                         "text": text, "sha256": digest})
            kept += 1
            if kept >= count:
                break
        print(f"[prep] {split}: kept {kept}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    manifest = {
        "repo": REPO, "revision": args.revision, "license": "MIT",
        "selection": "First long-enough unique chats in separate official "
                     "train_sft/test_sft splits; rendered with the Qwen chat template.",
        "train": sum(1 for r in rows if r["split"] == "train"),
        "validation": sum(1 for r in rows if r["split"] == "validation"),
        "sha256": {r["id"]: r["sha256"] for r in rows},
    }
    manifest_path = Path(args.manifest) if args.manifest else out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[prep] wrote {len(rows)} rows -> {out}\n[prep] manifest -> {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
