"""Build a secondary-corpus text file for the corpus control.

docs/SCALING-PROTOCOL.md §1 asks for a second, differently-distributed corpus to
confirm the retention-vs-size trend is not corpus-specific. This mirrors the
WikiText primary (`wikitext_3000.txt`): a plain-text corpus that `rmd_kd`
re-tokenizes on load, sized to the same order (~26 MB / ~6 M tokens).

    python scripts/pilot/prep_secondary_corpus.py \
        --out artifacts/ternary/pilot/fineweb_3000.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HB))
from bonsai_forensics.provenance import file_fingerprint  # noqa: E402

CANARY = Path(__file__).resolve().parents[2] / "artifacts" / "ternary" / "canary" / "hf"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HB / "artifacts/ternary/pilot/fineweb_3000.txt"))
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--dataset-revision", default=None,
                    help="immutable dataset revision/commit")
    ap.add_argument("--config", default="sample-10BT")
    ap.add_argument("--tokenizer", default=str(CANARY))
    ap.add_argument("--tokenizer-revision", default=None)
    ap.add_argument("--chars", type=int, default=27_000_000, help="approximate text size to collect")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset(
        args.dataset, name=args.config, split="train", streaming=True,
        revision=args.dataset_revision)
    parts: list[str] = []
    total = 0
    for row in ds:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        parts.append(text)
        total += len(text) + 2
        if total >= args.chars:
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(parts), encoding="utf-8")

    tok = AutoTokenizer.from_pretrained(
        args.tokenizer, revision=args.tokenizer_revision)
    ids = tok(out.read_text(encoding="utf-8"), return_tensors="np")["input_ids"].reshape(-1)
    manifest = {
        "schema": "bonsai-ternary-corpus-manifest-v1",
        "dataset": args.dataset,
        "dataset_config": args.config,
        "dataset_revision": args.dataset_revision,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "output": file_fingerprint(out),
        "token_count": int(ids.size),
        "token_ids_sha256": hashlib.sha256(
            ids.astype("<i8", copy=False).tobytes()).hexdigest(),
    }
    out.with_suffix(out.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[prep2] {out} ({out.stat().st_size / 1e6:.1f} MB, {ids.size} tokens, "
          f"{ids.size // 512} windows of 512)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
