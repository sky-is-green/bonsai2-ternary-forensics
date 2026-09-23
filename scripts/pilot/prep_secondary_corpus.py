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
from pathlib import Path

HB = Path(__file__).resolve().parents[2]
CANARY = Path.home() / "Desktop/work/hivebench/artifacts/ternary/canary/hf"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HB / "artifacts/ternary/pilot/fineweb_3000.txt"))
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--config", default="sample-10BT")
    ap.add_argument("--chars", type=int, default=27_000_000, help="approximate text size to collect")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset(args.dataset, name=args.config, split="train", streaming=True)
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

    tok = AutoTokenizer.from_pretrained(CANARY)
    ids = tok(out.read_text(encoding="utf-8"), return_tensors="np")["input_ids"].reshape(-1)
    print(f"[prep2] {out} ({out.stat().st_size / 1e6:.1f} MB, {ids.size} tokens, "
          f"{ids.size // 512} windows of 512)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
