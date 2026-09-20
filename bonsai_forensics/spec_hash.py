"""Recompute the canonical TBR spec hash from the machine-readable block.

`spec.md` §6 carries exactly one fenced ```json block; consumers (every module
pins `SPEC_SHA256`) embed the sha256 of that block serialized with
`json.dumps(constants, sort_keys=True, separators=(",", ":"))`.

Usage::

    python -m bonsai_forensics.spec_hash            # print the hash
    python -m bonsai_forensics.spec_hash --check    # verify the spec literal

This module exists because spec v1.0 referenced it without shipping it (T22).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

SPEC_PATH = Path(__file__).resolve().parent / "spec.md"


def constants(path: str | Path = SPEC_PATH) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    blocks = re.findall(r"```json\n(.*?)\n```", text, re.S)
    if len(blocks) != 1:
        raise ValueError(f"spec must contain exactly one fenced json block, found {len(blocks)}")
    return json.loads(blocks[0])


def canonical_sha256(path: str | Path = SPEC_PATH) -> str:
    canon = json.dumps(constants(path), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the literal in spec.md")
    path = SPEC_PATH
    args = parser.parse_args(argv)
    digest = canonical_sha256(path)
    if args.check:
        text = path.read_text(encoding="utf-8")
        expected = re.search(r'SPEC_SHA256 = "([0-9a-f]{64})"', text)
        if not expected or expected.group(1) != digest:
            print(f"spec hash mismatch: doc={expected.group(1) if expected else None} computed={digest}")
            return 1
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
