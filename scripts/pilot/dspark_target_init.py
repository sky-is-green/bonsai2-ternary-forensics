#!/usr/bin/env python
"""Extract the Bonsai-2 target's token embedding + output head for DSpark init.

The draft's own ``token_embd``/``output`` start from the target's *deployed*
(dequantized, un-rotated) weights so the frozen/shared head can predict before
the trunk has learned anything.  ``prism_loader.recover`` gives ``W_hf = D @ R``.

Saves ``{token_embd: bf16[vocab,hidden], output: bf16[vocab,hidden]}``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HB))

from bonsai_forensics.prism_loader import load_prism_gguf  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", required=True, help="Prism PQ2_0 Bonsai-2 GGUF")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    model = load_prism_gguf(args.gguf)
    out: dict[str, torch.Tensor] = {}
    for gguf_name, key in (("token_embd.weight", "token_embd"), ("output.weight", "output")):
        if gguf_name not in model.header.tensors:
            print(f"[init] {gguf_name} absent; skipping", flush=True)
            continue
        print(f"[init] recovering {gguf_name} ...", flush=True)
        weights = model.recover(gguf_name)  # float32 (vocab, hidden)
        out[key] = torch.from_numpy(weights).to(torch.bfloat16)
        del weights
        print(f"[init]   -> {key} {tuple(out[key].shape)} {out[key].dtype}", flush=True)
    if "token_embd" not in out:
        raise SystemExit("token_embd.weight missing from target GGUF")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"[init] saved {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
