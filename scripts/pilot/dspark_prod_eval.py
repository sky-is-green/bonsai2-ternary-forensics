#!/usr/bin/env python
"""Measure a trained DSpark draft's next-block top-1/top-5 accuracy + NLL.

This is the ceiling for speculative acceptance: the runtime accepts a draft
token only when its argmax equals the target's.  Reports per block-slot
accuracy so a low accept rate can be attributed to the draft itself vs the
runtime's drafting loop.

    python scripts/pilot/dspark_prod_eval.py --checkpoint artifacts/dspark/prod-run-6k.pt \
        --features artifacts/dspark/features --device cuda:0 --samples 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HB))

from bonsai_forensics.dspark_prod import DSparkProdConfig, DSparkProdDraft  # noqa: E402

_TRAIN = Path(__file__).resolve().parent / "dspark_prod_train.py"


def _load_train_mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location("dspark_prod_train_eval", _TRAIN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--layers", default="5,19,33,47,61")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16")
    ap.add_argument("--split", default="test", help="feature id prefix to evaluate")
    args = ap.parse_args(argv)

    t = _load_train_mod()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    c = ck["config"]
    # trust the actual fc input width over the saved (possibly inferred) n_capture
    fc_w = ck["model"].get("fc.weight")
    n_capture = int(fc_w.shape[1] // int(c["hidden"])) if fc_w is not None else int(c["n_capture"])
    cfg = DSparkProdConfig(
        hidden_size=int(c["hidden"]), vocab_size=int(c["vocab"]),
        num_layers=int(c["layers_count"]), num_heads=int(c["heads"]),
        num_kv_heads=int(c["kv_heads"]), head_dim=int(c["head_dim"]),
        intermediate_size=int(c["intermediate"]), n_capture=n_capture,
        markov_rank=int(c["markov_rank"]), block_size=int(c["block_size"]),
        mask_token_id=int(c["vocab"]) - 1, confidence_head=False,
    )
    model = DSparkProdDraft(cfg)
    model.load_state_dict(ck["model"])
    if args.dtype == "bf16":
        model = model.to(torch.bfloat16)
    model = model.to(args.device).eval()

    feature_dir = Path(args.features)
    ids = [i for i in t._feature_ids(feature_dir) if i.startswith(args.split)] or t._feature_ids(feature_dir)
    layers = [int(x) for x in args.layers.split(",") if x]
    bs = cfg.block_size
    top1 = np.zeros(bs); top5 = np.zeros(bs); seen = np.zeros(bs); nll = 0.0; total = 0
    with torch.no_grad():
        for si, sample_id in enumerate(ids[: args.samples]):
            tokens, taps = t._load_sample(feature_dir, sample_id, layers)
            for start in range(max(args.window, 1), len(tokens) - bs, bs):
                ctx, draft, labels, prev, pos = t.make_block(
                    tokens, taps, start, bs, args.window, cfg.mask_token_id)
                ctx_t = torch.from_numpy(np.ascontiguousarray(ctx)).to(
                    device=args.device, dtype=model.embed_tokens.weight.dtype).unsqueeze(0)
                logits, _ = model(torch.from_numpy(draft).unsqueeze(0).to(args.device), ctx_t,
                                  position_ids=torch.from_numpy(pos).unsqueeze(0).to(args.device),
                                  prev_tokens=torch.from_numpy(prev).unsqueeze(0).to(args.device))
                logits = logits[0].float()
                lab = torch.from_numpy(labels).to(args.device)
                nll += float(torch.nn.functional.cross_entropy(logits, lab, reduction="sum"))
                total += bs
                a = logits.argmax(-1)
                for j in range(bs):
                    seen[j] += 1
                    top1[j] += int(a[j].item() == labels[j])
                    top5[j] += int(labels[j] in logits[j].topk(5).indices.tolist())
    report = {
        "checkpoint": args.checkpoint, "samples": len(ids[: args.samples]),
        "blocks": int(seen[0]), "nll": nll / max(total, 1), "ppl": float(np.exp(nll / max(total, 1))),
        "top1_overall": float((top1.sum()) / max(seen.sum(), 1)),
        "top5_overall": float((top5.sum()) / max(seen.sum(), 1)),
        "top1_by_slot": [float(top1[j] / max(seen[j], 1)) for j in range(bs)],
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
