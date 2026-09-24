#!/usr/bin/env python
"""Smoke / minimal trainer for a DSpark draft from collected target features.

Trains `bonsai_forensics.dspark.DSparkDraft` to predict the next token from the
target's intermediate hidden states, using the `dspark_collect` output
(`<id>.tokens.i32` + `<id>.L<layer>.hidden.f16`).  This is a proof-of-concept
trainer: it confirms the data pipeline and the draft forwards/learns, not a
production DeepSpec run.

Example (smoke, CPU):

    python scripts/pilot/dspark_train.py \
        --features /tmp/opencode/dspark-test/out --layers 5,19,33,47,61 \
        --steps 20 --hidden 128 --correction-size 128 --markov-rank 64 \
        --out /tmp/opencode/dspark-smoke.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HB))

from bonsai_forensics.dspark import DSparkDraft, count_parameters  # noqa: E402


def load_sample(feature_dir: Path, sample_id: str, layers: list[int]) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.fromfile(feature_dir / f"{sample_id}.tokens.i32", dtype=np.int32).astype(np.int64)
    hiddens = [
        np.fromfile(feature_dir / f"{sample_id}.L{layer}.hidden.f16", dtype=np.float16)
        .reshape(len(tokens), -1)
        for layer in layers
    ]
    target_hidden = np.concatenate(hiddens, axis=-1)
    return tokens, target_hidden


def feature_ids(feature_dir: Path) -> list[str]:
    """Ids with a token stream and the expected per-layer hidden files."""
    ids = []
    for meta_path in sorted(feature_dir.glob("*.json")):
        if meta_path.name == "index.jsonl":
            continue
        meta = json.loads(meta_path.read_text())
        if (feature_dir / f"{meta['id']}.tokens.i32").exists():
            ids.append(meta["id"])
    return ids


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True, help="dspark_collect output dir")
    ap.add_argument("--layers", default="5,19,33,47,61")
    ap.add_argument("--vocab-size", type=int, default=248320)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--intermediate", type=int, default=256)
    ap.add_argument("--correction-size", type=int, default=128)
    ap.add_argument("--markov-rank", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    feature_dir = Path(args.features)
    layers = [int(x) for x in args.layers.split(",") if x]

    metas = sorted(p for p in feature_dir.glob("*.json") if p.name != "index.jsonl")
    ids = feature_ids(feature_dir)
    if not ids:
        print("no samples found (looking for <id>.tokens.i32 + .L<layer>.hidden.f16)", file=sys.stderr)
        return 2
    first_meta = json.loads((feature_dir / f"{ids[0]}.json").read_text())
    target_hidden_size = int(first_meta["hidden_dim"])
    print(f"[dspark] {len(ids)} samples, target hidden {target_hidden_size}/layer, "
          f"{len(layers)} layers -> {target_hidden_size * len(layers)} features (lazy-loaded)", flush=True)

    model = DSparkDraft(
        hidden_size=args.hidden, vocab_size=args.vocab_size, num_layers=args.num_layers,
        num_heads=args.heads, num_kv_heads=args.kv_heads, head_dim=args.head_dim,
        intermediate_size=args.intermediate, target_hidden_size=target_hidden_size,
        num_target_layers=len(layers), markov_rank=args.markov_rank,
        correction_size=args.correction_size,
    ).to(args.device)
    print(f"[dspark] draft parameters: {count_parameters(model)/1e6:.2f}M", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()
    losses = []
    for step in range(args.steps):
        sample_id = ids[step % len(ids)]
        tokens_np, hidden_np = load_sample(feature_dir, sample_id, layers)
        n = min(args.seq, len(tokens_np) - 1)
        start = (step * 37) % max(1, len(tokens_np) - n - 1)
        tokens = torch.tensor(tokens_np[start:start + n + 1], dtype=torch.long,
                              device=args.device).unsqueeze(0)
        hidden = torch.tensor(hidden_np[start:start + n + 1], dtype=torch.float32,
                              device=args.device).unsqueeze(0)
        logits, confidence = model(tokens[:, :-1], hidden[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        if step % max(1, args.steps // 5) == 0 or step == args.steps - 1:
            print(f"[dspark] step {step:4d} loss {loss.item():.4f} "
                  f"mean-conf {confidence.mean().item():+.3f}", flush=True)

    report = {
        "samples": len(ids), "steps": args.steps,
        "parameters": count_parameters(model),
        "loss_first": losses[0], "loss_last": losses[-1],
        "loss_mean_last5": float(np.mean(losses[-5:])),
        "smoke_ok": bool(losses[-1] < losses[0]),
    }
    print("[dspark] report:", json.dumps(report, indent=2), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(),
                    "config": vars(args),
                    "report": report}, args.out)
        print(f"[dspark] saved {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
