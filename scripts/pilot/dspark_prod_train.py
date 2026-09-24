#!/usr/bin/env python
"""Smoke/first trainer for the fork-compatible DSpark trunk (`dspark_prod`).

Block-diffusion objective matching `src/models/dspark.cpp`:

    context = target taps of the positions before the block  ->  dspark.fc ->
              dspark.hidden_norm  (the static target context)
    draft   = block_size ``mask_token_id`` rows
    logits  = trunk(context, draft) + markov_bias(previous token)
    loss    = CE(logits[j], true_token[start + j])

Synthetic mode (default) validates the loop at tiny dims on CPU.  `--features`
streams the real Bonsai-2 taps (the real run is necessarily hidden 5120, since
the runtime's ``fc`` maps ``n_capture*n_embd -> n_embd``).

    # CPU loop smoke
    python scripts/pilot/dspark_prod_train.py --steps 30 --out /tmp/dspark-smoke.pt
    # real features (GPU)
    python scripts/pilot/dspark_prod_train.py --features artifacts/dspark/features \
        --layers 5,19,33,47,61 --hidden 5120 --intermediate 9728 --steps 200 \
        --freeze-embed --device cuda:0 --out artifacts/dspark/prod-smoke.pt
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

from bonsai_forensics.dspark_prod import DSparkProdConfig, DSparkProdDraft, count_parameters  # noqa: E402


def make_block(tokens: np.ndarray, taps: np.ndarray, start: int, block_size: int,
               window: int, mask_token_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One block-diffusion training example.

    Returns ``(ctx_taps, draft_tokens, labels, prev_tokens, position_ids)`` where
    the context is the last ``window`` tap rows before ``start`` and ``prev[j]``
    is the token the Markov head conditions slot ``j`` on (the block anchor for
    slot 0, else the preceding slot).
    """
    ctx_start = max(0, start - window)
    ctx_taps = taps[ctx_start:start]
    draft = np.full(block_size, mask_token_id, dtype=np.int64)
    labels = tokens[start:start + block_size]
    prev = np.empty(block_size, dtype=np.int64)
    for j in range(block_size):
        p = start + j - 1
        prev[j] = tokens[p] if p >= 0 else mask_token_id
    positions = np.arange(ctx_start, start + block_size, dtype=np.int64)
    return ctx_taps, draft, labels, prev, positions


def _feature_ids(feature_dir: Path) -> list[str]:
    ids = []
    for meta_path in sorted(feature_dir.glob("*.json")):
        if meta_path.name == "index.jsonl":
            continue
        meta = json.loads(meta_path.read_text())
        if (feature_dir / f"{meta['id']}.tokens.i32").exists():
            ids.append(meta["id"])
    return ids


def _load_sample(feature_dir: Path, sample_id: str, layers: list[int]) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.fromfile(feature_dir / f"{sample_id}.tokens.i32", dtype=np.int32).astype(np.int64)
    hiddens = [np.fromfile(feature_dir / f"{sample_id}.L{layer}.hidden.f16", dtype=np.float16)
               .reshape(len(tokens), -1) for layer in layers]
    return tokens, np.concatenate(hiddens, axis=-1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=None, help="dspark_collect output dir (else synthetic)")
    ap.add_argument("--exclude-prefix", default="", help="comma-separated feature-id prefixes to hold out")
    ap.add_argument("--layers", default="5,19,33,47,61")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers-count", type=int, default=2)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--kv-heads", type=int, default=1)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--intermediate", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--n-capture", type=int, default=2)
    ap.add_argument("--markov-rank", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=4)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--seq", type=int, default=128, help="synthetic token length")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch-blocks", type=int, default=1, help="blocks sampled per optimizer step")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    ap.add_argument("--optimizer", choices=("adam", "adafactor"), default="adam")
    ap.add_argument("--init-from", default=None,
                    help="dspark_target_init.pt with token_embd/output to copy in")
    ap.add_argument("--freeze-embed", action="store_true")
    ap.add_argument("--freeze-head", action="store_true")
    ap.add_argument("--overfit", action="store_true",
                    help="reuse one fixed real block (sanity: loss must plunge)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    mask_token_id = args.vocab - 1
    feature_layers = [int(x) for x in args.layers.split(",") if x]
    # real taps are len(layers) * target_hidden wide; fc input must be n_capture*hidden
    n_capture = len(feature_layers) if args.features else args.n_capture
    args.n_capture = n_capture  # persist the inferred value in the saved config
    cfg = DSparkProdConfig(
        hidden_size=args.hidden, vocab_size=args.vocab, num_layers=args.layers_count,
        num_heads=args.heads, num_kv_heads=args.kv_heads, head_dim=args.head_dim,
        intermediate_size=args.intermediate, n_capture=n_capture,
        markov_rank=args.markov_rank, block_size=args.block_size,
        mask_token_id=mask_token_id, confidence_head=False,
    )
    # Build directly in the target dtype: constructing fp32 then casting needs
    # ~2x the model in host RAM and OOMs a 30 GB box on the 5-layer/ff4096 draft.
    prev_dtype = torch.get_default_dtype()
    if args.dtype == "bf16":
        torch.set_default_dtype(torch.bfloat16)
    try:
        model = DSparkProdDraft(cfg)
    finally:
        torch.set_default_dtype(prev_dtype)
    if args.init_from:
        init = torch.load(args.init_from, map_location="cpu", weights_only=False, mmap=True)
        with torch.no_grad():
            if "token_embd" in init:
                model.embed_tokens.weight.copy_(init["token_embd"].to(model.embed_tokens.weight.dtype))
            if "output" in init and not cfg.tie_word_embeddings:
                model.lm_head.weight.copy_(init["output"].to(model.lm_head.weight.dtype))
        del init
        print(f"[train] initialised from {args.init_from}", flush=True)
    model = model.to(args.device)
    if args.freeze_embed:
        model.embed_tokens.weight.requires_grad_(False)
    if args.freeze_head:
        model.lm_head.weight.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[train] params={count_parameters(model)/1e6:.2f}M trainable={sum(p.numel() for p in trainable)/1e6:.2f}M", flush=True)

    real = None
    if args.features:
        feature_dir = Path(args.features)
        ids = _feature_ids(feature_dir)
        excl = tuple(x for x in args.exclude_prefix.split(",") if x)
        if excl:
            ids = [i for i in ids if not i.startswith(excl)]
        if not ids:
            raise SystemExit(f"no feature samples under {feature_dir} after excluding {excl}")
        real = (feature_dir, ids, [int(x) for x in args.layers.split(",") if x])
        print(f"[train] {len(ids)} real feature samples", flush=True)

    rng = np.random.default_rng(args.seed)
    optimizer = (torch.optim.Adafactor(trainable, lr=args.lr)
                 if args.optimizer == "adafactor" else torch.optim.Adam(trainable, lr=args.lr))
    model.train()
    losses: list[float] = []
    fixed_block = None
    batch = max(1, args.batch_blocks)
    for step in range(args.steps):
        if args.overfit and fixed_block is not None:
            ctxb, draftb, labelb, prevb, posb = ([fixed_block[0]], [fixed_block[1]],
                                                [fixed_block[2]], [fixed_block[3]], [fixed_block[4]])
        else:
            ctxb, draftb, labelb, prevb, posb = [], [], [], [], []
            attempts = 0
            while len(ctxb) < batch and attempts < batch * 4:
                attempts += 1
                if real is not None:
                    feature_dir, ids, layers = real
                    tokens, taps = _load_sample(feature_dir, ids[int(rng.integers(0, len(ids)))], layers)
                    if len(tokens) <= args.window + args.block_size:
                        continue
                else:
                    tokens = rng.integers(0, args.vocab, size=args.seq, dtype=np.int64)
                    taps = rng.standard_normal((args.seq, cfg.n_embd_cap)).astype(np.float32)
                if args.overfit:
                    start = max(1, min(args.window, len(tokens) - args.block_size - 1))
                else:
                    lo = min(args.window, len(tokens) - args.block_size - 1)
                    start = int(rng.integers(max(1, lo), len(tokens) - args.block_size))
                ctx, draft, labels, prev, pos = make_block(
                    tokens, taps, start, args.block_size, args.window, mask_token_id)
                if ctx.shape[0] != args.window:  # keep a fixed context width for batching
                    continue
                ctxb.append(ctx); draftb.append(draft); labelb.append(labels)
                prevb.append(prev); posb.append(pos)
            if not ctxb:
                continue
            if args.overfit:
                fixed_block = (ctxb[0], draftb[0], labelb[0], prevb[0], posb[0])
        ctx_t = torch.from_numpy(np.stack(ctxb)).to(
            device=args.device, dtype=model.embed_tokens.weight.dtype)
        draft_t = torch.from_numpy(np.stack(draftb)).to(args.device)
        labels_t = torch.from_numpy(np.stack(labelb)).to(args.device)
        prev_t = torch.from_numpy(np.stack(prevb)).to(args.device)
        pos_t = torch.from_numpy(np.stack(posb)).to(args.device)
        logits, _ = model(draft_t, ctx_t, position_ids=pos_t, prev_tokens=prev_t)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels_t.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
        if step % max(1, args.steps // 5) == 0 or step == args.steps - 1:
            print(f"[train] step {step:4d} loss {loss.item():.4f}", flush=True)

    report = {
        "steps": args.steps, "parameters": count_parameters(model),
        "trainable": sum(p.numel() for p in trainable),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_mean_last5": float(np.mean(losses[-5:])) if losses else None,
        "loss_fell": bool(len(losses) >= 2 and losses[-1] < losses[0]),
    }
    print("[train] report:", json.dumps(report, indent=2), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "config": vars(args), "report": report}, args.out)
        print(f"[train] saved {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
