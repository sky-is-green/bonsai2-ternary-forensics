"""KLD vs the FP teacher on a locked evaluation split.

Historical versions of this script silently selected the first corpus chunks,
which were inside the RMD training pool for the WikiText runs.  This version
requires the checkpoint's ``eval-regions.json`` by default and records the
selected token ranges.  ``--allow-training-data`` exists only for explicitly
reproducing a contaminated historical diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HB))

from bonsai_forensics.modeling import load_text_causal_lm  # noqa: E402
from bonsai_forensics.provenance import file_fingerprint  # noqa: E402
from bonsai_forensics.rotation import load_sign_manifest  # noqa: E402
from bonsai_forensics.targets import infer_profile  # noqa: E402
from scripts.pilot.rmd_kd import wrap_rotated  # noqa: E402

LOG_BASE_FLOOR = -16.0


def _eval_windows(ids: np.ndarray, args, checkpoint: Path):
    """Return windows and a JSON-safe description of the selected split."""
    region_path = Path(args.regions) if args.regions else checkpoint.parent / "eval-regions.json"
    if region_path.is_file():
        payload = json.loads(region_path.read_text(encoding="utf-8"))
        regions = payload.get("regions", payload)
        windows = []
        ranges = []
        for region in regions:
            start = int(region["start_token"])
            n_windows = int(region["n_windows"])
            region_seq = int(args.region_seq)
            stop = start + n_windows * region_seq
            for left in range(start, stop, args.ctx):
                right = left + args.ctx
                if right > stop:
                    raise SystemExit(
                        f"region {region.get('name', '?')}: window {left}:{right} crosses "
                        f"the region end {stop}; ctx={args.ctx} must divide the region span "
                        f"{n_windows * region_seq}")
                if right > len(ids):
                    raise SystemExit(
                        f"region {region.get('name', '?')} exceeds tokenized corpus: "
                        f"{right} > {len(ids)}")
                windows.append(ids[left:right])
                ranges.append({"start": left, "stop": right,
                               "name": region.get("name", "region")})
        if not windows:
            raise SystemExit(f"no evaluation windows in {region_path}")
        selected = np.stack(windows[: args.chunks])
        return selected, {
            "split": "locked_eval_regions",
            "source": str(region_path),
            "region_seq": args.region_seq,
            "ranges": ranges[: args.chunks],
            "source_sha256": hashlib.sha256(region_path.read_bytes()).hexdigest(),
        }
    if not args.allow_training_data:
        raise SystemExit(
            f"no evaluation-region manifest beside {checkpoint}; pass --regions or "
            "--allow-training-data for an explicitly labelled historical diagnostic")
    n = (len(ids) // args.ctx) * args.ctx
    selected = ids[: min(n, args.chunks * args.ctx)].reshape(-1, args.ctx)
    return selected, {
        "split": "TRAINING_DATA_CONTAMINATED",
        "source": str(args.corpus),
        "ranges": [{"start": 0, "stop": int(selected.size), "name": "prefix"}],
        "source_sha256": None,
    }


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-revision", default=None)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--regions", default=None,
                    help="eval-regions.json; defaults to checkpoint sibling")
    ap.add_argument("--region-seq", type=int, default=512,
                    help="sequence length used by the training run's region manifest")
    ap.add_argument("--chunks", type=int, default=50)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--rot-seed", type=int, default=None)
    ap.add_argument("--rot-block", type=int, default=None)
    ap.add_argument("--target-profile", default=None)
    ap.add_argument("--target-suffixes", default=None)
    ap.add_argument("--include-lm-head", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--rotation-mode", choices=("residual", "input"), default=None)
    ap.add_argument("--signs-manifest", default=None)
    ap.add_argument("--allow-training-data", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--teacher-device", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    device = torch.device(args.device)
    teacher_device = torch.device(args.teacher_device or args.device)
    checkpoint = Path(args.checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    manifest = payload.get("manifest") or {}
    config = dict(payload.get("config") or {})
    model_revision = args.model_revision or manifest.get("model", {}).get("revision")
    tok = AutoTokenizer.from_pretrained(
        args.model_dir, revision=model_revision,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only)
    ids = tok(Path(args.corpus).read_text(encoding="utf-8"),
             return_tensors="np")["input_ids"].reshape(-1)
    ids, split = _eval_windows(np.asarray(ids), args, checkpoint)
    print(f"[kld] {ids.shape[0]} chunks x {args.ctx} tokens split={split['split']}", flush=True)

    teacher = load_text_causal_lm(
        args.model_dir, revision=model_revision, dtype=torch.bfloat16,
        device=teacher_device, trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only)
    teacher.eval()
    student = load_text_causal_lm(
        args.model_dir, revision=model_revision, dtype=torch.bfloat16,
        device=device, trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only)

    profile_name = args.target_profile or config.get("target_profile", "auto")
    profile = infer_profile(student, profile_name)
    suffix_value = args.target_suffixes if args.target_suffixes is not None else config.get("target_suffixes")
    suffixes = tuple(s.strip() for s in suffix_value.split(",") if s.strip()) if suffix_value else None
    include_head = args.include_lm_head if args.include_lm_head is not None else config.get("include_lm_head")
    rotation_mode = args.rotation_mode or config.get("rotation_mode", "residual")
    rot_seed = args.rot_seed if args.rot_seed is not None else config.get("rot_seed")
    if rot_seed is None or (not manifest and rot_seed == 0):
        # Legacy rmd configs stored the parser default 0 even though the old
        # implementation interpreted 0 as "follow seed".
        rot_seed = config.get("seed", args.seed)
    block = args.rot_block if args.rot_block is not None else config.get("rot_block")
    signs_path = args.signs_manifest or config.get("signs_manifest")
    sign_sets = load_sign_manifest(signs_path) if signs_path else None
    wrap_rotated(
        student, seed=int(rot_seed), ste=True,
        learn_scale=bool(config.get("learn_scale", False)), block=block,
        suffixes=suffixes, profile=profile, include_lm_head=include_head,
        rotation_mode=rotation_mode, sign_sets=sign_sets)
    missing, unexpected = student.load_state_dict(payload["state"], strict=False)
    print(f"[kld] loaded step={payload.get('step')} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    student.eval()

    kld_sum = nll_s = nll_t = 0.0
    same_top = 0
    dp_sum = dp2_sum = 0.0
    count = 0
    SLICE = 256
    with torch.inference_mode():
        for i in range(ids.shape[0]):
            batch = torch.tensor(ids[i], dtype=torch.long).unsqueeze(0).to(device)
            tl = teacher(batch.to(teacher_device)).logits[0, :-1]
            sl = student(batch).logits[0, :-1]
            targets = batch[0, 1:]
            for t0 in range(0, tl.shape[0], SLICE):
                tb = tl[t0:t0 + SLICE].float()
                sb = sl[t0:t0 + SLICE].float()
                tg = targets[t0:t0 + SLICE]
                tb_target = tg.to(tb.device)
                sb_target = tg.to(sb.device)
                logpb = F.log_softmax(tb, dim=-1)
                logpm = F.log_softmax(sb, dim=-1)
                mask = logpb > LOG_BASE_FLOOR
                pb = logpb.exp() * mask
                kld_sum += float((pb * (logpb - logpm)).sum())
                nll_t += float((-logpb.gather(-1, tb_target.unsqueeze(-1)).squeeze(-1)).sum())
                nll_s += float((-logpm.gather(-1, sb_target.unsqueeze(-1)).squeeze(-1)).sum())
                same_top += int((tb.argmax(-1) == sb.argmax(-1)).sum())
                dp = (logpm.gather(-1, sb_target.unsqueeze(-1)).exp()
                      - logpb.gather(-1, tb_target.unsqueeze(-1)).exp()).squeeze(-1)
                dp_sum += float(dp.sum())
                dp2_sum += float((dp ** 2).sum())
                count += tg.numel()
            print(f"[kld] chunk {i+1}/{ids.shape[0]} running mean KLD {kld_sum/count:.5f}", flush=True)

    token_bytes = np.asarray(ids, dtype="<i8").tobytes()
    report = {
        "checkpoint": args.checkpoint,
        "model": args.model_dir,
        "model_revision": model_revision,
        "corpus": file_fingerprint(args.corpus),
        "split": split,
        "selected_token_ids_sha256": hashlib.sha256(token_bytes).hexdigest(),
        "chunks": int(ids.shape[0]),
        "ctx": args.ctx,
        "tokens": count,
        "kld_mean": kld_sum / count,
        "ppl_base": math.exp(nll_t / count),
        "ppl_model": math.exp(nll_s / count),
        "ppl_ratio": math.exp(nll_s / count - nll_t / count),
        "same_top_p": same_top / count,
        "mean_delta_p": dp_sum / count,
        "rms_delta_p": math.sqrt(dp2_sum / count),
        "checkpoint_manifest_present": bool(manifest),
        "missing_state_keys": list(missing),
        "unexpected_state_keys": list(unexpected),
    }
    print("\n[kld] " + json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                                   for k, v in report.items()}, indent=1))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[kld] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
