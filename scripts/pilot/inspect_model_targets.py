"""Inspect a checkpoint's architecture and ternary-target coverage without weights.

This is the pre-flight check for cross-architecture experiments.  It downloads
only configuration files (unless ``--local-files-only`` is supplied), builds the
model on the ``meta`` device, and reports the exact selected linear modules,
parameter coverage, and effective Hadamard block sizes.  No model weights are
materialised and no GPU is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

HB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HB))

from bonsai_forensics.targets import target_coverage  # noqa: E402


def _load_meta_model(model_dir: str, revision: str | None, trust_remote_code: bool,
                     local_files_only: bool):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    kwargs = {
        "revision": revision,
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    config = AutoConfig.from_pretrained(model_dir, **kwargs)
    # Qwen3.8's public config is a multimodal Qwen3_5 config, not a
    # Qwen3_5TextConfig.  Build the conditional class on meta so the language
    # tower can still be inspected; no weights are read.
    conditional = "ConditionalGeneration" in " ".join(
        str(x) for x in (getattr(config, "architectures", None) or ()))
    if conditional:
        model = AutoModelForImageTextToText.from_config(config)
    else:
        model = AutoModelForCausalLM.from_config(config)
    return config, model


def inspect(args) -> dict:
    with torch.device("meta"):
        config, model = _load_meta_model(
            args.model, args.revision, args.trust_remote_code, args.local_files_only)
    report = {
        "schema": "bonsai-ternary-target-report-v1",
        "model": args.model,
        "revision": args.revision or "main (unpinned; resolve before a real run)",
        "model_type": getattr(config, "model_type", None),
        "architectures": list(getattr(config, "architectures", None) or ()),
        "config_class": type(config).__name__,
        "config_sha256": hashlib.sha256(
            config.to_json_string().encode("utf-8")).hexdigest(),
        "model_class": type(model).__name__,
        "meta_parameter_count": sum(p.numel() for p in model.parameters()),
        "meta_linear_parameter_count": sum(
            m.weight.numel() for m in model.modules()
            if isinstance(m, torch.nn.Linear)
        ),
    }
    report["targets"] = target_coverage(
        model, profile=args.profile, include_lm_head=args.include_lm_head,
        suffixes=(tuple(s.strip() for s in args.target_suffixes.split(",") if s.strip())
                  if args.target_suffixes else None))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="Hugging Face model ID or local checkpoint directory")
    parser.add_argument("--revision", default=None,
                        help="immutable model revision/commit SHA (recommended)")
    parser.add_argument("--profile", default="auto",
                        help="target profile; default infers from config.model_type")
    parser.add_argument("--target-suffixes", default=None,
                        help="comma-separated module suffixes overriding the profile")
    parser.add_argument("--include-lm-head", action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--allow-target-count-mismatch", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--out", default=None, help="write JSON report to this path")
    args = parser.parse_args(argv)
    report = inspect(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if (not report["targets"]["target_count_ok"]
            and not args.allow_target_count_mismatch):
        print("target inventory mismatch; refusing preflight", file=sys.stderr)
        return 2
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
