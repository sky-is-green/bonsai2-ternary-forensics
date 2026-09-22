"""Reusable, multi-region evaluation for ternary recovery runs.

This is the UI-facing evaluation surface. It exposes plain functions and a
JSON-serialisable result shape so a dashboard can render results without
importing the pilot scripts. Training holds out the *same* regions (via
`region_windows`), so the numbers are clean by construction.

Public API
----------
- `EvalRegion`               a named token range
- `make_regions(...)`        evenly spread, disjoint regions
- `region_windows(regions, seq_len)`  window ranges to exclude from training
- `region_ppl(model, ids, seq_len, region, device)`  one region's PPL
- `evaluate_regions(model, ids, seq_len, regions, device)`  structured result
- `summarize_ppls(ppls)`     pure aggregate stats (unit-testable, no torch)
- `load_checkpoint(...)`     load a `recover.py` student.pt into a wrapped model
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EvalRegion:
    """A disjoint evaluation region: `n_windows` windows of `seq_len` tokens."""

    name: str
    start_token: int
    n_windows: int

    def __post_init__(self) -> None:
        if self.start_token < 0:
            raise ValueError("start_token must be >= 0")
        if self.n_windows < 1:
            raise ValueError("n_windows must be >= 1")

    def window_range(self, seq_len: int) -> tuple[int, int]:
        start = self.start_token // seq_len
        return start, start + self.n_windows


def make_regions(total_tokens: int, seq_len: int, windows_per_region: int,
                 n_regions: int, margin_windows: int = 2) -> list[EvalRegion]:
    """Return `n_regions` disjoint regions spread evenly across the corpus.

    The final `margin_windows` windows are left unused so a region never runs
    past the end of the token stream.
    """
    if n_regions < 1:
        raise ValueError("n_regions must be >= 1")
    if windows_per_region < 1:
        raise ValueError("windows_per_region must be >= 1")
    usable = total_tokens // seq_len - margin_windows
    if usable < n_regions * windows_per_region:
        raise ValueError(
            f"{n_regions} regions x {windows_per_region} windows does not fit "
            f"in {usable} usable windows")
    block = usable // n_regions
    pad = (block - windows_per_region) // 2
    regions = []
    for i in range(n_regions):
        start_window = i * block + pad
        regions.append(EvalRegion(
            name=f"r{i}",
            start_token=start_window * seq_len,
            n_windows=windows_per_region,
        ))
    return regions


def region_windows(regions: list[EvalRegion], seq_len: int) -> tuple[tuple[int, int], ...]:
    """Window ranges `(start, stop)` for the regions, for training exclusion."""
    return tuple(r.window_range(seq_len) for r in regions)


def summarize_ppls(ppls: list[float]) -> dict:
    """Pure aggregate stats over per-region PPLs (no torch; unit-testable)."""
    if not ppls:
        raise ValueError("ppls must be non-empty")
    arr = np.asarray(ppls, dtype=np.float64)
    return {
        "n_regions": int(arr.size),
        "mean_ppl": float(arr.mean()),
        "min_ppl": float(arr.min()),
        "max_ppl": float(arr.max()),
        "std_ppl": float(arr.std(ddof=0)),
        "nats_mean": float(np.log(arr).mean()),
    }


@torch.no_grad()
def region_ppl(model, ids: np.ndarray, seq_len: int, region: EvalRegion, device) -> dict:
    """Perplexity of `model` on one region, plus its length-mean CE in nats."""
    model.eval()
    losses = []
    for k in range(region.n_windows):
        start = region.start_token + k * seq_len
        window = ids[start:start + seq_len]
        if window.size < seq_len:
            break
        batch = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(batch).logits[0, :-1].float()
        losses.append(F.cross_entropy(logits, batch[0, 1:]).item())
    if not losses:
        raise ValueError(f"region {region.name!r} produced no windows")
    nats = float(np.mean(losses))
    return {
        **asdict(region),
        "n_windows": len(losses),
        "nats": nats,
        "ppl": float(math.exp(nats)),
    }


def evaluate_regions(model, ids: np.ndarray, seq_len: int,
                     regions: list[EvalRegion], device) -> dict:
    """Evaluate `model` on every region; returns a JSON-serialisable result."""
    per_region = [region_ppl(model, ids, seq_len, r, device) for r in regions]
    return {
        "regions": per_region,
        "aggregate": summarize_ppls([r["ppl"] for r in per_region]),
    }


def add_ratios(result: dict, teacher_ppls: list[float]) -> dict:
    """Attach per-region and aggregate PPL ratios to an `evaluate_regions` result."""
    if len(teacher_ppls) != len(result["regions"]):
        raise ValueError("teacher_ppls must align with the evaluated regions")
    for region, teacher in zip(result["regions"], teacher_ppls):
        region["teacher_ppl"] = float(teacher)
        region["ratio"] = float(region["ppl"] / teacher)
    ratios = [r["ratio"] for r in result["regions"]]
    result["aggregate"].update({
        "mean_ratio": float(np.mean(ratios)),
        "min_ratio": float(np.min(ratios)),
        "max_ratio": float(np.max(ratios)),
        "std_ratio": float(np.std(ratios)),
    })
    return result


def result_markdown(result: dict, title: str = "Multi-region evaluation") -> str:
    """Render a result as a Markdown table (preview-friendly)."""
    has_ratio = "ratio" in result["regions"][0]
    head = "| region | start token | windows | PPL |"
    sep = "|---|---|---|---|"
    if has_ratio:
        head = "| region | start token | windows | PPL | teacher | ratio |"
        sep = "|---|---|---|---|---|---|"
    lines = [f"# {title}", "", head, sep]
    for r in result["regions"]:
        row = f"| {r['name']} | {r['start_token']} | {r['n_windows']} | {r['ppl']:.3f} |"
        if has_ratio:
            row += f" {r['teacher_ppl']:.3f} | {r['ratio']:.4f} |"
        lines.append(row)
    agg = result["aggregate"]
    if has_ratio:
        lines += ["", f"Aggregate ratio: mean **{agg['mean_ratio']:.4f}**, "
                      f"min {agg['min_ratio']:.4f}, max {agg['max_ratio']:.4f}, "
                      f"std {agg['std_ratio']:.4f} over {agg['n_regions']} regions."]
    else:
        lines += ["", f"Aggregate PPL: mean **{agg['mean_ppl']:.3f}**, "
                      f"min {agg['min_ppl']:.3f}, max {agg['max_ppl']:.3f}, "
                      f"std {agg['std_ppl']:.3f} over {agg['n_regions']} regions."]
    return "\n".join(lines) + "\n"


def load_checkpoint(model_dir: str, checkpoint: str, device, *, scale: str = "absmean"):
    """Load a `recover.py` `student.pt` into the ternary-wrapped base model.

    The checkpoint stores master weights only; the deployed forward is the
    deterministic STE, so re-wrapping with the same `scale` reproduces the
    trained model exactly.
    """
    from transformers import AutoModelForCausalLM

    from bonsai_forensics.recover import wrap_ternary

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    wrap_ternary(model, scale=scale)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state"])
    return model.to(device)
