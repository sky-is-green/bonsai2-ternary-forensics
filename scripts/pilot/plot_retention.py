"""Retention vs scale — Prism's published curve vs our ladder, with an
absolute-PPL-delta view and a per-corpus split.

Why per-corpus: the ratio `student_ppl / teacher_ppl` is not comparable across
corpora when the FP teacher itself differs (a weaker teacher is easier to
match). tinyshakespeare gives teacher PPL 49-73 (out of distribution);
WikiText gives 22-28. The same recipe shows opposite scale trends on the two,
which is the finding, so they must be plotted separately.

Prism points are transcribed from the released whitepapers
(see docs/RETENTION-VS-SCALE.md).

    python scripts/pilot/plot_retention.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HB = Path(__file__).resolve().parents[2]

PRISM = {
    "6-bench": [(1.7, 0.878), (4.0, 0.916), (8.0, 0.952)],
    "10-bench": [(1.7, 0.851), (4.0, 0.920), (8.0, 0.917)],
    "27B": [(27.0, 0.946)],
    "27B (Bonsai 2)": [(27.0, 0.982)],
}

# label -> (report dir, params_B, corpus)
RUNS = {
    "shakespeare 0.6B (blk1024)": (HB / "artifacts/rmd/ladder-0.6B", 0.6, "tinyshakespeare"),
    "shakespeare 1.7B (blk1024)": (HB / "artifacts/rmd/ladder-1.7B", 1.7, "tinyshakespeare"),
    "shakespeare 1.7B (blk512)": (HB / "artifacts/rmd/ste-rotate-10k-block512", 1.7, "tinyshakespeare"),
    "wiki 0.6B (blk1024)": (HB / "artifacts/rmd/wiki-0.6B", 0.6, "wikitext"),
    "wiki 1.7B (blk1024)": (HB / "artifacts/rmd/wiki-1.7B", 1.7, "wikitext"),
}

COLORS = {"wikitext": "seagreen", "tinyshakespeare": "crimson"}


def load(path: Path):
    rep = path / "rmd-report.json"
    if not rep.exists():
        return None
    d = json.loads(rep.read_text())
    best = min(d["events"], key=lambda e: e["aggregate"]["mean_ratio"])
    fin = d["events"][-1]

    def delta(ev):
        return sum(r["ppl"] - r["teacher_ppl"] for r in ev["regions"]) / len(ev["regions"])

    return dict(
        teacher_ppl=d.get("teacher_ppl"),
        best_step=best["step"],
        best_ratio=best["aggregate"]["mean_ratio"],
        best_retention=1.0 / best["aggregate"]["mean_ratio"],
        best_delta=delta(best),
        final_ratio=fin["aggregate"]["mean_ratio"],
        final_delta=delta(fin),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HB / "docs/figures/retention-vs-scale.png"))
    ap.add_argument("--summary", default=str(HB / "artifacts/rmd/ladder-summary.md"))
    args = ap.parse_args()

    rows = {}
    for label, (path, params, corpus) in RUNS.items():
        r = load(path)
        if r:
            r["params"] = params
            r["corpus"] = corpus
            rows[label] = r

    # --- table ---
    lines = ["# Size ladder — ratio and absolute-delta views", "",
             "| run | corpus | params | teacher PPL | best step | best ratio | "
             "retention | ΔPPL (best) | final ratio | final ΔPPL |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for label, r in sorted(rows.items(), key=lambda kv: (kv[1]["corpus"], kv[1]["params"])):
        lines.append(
            f"| {label} | {r['corpus']} | {r['params']}B | {r['teacher_ppl']:.2f} | "
            f"{r['best_step']} | {r['best_ratio']:.4f} | {r['best_retention']*100:.1f}% | "
            f"{r['best_delta']:.2f} | {r['final_ratio']:.4f} | {r['final_delta']:.2f} |")
    table = "\n".join(lines) + "\n"
    Path(args.summary).write_text(table)
    print(table)

    # --- figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    for label, pts in PRISM.items():
        ax1.plot([p for p, _ in pts], [r for _, r in pts], "o--",
                 label=f"Prism {label}", alpha=0.7)
    for corpus in ("wikitext", "tinyshakespeare"):
        pts = sorted((r["params"], r["best_retention"])
                     for r in rows.values() if r["corpus"] == corpus)
        if pts:
            ax1.plot([p for p, _ in pts], [v for _, v in pts], "s-",
                     color=COLORS[corpus], lw=2, label=f"ours — {corpus} (PPL ratio)")
    ax1.axhline(0.97, color="grey", ls=":", lw=1)
    ax1.text(0.62, 0.973, "mission bar (27B)", color="grey", fontsize=8)
    ax1.set_xscale("log")
    ax1.set_xticks([0.6, 1.7, 4, 8, 27])
    ax1.set_xticklabels(["0.6B", "1.7B", "4B", "8B", "27B"])
    ax1.set_xlabel("parameters")
    ax1.set_ylabel("retention (1 / ratio)")
    ax1.set_ylim(0.2, 1.02)
    ax1.grid(alpha=0.3)
    ax1.set_title("Retention vs scale — direction depends on corpus")
    ax1.legend(fontsize=7, loc="lower right")

    for corpus in ("wikitext", "tinyshakespeare"):
        pts = sorted((r["params"], r["best_delta"])
                     for r in rows.values() if r["corpus"] == corpus)
        if pts:
            ax2.plot([p for p, _ in pts], [v for _, v in pts], "s-",
                     color=COLORS[corpus], lw=2, label=corpus)
    ax2.set_xscale("log")
    ax2.set_xticks([0.6, 1.7, 4, 8, 27])
    ax2.set_xticklabels(["0.6B", "1.7B", "4B", "8B", "27B"])
    ax2.set_xlabel("parameters")
    ax2.set_ylabel("ΔPPL = student − teacher (best step)")
    ax2.grid(alpha=0.3)
    ax2.set_title("Absolute degradation vs scale")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out} and {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
