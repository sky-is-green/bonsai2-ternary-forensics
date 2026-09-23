"""Retention vs scale — Prism's published curve vs our ladder, with an
absolute-PPL-delta view.

Why two panels: the ratio `student_ppl / teacher_ppl` is not scale-comparable
when the FP teacher itself differs across sizes (a weaker teacher is easier to
match). Absolute delta `student_ppl - teacher_ppl` is the complementary view.
Report both; read neither alone.

Prism points are transcribed from the released whitepapers
(see docs/RETENTION-VS-SCALE.md). Our points come from the ladder reports.

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

# label -> (report dir, params_B)
RUNS = {
    "0.6B (blk 1024)": (HB / "artifacts/rmd/ladder-0.6B", 0.6),
    "1.7B (blk 1024)": (HB / "artifacts/rmd/ladder-1.7B", 1.7),
    "4B (blk 512)": (HB / "artifacts/rmd/ladder-4B", 4.0),
    "1.7B (blk 512)": (HB / "artifacts/rmd/ste-rotate-10k-block512", 1.7),
}


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
        params=None,
        teacher_ppl=d.get("teacher_ppl"),
        steps=d.get("steps"),
        best_step=best["step"],
        best_ratio=best["aggregate"]["mean_ratio"],
        best_retention=1.0 / best["aggregate"]["mean_ratio"],
        best_ppl=best["aggregate"]["mean_ppl"],
        best_delta=delta(best),
        final_step=fin["step"],
        final_ratio=fin["aggregate"]["mean_ratio"],
        final_retention=1.0 / fin["aggregate"]["mean_ratio"],
        final_delta=delta(fin),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HB / "docs/figures/retention-vs-scale.png"))
    ap.add_argument("--summary", default=str(HB / "artifacts/rmd/ladder-summary.md"))
    args = ap.parse_args()

    rows = {}
    for label, (path, params) in RUNS.items():
        r = load(path)
        if r:
            r["params"] = params
            rows[label] = r

    # --- table ---
    lines = ["# Size ladder — ratio and absolute-delta views", "",
             "| run | params | teacher PPL | best step | best ratio | retention | "
             "best PPL | ΔPPL (best) | final ratio | final ΔPPL |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for label, r in sorted(rows.items(), key=lambda kv: kv[1]["params"]):
        lines.append(
            f"| {label} | {r['params']}B | {r['teacher_ppl']:.2f} | {r['best_step']} | "
            f"{r['best_ratio']:.4f} | {r['best_retention']*100:.1f}% | {r['best_ppl']:.2f} | "
            f"{r['best_delta']:.2f} | {r['final_ratio']:.4f} | {r['final_delta']:.2f} |")
    table = "\n".join(lines) + "\n"
    Path(args.summary).write_text(table)
    print(table)

    # --- figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    for label, pts in PRISM.items():
        ax1.plot([p for p, _ in pts], [r for _, r in pts], "o--",
                 label=f"Prism {label}", alpha=0.8)
    xs = [r["params"] for r in rows.values()]
    ys = [r["best_retention"] for r in rows.values()]
    if xs:
        ax1.plot(xs, ys, "s-", color="crimson", lw=2, label="ours (PPL ratio)")
    ax1.axhline(0.97, color="grey", ls=":", lw=1)
    ax1.text(0.62, 0.973, "mission bar (27B)", color="grey", fontsize=8)
    ax1.set_xscale("log")
    ax1.set_xticks([0.6, 1.7, 4, 8, 27])
    ax1.set_xticklabels(["0.6B", "1.7B", "4B", "8B", "27B"])
    ax1.set_xlabel("parameters")
    ax1.set_ylabel("retention (1 / ratio)")
    ax1.set_ylim(0.4, 1.02)
    ax1.grid(alpha=0.3)
    ax1.set_title("Retention vs scale (ratio view)")
    ax1.legend(fontsize=7, loc="lower right")

    if xs:
        ys_delta = [r["best_delta"] for r in rows.values()]
        ax2.plot(xs, ys_delta, "s-", color="darkorange", lw=2)
        for label, r in rows.items():
            ax2.annotate(label, (r["params"], r["best_delta"]), fontsize=7,
                         textcoords="offset points", xytext=(4, 4))
    ax2.set_xscale("log")
    ax2.set_xticks([0.6, 1.7, 4, 8, 27])
    ax2.set_xticklabels(["0.6B", "1.7B", "4B", "8B", "27B"])
    ax2.set_xlabel("parameters")
    ax2.set_ylabel("ΔPPL = student − teacher (best step)")
    ax2.grid(alpha=0.3)
    ax2.set_title("Absolute degradation vs scale")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out} and {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
