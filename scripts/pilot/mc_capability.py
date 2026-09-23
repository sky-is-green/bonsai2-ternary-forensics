"""Capability vs access — is the quantized model's loss a ceiling loss or a
reliability loss?

A single greedy pass gives one accuracy number, which measures *access* (how
often it succeeds) and only upper-bounds *capability* (what it can do). This
combines the FP and quantized per-item results and separates the two:

  - the 2x2 split (both right / both wrong / FP-right & quant-wrong / ...)
  - for items the FP got right and the quantized got wrong: where does the gold
    choice rank in the quantized model's scores, and what is P(gold) if the
    decision is sampled from softmax(score / T)? A nonzero, temperature-rising
    P(gold) means the capability is present but accessed unreliably; P(gold)
    pinned at chance means the choice is genuinely unavailable.

    python scripts/pilot/mc_capability.py --fp artifacts/mc/fp-1.7B.json \
        --quant artifacts/mc/quant-wikiconv-1.7B.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TEMPS = (0.5, 1.0, 2.0, 5.0, 10.0)
DRAWS = 4000


def argmax(xs):
    return max(range(len(xs)), key=lambda i: xs[i])


def p_gold(scores, gold, temp, rng):
    s = np.asarray(scores, dtype=np.float64) / temp
    p = np.exp(s - s.max())
    p /= p.sum()
    return float((rng.choice(len(scores), size=DRAWS, p=p) == gold).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp", required=True)
    ap.add_argument("--quant", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fp = json.loads(Path(args.fp).read_text())
    q = json.loads(Path(args.quant).read_text())
    rng = np.random.default_rng(0)

    totals = {"both_right": 0, "both_wrong": 0,
              "fp_right_quant_wrong": 0, "fp_wrong_quant_right": 0}
    ranks, margins, per_temp = [], [], {t: [] for t in TEMPS}

    for ft, qt in zip(fp["tasks"], q["tasks"]):
        assert ft["task"] == qt["task"]
        for fi, qi in zip(ft["items"], qt["items"]):
            gold = fi["gold"]
            fp_ok = argmax(fi["scores"]) == gold
            q_ok = argmax(qi["scores"]) == gold
            if fp_ok and q_ok:
                totals["both_right"] += 1
            elif not fp_ok and not q_ok:
                totals["both_wrong"] += 1
            elif fp_ok and not q_ok:
                totals["fp_right_quant_wrong"] += 1
                order = sorted(range(len(qi["scores"])), key=lambda i: -qi["scores"][i])
                ranks.append(order.index(gold) + 1)
                margins.append(qi["scores"][order[0]] - qi["scores"][gold])
                for t in TEMPS:
                    per_temp[t].append(p_gold(qi["scores"], gold, t, rng))
            else:
                totals["fp_wrong_quant_right"] += 1

    n = sum(totals.values())
    fp_acc = (totals["both_right"] + totals["fp_right_quant_wrong"]) / n
    q_acc = (totals["both_right"] + totals["fp_wrong_quant_right"]) / n

    print(f"items: {n}   FP acc {fp_acc*100:.1f}%   quant acc {q_acc*100:.1f}%   "
          f"retention {q_acc/fp_acc*100:.1f}%")
    print(f"  both right {totals['both_right']}   both wrong {totals['both_wrong']}   "
          f"FP-right/quant-wrong {totals['fp_right_quant_wrong']}   "
          f"FP-wrong/quant-right {totals['fp_wrong_quant_right']}")
    if ranks:
        r = np.array(ranks)
        print(f"  among FP-right/quant-wrong ({len(ranks)}):")
        print(f"    gold rank: 2nd {int((r==2).sum())}  3rd {int((r==3).sum())}  "
              f"4th+ {int((r>=4).sum())}   (rank 1 = would be correct)")
        print(f"    mean margin top-vs-gold: {np.mean(margins):.3f} nats")
        for t in TEMPS:
            v = np.array(per_temp[t])
            print(f"    P(gold) sampled @T={t:<4}: {v.mean()*100:5.1f}%  "
                  f"(chance {100/4:.1f}%, never-picks-gold {int((v==0).sum())}/{len(v)})")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"items": n, "fp_acc": fp_acc, "quant_acc": q_acc, "split": totals,
             "gold_ranks": ranks, "margins": margins}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
