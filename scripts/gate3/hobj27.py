"""Whose codes fit the captured activations better: ours (RTN) or theirs, under H-metric LS scales."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HB))

from bonsai_forensics import gate1_forensics as g1, rotation
from bonsai_forensics.run_quant import rotate_hessian

BASE = HB / "artifacts/base27"
GGUF = HB / "artifacts/oracle/bonsai27/Ternary-Bonsai-2-27B-PQ2_0.gguf"
MANIFEST = HB / "artifacts/oracle/bonsai27/hadamard-manifest.json"
HESS = HB / "artifacts/gate3/prefix27/hessians"


def h_metric_error(v, codes, h_blocks, rows, in_f, ng):
    """Per-group H-metric LS scale for `codes`, returns total error and base energy."""
    W = v.reshape(rows, ng, 128).astype(np.float64)
    T = codes.reshape(rows, ng, 128).astype(np.float64)
    err = 0.0
    energy = 0.0
    for gi in range(ng):
        Hg = h_blocks[gi]
        TW = T[:, gi, :] @ Hg
        num = (TW * W[:, gi, :]).sum(-1)
        den = (TW * T[:, gi, :]).sum(-1)
        s = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
        R = W[:, gi, :] - s[:, None] * T[:, gi, :]
        err += float((R @ Hg * R).sum())
        energy += float((W[:, gi, :] @ Hg * W[:, gi, :]).sum())
    return err, energy


def main():
    from safetensors import safe_open

    header = g1.parse_gguf_header(GGUF)
    sign_sets = rotation.load_sign_manifest(MANIFEST)
    index = json.loads((BASE / "model.safetensors.index.json").read_text())
    handles = {}
    print(f"{'tensor':30s} {'err_theirs':>10s} {'err_rtn':>10s} {'theirs/rtn':>10s}")
    ratios = []
    for layer in (0, 3):
        for entry in g1.layer_map(header, layer):
            gname = entry["gguf"]
            info = header.tensors[gname]
            if info.ggml_type != g1.GGML_TYPE_PQ2_0:
                continue
            hf = entry["hf"]
            hp = HESS / f"{hf.replace('model.language_model.', '')}.hessian.npy"
            if not hp.is_file():
                continue
            shard = index["weight_map"][hf]
            handle = handles.get(shard) or handles.setdefault(shard, safe_open(BASE / shard, framework="pt", device="cpu"))
            w = handle.get_tensor(hf).float().numpy()
            if w.shape[-1] != info.shape[0]:
                w = w.T
            w = g1.apply_layout(w, entry["layout"])
            signs = sign_sets[str(w.shape[-1])]
            v = rotation.absorb_input(w, signs).astype(np.float32)
            theirs, _ = g1.read_pq2_0(header, gname)
            rtn, _ = g1.absmean_trits(v)
            h_rot = rotate_hessian(np.load(hp), signs).astype(np.float32)
            rows, in_f = v.shape
            ng = in_f // 128
            hb = h_rot.reshape(ng, 128, ng, 128).transpose(0, 2, 1, 3).reshape(ng, ng, 128, 128)
            blocks = [hb[gi, gi] for gi in range(ng)]
            e_t, energy = h_metric_error(v, theirs, blocks, rows, in_f, ng)
            e_r, _ = h_metric_error(v, rtn, blocks, rows, in_f, ng)
            ratio = e_t / max(e_r, 1e-12)
            ratios.append(ratio)
            print(f"{gname:30s} {e_t/energy:10.5f} {e_r/energy:10.5f} {ratio:10.4f}")
    print("median err_theirs/err_rtn:", float(np.median(ratios)))
    out = HB / "artifacts/gate3/prefix27/h-objective.json"
    out.write_text(json.dumps({"median_ratio": float(np.median(ratios)), "ratios": ratios}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
