"""Hessian-variant sweep for the 27B prefix: which (if any) error-compensated rule matches their codes?"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HB))

from bonsai_forensics import gate1_forensics as g1, gptq, rotation
from bonsai_forensics.run_quant import rotate_hessian

BASE = HB / "artifacts/base27"
GGUF = HB / "artifacts/oracle/bonsai27/Ternary-Bonsai-2-27B-PQ2_0.gguf"
MANIFEST = HB / "artifacts/oracle/bonsai27/hadamard-manifest.json"
HESS = HB / "artifacts/gate3/prefix27/hessians"


def group_scale_search(w, h_blocks, rows, in_f, use_diag):
    """Per-group activation-aware scale search: s = argmin (w - s t)^T H_g (w - s t)."""
    ng = in_f // 128
    W = w.reshape(rows, ng, 128).astype(np.float64)
    codes = np.empty((rows, ng, 128), dtype=np.int8)
    scales = np.empty((rows, ng), dtype=np.float32)
    for gi in range(ng):
        if use_diag:
            hd = np.diag(h_blocks[gi])
            s = np.abs(W[:, gi, :]).mean(-1)
            for _ in range(4):
                t = np.clip(np.sign(W[:, gi, :] / s[:, None]) * np.floor(np.abs(W[:, gi, :] / s[:, None]) + 0.5), -1, 1)
                num = (hd[None, :] * t * W[:, gi, :]).sum(-1)
                den = (hd[None, :] * t * t).sum(-1)
                s = np.where(den > 0, num / np.maximum(den, 1e-12), s)
        else:
            Hg = h_blocks[gi]
            s = np.abs(W[:, gi, :]).mean(-1)
            for _ in range(4):
                t = np.clip(np.sign(W[:, gi, :] / s[:, None]) * np.floor(np.abs(W[:, gi, :] / s[:, None]) + 0.5), -1, 1)
                TW = t @ Hg
                num = (TW * W[:, gi, :]).sum(-1)
                den = (TW * t).sum(-1)
                s = np.where(den > 0, num / np.maximum(den, 1e-12), s)
        t = np.clip(np.sign(W[:, gi, :] / s[:, None]) * np.floor(np.abs(W[:, gi, :] / s[:, None]) + 0.5), -1, 1)
        codes[:, gi, :] = t.astype(np.int8)
        scales[:, gi] = s
    return codes.reshape(rows, in_f)


def main() -> int:
    from safetensors import safe_open

    header = g1.parse_gguf_header(GGUF)
    sign_sets = rotation.load_sign_manifest(MANIFEST)
    index = json.loads((BASE / "model.safetensors.index.json").read_text())
    handles = {}
    layer = 0
    results = {}
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
        h = np.load(hp)
        h_rot = rotate_hessian(h, signs).astype(np.float32)
        h_t = rotate_hessian(h, signs)
        h_alt = h_t.T.copy()  # R^T H R
        rows, in_f = v.shape
        ng = in_f // 128

        variants = {}
        rtn, _ = g1.absmean_trits(v)
        variants["rtn"] = rtn
        for damp in (0.01, 0.1, 0.3):
            variants[f"gptq_d{damp}"] = gptq.gptq_quantize(
                torch.from_numpy(v), torch.from_numpy(h_rot), group_size=128, damp=damp, device="cuda"
            ).codes.cpu().numpy().astype(np.int8)
        variants["gptq_act"] = gptq.gptq_quantize(
            torch.from_numpy(v), torch.from_numpy(h_rot), group_size=128, damp=0.01, act_order=True, device="cuda"
        ).codes.cpu().numpy().astype(np.int8)
        variants["gptq_altH"] = gptq.gptq_quantize(
            torch.from_numpy(v), torch.from_numpy(h_alt), group_size=128, damp=0.01, device="cuda"
        ).codes.cpu().numpy().astype(np.int8)
        variants["gptq_rawH"] = gptq.gptq_quantize(
            torch.from_numpy(v), torch.from_numpy(h.astype(np.float32)), group_size=128, damp=0.01, device="cuda"
        ).codes.cpu().numpy().astype(np.int8)
        hb = h_rot.reshape(ng, 128, ng, 128).transpose(0, 2, 1, 3).reshape(ng, ng, 128, 128)
        blocks = [hb[gi, gi] for gi in range(ng)]
        variants["scale_diag"] = group_scale_search(v, blocks, rows, in_f, True)
        variants["scale_block"] = group_scale_search(v, blocks, rows, in_f, False)
        results[gname] = {k: float((c == theirs).mean()) for k, c in variants.items()}
        print(f"{gname}: " + " ".join(f"{k}={v:.4f}" for k, v in results[gname].items()), flush=True)

    agg = {k: float(np.mean([r[k] for r in results.values()])) for k in results[next(iter(results))]}
    print("AGGREGATE:", json.dumps(agg, indent=1))
    out = HB / "artifacts/gate3/prefix27/sweep-layer0.json"
    out.write_text(json.dumps({"per_tensor": results, "aggregate": agg}, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
