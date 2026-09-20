"""27B prefix diagnostic: real Hessians from a 4-layer prefix, GPTQ in Prism's basis.

Loads only the embedding + selected layers of Qwen3.8-27B (no 55 GB residency),
captures Hessians on real text, then runs GPTQ in the T30-corrected basis
(layout + rotation) and measures trit agreement vs the released PQ2_0 codes,
overall and in the threshold band.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HB))

from bonsai_forensics import activations, gate1_forensics as g1, gptq, rotation
from bonsai_forensics.run_quant import rotate_hessian

BASE = HB / "artifacts/base27"
GGUF = HB / "artifacts/oracle/bonsai27/Ternary-Bonsai-2-27B-PQ2_0.gguf"
MANIFEST = HB / "artifacts/oracle/bonsai27/hadamard-manifest.json"
CORPUS = HB / "artifacts/canary/tinyshakespeare.txt"


def log(msg):
    print(f"[prefix] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--out", default=str(HB / "artifacts/gate3/prefix27"))
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    cfg = AutoConfig.from_pretrained(BASE)
    tcfg = cfg.text_config
    tcfg.num_hidden_layers = max(layers) + 1
    tcfg.layer_types = list(tcfg.layer_types)[: max(layers) + 1]
    if hasattr(tcfg, "mtp_num_hidden_layers"):
        tcfg.mtp_num_hidden_layers = 0
    log(f"building text model with {tcfg.num_hidden_layers} layers")
    model = Qwen3_5TextModel(tcfg).to(dtype=torch.bfloat16, device="cuda")

    index = json.loads((BASE / "model.safetensors.index.json").read_text())
    handles: dict[str, object] = {}
    sd = {}
    for key, shard in index["weight_map"].items():
        if not key.startswith("model.language_model."):
            continue
        stripped = key[len("model.language_model."):]
        if stripped.startswith("layers."):
            if int(stripped.split(".")[1]) > max(layers):
                continue
        elif stripped not in ("embed_tokens.weight", "norm.weight"):
            continue
        handle = handles.get(shard)
        if handle is None:
            handle = handles.setdefault(shard, safe_open(BASE / shard, framework="pt", device="cpu"))
        sd[stripped] = handle.get_tensor(key)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    log(f"weights loaded: {len(sd)} tensors, missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    ids = tokenizer(CORPUS.read_text(encoding="utf-8"), return_tensors="np")["input_ids"].reshape(-1)
    calib = ids[: args.windows * args.seq].reshape(args.windows, args.seq)
    batches = [
        {"input_ids": torch.tensor(w, dtype=torch.long).unsqueeze(0),
         "attention_mask": torch.ones((1, args.seq), dtype=torch.long)}
        for w in calib
    ]

    wanted = []
    for layer in layers:
        for suffix in (
            "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
            "linear_attn.in_proj_a", "linear_attn.in_proj_b",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        ):
            wanted.append(f"layers.{layer}.{suffix}.weight")
    state_keys = set(model.state_dict())
    wanted = [w for w in wanted if w in state_keys]
    hess_dir = out / "hessians"
    log(f"capturing {len(wanted)} Hessians over {args.windows} windows")
    activations.capture_hessians(model, batches, names=wanted, out_dir=hess_dir, dtype=torch.float32)
    log(f"capture done at {time.time() - started:.0f}s")
    del model
    torch.cuda.empty_cache()

    header = g1.parse_gguf_header(GGUF)
    sign_sets = rotation.load_sign_manifest(MANIFEST)
    report = {"layers": layers, "windows": args.windows, "seq": args.seq, "damp": args.damp, "tensors": {}}
    for layer in layers:
        for entry in g1.layer_map(header, layer):
            gname = entry["gguf"]
            info = header.tensors[gname]
            if info.ggml_type != g1.GGML_TYPE_PQ2_0:
                continue
            hess_name = entry["hf"].replace("model.language_model.", "")
            hpath = hess_dir / f"{hess_name}.hessian.npy"
            if not hpath.is_file():
                continue
            shard = index["weight_map"][entry["hf"]]
            handle = handles.get(shard) or handles.setdefault(shard, safe_open(BASE / shard, framework="pt", device="cpu"))
            w = handle.get_tensor(entry["hf"]).float().numpy()
            if w.shape[-1] != info.shape[0]:
                w = w.T
            w = g1.apply_layout(w, entry["layout"])
            signs = sign_sets[str(w.shape[-1])]
            v = rotation.absorb_input(w, signs).astype(np.float32)
            theirs, _ = g1.read_pq2_0(header, gname)
            rtn, _ = g1.absmean_trits(v)
            h = np.load(hpath)
            h_rot = rotate_hessian(h, signs)
            g = gptq.gptq_quantize(
                torch.from_numpy(v), torch.from_numpy(h_rot),
                group_size=128, damp=args.damp, device="cuda",
            ).codes.cpu().numpy().astype(np.int8)
            absv = np.abs(v)
            rows, in_f = v.shape
            grouped = absv.reshape(rows, -1, 128)
            thr = np.quantile(grouped, 0.8, axis=-1, keepdims=True)
            salient = (grouped >= thr).reshape(rows, in_f)
            report["tensors"][gname] = {
                "layout": entry["layout"],
                "rtn_overall": float((rtn == theirs).mean()),
                "rtn_top20": float((rtn == theirs)[salient].mean()),
                "gptq_overall": float((g == theirs).mean()),
                "gptq_top20": float((g == theirs)[salient].mean()),
                "gptq_bot80": float((g == theirs)[~salient].mean()),
                "rtn_bot80": float((rtn == theirs)[~salient].mean()),
                "gptq_moved": float((g != rtn).mean()),
                "theirs_zero": float((theirs == 0).mean()),
            }
            log(f"{gname}: rtn={report['tensors'][gname]['rtn_overall']:.4f} "
                f"gptq={report['tensors'][gname]['gptq_overall']:.4f} moved={report['tensors'][gname]['gptq_moved']:.4f}")

    vals = list(report["tensors"].values())
    report["aggregate"] = {
        key: float(np.mean([v[key] for v in vals]))
        for key in ("rtn_overall", "gptq_overall", "rtn_top20", "gptq_top20", "rtn_bot80", "gptq_bot80", "gptq_moved")
    } if vals else {}
    (out / "prefix-report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    log(f"aggregate: {json.dumps(report['aggregate'])}")
    log(f"seconds: {time.time() - started:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
