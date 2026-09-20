"""SCR falsification test: does 5% random-token noise in calibration improve trit agreement?

Compares, per tensor, against Prism's released 1.7B Q2_0_g64 trits:
  rtn_absmean, gptq_clean, gptq_noise5
Clean Hessians come from the T10 canary capture (32x2048); noisy Hessians are
captured here with 5% of tokens replaced by uniform-random vocab ids.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HB))

from bonsai_forensics import activations, gptq, oracle
from bonsai_forensics.quant import quantize_rtn_absmean

MODEL_DIR = HB / "artifacts/canary/hf"
CORPUS = HB / "artifacts/canary/tinyshakespeare.txt"
CLEAN_HESS = HB / "artifacts/canary/hessians"
NOISY_HESS = HB / "artifacts/gate3/hessians-noise5"
THEIR_GGUF = HB / "artifacts/oracle/bonsai17/Ternary-Bonsai-1.7B-PQ2_0.gguf"
OUT = HB / "artifacts/gate3/scr-report.json"

SAMPLES, SEQ_LEN, NOISE = 32, 2048, 0.05
GROUP = 128
DAMP = 0.1

SUFFIX = {
    "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
}


def log(msg):
    print(f"[scr] {msg}", flush=True)


def gguf_name(hf_name: str) -> str | None:
    if hf_name == "lm_head.weight":
        return "output.weight"
    m = re.match(r"model\.layers\.(\d+)\.(.+)\.weight$", hf_name)
    if not m:
        return None
    suffix = SUFFIX.get(m.group(2))
    return f"blk.{m.group(1)}.{suffix}.weight" if suffix else None


def their_codes(raw: bytes, table, name: str) -> np.ndarray:
    info = table.tensors[name]
    assert info.ggml_type == 142, info.ggml_type
    ne0, ne1 = info.shape
    row = oracle._row_size(info.ggml_type, info.shape)
    start = table.data_offset + info.offset
    blocks = np.frombuffer(raw[start:start + row * ne1], dtype=np.uint8).reshape(ne1, ne0 // 128, 34)
    qs = blocks[:, :, 2:34]
    codes = np.empty((ne1, ne0 // 128, 128), dtype=np.int8)
    for i in range(128):
        codes[:, :, i] = ((qs[:, :, i // 4] >> (2 * (i % 4))) & 3).astype(np.int8) - 1
    return codes.reshape(ne1, ne0)


def main() -> int:
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    log("tokenizer + windows")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    ids = tokenizer(CORPUS.read_text(encoding="utf-8"), return_tensors="np")["input_ids"].reshape(-1)
    calib = ids[: SAMPLES * SEQ_LEN].reshape(SAMPLES, SEQ_LEN)
    rng = np.random.default_rng(0)
    noisy = calib.copy()
    mask = rng.random(noisy.shape) < NOISE
    noisy[mask] = rng.integers(0, len(tokenizer), size=int(mask.sum()))
    log(f"noise injected on {int(mask.sum())} tokens ({mask.mean():.4f})")

    if not all((NOISY_HESS / f"{n}.hessian.npy").is_file() for n in ("model.layers.0.self_attn.q_proj.weight",)):
        log("loading model (cuda) for noisy Hessian capture")
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cuda")
        batches = [
            {"input_ids": torch.tensor(w, dtype=torch.long).unsqueeze(0),
             "attention_mask": torch.ones((1, SEQ_LEN), dtype=torch.long)}
            for w in noisy
        ]
        NOISY_HESS.mkdir(parents=True, exist_ok=True)
        captured = activations.capture_hessians(model, batches, out_dir=NOISY_HESS, dtype=torch.float32)
        log(f"captured {len(captured)} noisy Hessians in {time.time() - started:.0f}s")
        del model
        torch.cuda.empty_cache()
    else:
        log("noisy Hessians already present, skipping capture")

    table = oracle.parse_gguf_table(THEIR_GGUF)
    raw = Path(THEIR_GGUF).read_bytes()
    index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
    shards = {s: safe_open(MODEL_DIR / s, framework="pt", device="cpu") for s in set(index["weight_map"].values())}

    names = [n for n in index["weight_map"] if gguf_name(n) in table.tensors]
    log(f"comparing {len(names)} tensors")
    per_tensor = {}
    for hf_name in sorted(names):
        gname = gguf_name(hf_name)
        w = shards[index["weight_map"][hf_name]].get_tensor(hf_name).float().numpy()
        codes_theirs = their_codes(raw, table, gname)
        if codes_theirs.shape != w.shape:
            log(f"skip {gname}: shape {codes_theirs.shape} vs {w.shape}")
            continue
        rtn = quantize_rtn_absmean(w, group_size=GROUP).codes[..., : w.shape[-1]].astype(np.int8)
        wt = torch.from_numpy(w)
        h_clean = torch.from_numpy(np.load(CLEAN_HESS / f"{hf_name}.hessian.npy"))
        h_noisy = torch.from_numpy(np.load(NOISY_HESS / f"{hf_name}.hessian.npy"))
        g_clean = gptq.gptq_quantize(wt, h_clean, group_size=GROUP, damp=DAMP, device="cuda").codes.cpu().numpy().astype(np.int8)
        g_noisy = gptq.gptq_quantize(wt, h_noisy, group_size=GROUP, damp=DAMP, device="cuda").codes.cpu().numpy().astype(np.int8)
        per_tensor[gname] = {
            "rtn": float((rtn == codes_theirs).mean()),
            "gptq_clean": float((g_clean == codes_theirs).mean()),
            "gptq_noise5": float((g_noisy == codes_theirs).mean()),
            "noise_moved": float((g_noisy != g_clean).mean()),
        }
        log(f"{gname}: rtn={per_tensor[gname]['rtn']:.4f} clean={per_tensor[gname]['gptq_clean']:.4f} "
            f"noise={per_tensor[gname]['gptq_noise5']:.4f}")

    agg = {k: float(np.mean([v[k] for v in per_tensor.values()])) for k in ("rtn", "gptq_clean", "gptq_noise5", "noise_moved")}
    report = {
        "task": "SCR falsification",
        "date": "2026-09-20",
        "settings": {"samples": SAMPLES, "seq_len": SEQ_LEN, "noise_ratio": NOISE, "group": GROUP, "damp": DAMP},
        "aggregate": agg,
        "per_tensor": per_tensor,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    log(f"aggregate: {json.dumps(agg)}")
    log(f"seconds: {time.time() - started:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
