"""Pack block-wise trained masters into a PQ2_0 artifact (in-place patch of Prism's GGUF).

Trained masters are grouped-basis absorbed weights `R(W)`; the runtime stores the
tiled layout, and row tiling commutes with input-axis rotation, so the packer
applies the T30 layout at pack time. Only the quantized linears are replaced;
exempt tensors stay from the source artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HB))

from bonsai_forensics import gate1_forensics as g1
from bonsai_forensics import pq2_0

MODULE_TO_SUFFIX = {
    "linear_attn.in_proj_qkv": "attn_qkv",
    "linear_attn.in_proj_z": "attn_gate",
    "linear_attn.out_proj": "ssm_out",
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(HB / "artifacts/oracle/bonsai27/Ternary-Bonsai-2-27B-PQ2_0.gguf"))
    ap.add_argument("--dest", default=str(HB / "artifacts/pilot/blockwise/bw-artifact.gguf"))
    ap.add_argument("--masters", default=str(HB / "artifacts/pilot/blockwise"))
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--out", default=str(HB / "artifacts/pilot/blockwise/pack-report.json"))
    args = ap.parse_args()

    import torch

    source, dest = Path(args.source), Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    header = g1.parse_gguf_header(source)
    report = {"patched": [], "missing_masters": [], "seconds": None}

    with dest.open("r+b") as out:
        for layer in range(64):
            path = Path(args.masters) / f"layer{layer:02d}-masters.pt"
            if not path.is_file():
                report["missing_masters"].append(layer)
                continue
            payload = torch.load(path, map_location="cpu")
            for module_name, master in payload.items():
                suffix = MODULE_TO_SUFFIX.get(module_name)
                if suffix is None:
                    continue
                gname = f"blk.{layer}.{suffix}.weight"
                info = header.tensors[gname]
                weight = g1.apply_layout(master.float().numpy(), g1._LAYOUTS[suffix][1])
                codes, scales = g1.absmean_trits(weight, args.group)
                packed = pq2_0.pack_pq2_0(codes, scales)
                expected = g1.tensor_nbytes(info)
                if len(packed) != expected:
                    raise ValueError(f"{gname}: {len(packed)} != {expected} bytes")
                out.seek(header.data_offset + info.offset)
                out.write(packed)
                report["patched"].append({
                    "tensor": gname,
                    "sha256": hashlib.sha256(packed).hexdigest(),
                    "zero_fraction": float((codes == 0).mean()),
                })
            print(f"[pack] layer {layer}: {len(payload)} masters", flush=True)

    report["aggregate"] = {
        "patched": len(report["patched"]),
        "layers": 64 - len(report["missing_masters"]),
        "zero_fraction_mean": float(np.mean([item["zero_fraction"] for item in report["patched"]]))
        if report["patched"] else None,
    }
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
