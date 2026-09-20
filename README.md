# bonsai-ternary-forensics

Independent forensics on **Bonsai 2 27B**, Prism ML's ~2 bpw ternary model built
from `Qwen/Qwen3.8-27B`. We recovered the storage format and rotation basis,
localized the quality gap to trained weights, and showed that a widely cited
public PTQ baseline is a calibration-passage artifact.

This is a study. It is not affiliated with Prism ML, Alibaba Cloud, or
ThakiCloud, and it redistributes no model weights.

## Findings at a glance

- **The format is not the secret.** The GGUF declares its own blockwise
  Walsh-Hadamard rotation (block 1024, explicit signs, input axis). Once the
  tensor layouts are lined up, plain absmean round-to-nearest on the rotated
  base agrees with Prism's trits **0.920 mean / 0.885 min** (chance: 0.33).
- **The residual carries the quality.** Replacing all 402 ternary payloads with
  our own rotate+RTN weights, keeping metadata byte-identical, collapses
  perplexity from **18.5851 to 23,606 (1,270x)** in the same binary, corpus, and
  settings.
- **It is not a quantizer trick.** GPTQ with real Hessians, damping sweeps,
  act-order, rotation-aware Hessians, and group scale search all move the codes
  *away* from Prism's; their codes fit our activation Hessians worse than plain
  RTN. The residual is trained weight movement.
- **The public 2.1x is a calibration artifact.** Reproducing the public QuIP
  reference gives the published 2.1x on a fixed passage; on held-out windows
  with real calibration data the same configurations collapse (1,178x to
  ~35,000x). Public PTQ at ~2 bpw does not survive off-calibration here.

Full write-up: [`docs/WHITEPAPER.md`](docs/WHITEPAPER.md).
Every falsified route, with evidence and revisit cost:
[`docs/FAILURES.md`](docs/FAILURES.md).
Detailed gate reports: [`research/`](research/).

## What is here

```
bonsai_forensics/        core library (rotation, quant, GPTQ, PQ2_0 codec,
                         oracle, gates, recovery) + spec.md
scripts/gate3/           SCR noise test, 27B prefix Hessians, GPTQ sweep,
                         H-objective diagnostics
scripts/pilot/           block-wise KD and data-selection pilot scripts
scripts/eval_llama_server.py  standalone smoke / perplexity evaluation
tests/ternary/           offline test suite (synthetic tensors; no downloads)
docs/                    white paper, failure register, runtime notes
research/                Gate-1/2/3 write-ups
configs/                 run configs
```

No weights or large artifacts are committed. The scripts expect data under
`artifacts/` (gitignored).

## Install

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The offline suite needs only numpy/pyyaml/pytest. The gate and pilot scripts
need torch/transformers/datasets; on ROCm install torch from the ROCm index.

## Reproduce

Download the released model and (for forensics) the public base. Both are
Apache-2.0:

```sh
huggingface-cli download prism-ml/Ternary-Bonsai-2-27B-gguf \
    Ternary-Bonsai-2-27B-PQ2_0.gguf --local-dir artifacts/oracle/bonsai27
huggingface-cli download Qwen/Qwen3.8-27B \
    --local-dir artifacts/base27
```

Then, from the repo root:

```sh
# Gate 1: does rotate+RTN in their basis reproduce their trits?
python -m bonsai_forensics.gate1_forensics \
    --gguf artifacts/oracle/bonsai27/Ternary-Bonsai-2-27B-PQ2_0.gguf \
    --layers 0,3,62,63 --out artifacts/gate1/gate1-report.json

# Gate 2: swap all PQ2_0 payloads for rotate+RTN and evaluate
python -m bonsai_forensics.gate2_rtn_artifact \
    --gguf artifacts/oracle/bonsai27/Ternary-Bonsai-2-27B-PQ2_0.gguf \
    --base-dir artifacts/base27 --out artifacts/gate2/patch-report.json

# Gate 3 diagnostics (see scripts/gate3/*.py for their arguments)
python scripts/gate3/scr_test.py
python scripts/gate3/prefix27.py --layers 0,3 --windows 8
python scripts/gate3/sweep27.py
python scripts/gate3/hobj27.py

# Evaluate a GGUF (one heavy ROCm process at a time)
python scripts/eval_llama_server.py smoke \
    --server-bin /path/to/llama-server \
    --gguf artifacts/oracle/bonsai27/Ternary-Bonsai-2-27B-PQ2_0.gguf \
    --ngl 99 --ctx 4096 --no-thinking --out artifacts/eval/smoke.json
```

The scripts take their artifact locations from the repo root; set
`HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` to pin the device.

## Tests

```sh
pytest tests/ternary -q
python -m bonsai_forensics.spec_hash --check
```

The suite is offline and uses synthetic tensors only. It pins the frozen wire
contract (`bonsai_forensics/spec.md`, canonical hash
`0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b`).

## Caveats

One corpus (about 1 MB of Shakespeare), 8x512 context, single runs each. The
effect sizes are far outside the noise and the payload-swap experiment is
causal, but this is not a benchmark suite. We did not run 27B QAT, so we cannot
claim to reproduce Prism's numbers; we localized the gap and identified the only
remaining route.

## Provenance

This repository was extracted from a private research project. Task identifiers
(T30, T31, T32, ...), round numbers, and references to "project records" /
"project handoff" in the docs and research write-ups refer to that project's
internal coordination log, which is not published. Commit hashes and artifact
paths cited as evidence in `docs/FAILURES.md` are provenance from the original
working repository; the reports in `research/` and the numbers reproduced here
are the authoritative record. No model weights are redistributed.

## License and attribution

Apache-2.0 (see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)). The analyzed
model, Bonsai 2 27B, is by **Prism ML** (Apache-2.0), derived from **Qwen3.8-27B**
by **Alibaba Cloud** (Apache-2.0); "Created using Bonsai by Prism ML." The
public PTQ reference reproduced by `bonsai_forensics/reference.py` is
**ThakiCloud/bonsai-1bit-repro** (Apache-2.0). Runtime evaluation uses Prism's
llama.cpp fork (MIT) on ggml (MIT).
