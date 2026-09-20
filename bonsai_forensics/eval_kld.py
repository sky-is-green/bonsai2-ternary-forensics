"""T8 — `llama-perplexity` KLD/PPL wrapper with a stable JSON schema (ADR-7).

Wraps the mainline binary: one run saves base logits
(`--kl-divergence-base FNAME`), the second run computes KLD against them
(`--kl-divergence --kl-divergence-base FNAME`). Output parsing is pinned to
the format strings in `tools/perplexity/perplexity.cpp` (llama.cpp build
11030) and tested offline against a fixture, because the local build ships no
`llama-perplexity` binary yet (QUEEN escalation: rebuild target needed).

The parsed schema is stable and JSON-serializable:

    {
      "command": [...], "returncode": 0,
      "ppl":     {"q", "q_std", "base", "base_std", "ratio", "ratio_std",
                  "log_ratio", "log_ratio_std", "correlation", "diff", "diff_std"},
      "kld":     {"mean", "std", "max", "min", "median", "percentiles": {...}},
      "token":   {"mean_delta_p", "std_delta_p", "rms_delta_p", "same_top_p",
                  "same_top_p_std", "percentiles": {...}},
      "final_ppl": float | None,
      "chunks":  [{"chunk", "ppl", "ppl_unc", "log_ratio", "log_ratio_unc",
                   "kld", "kld_unc", "delta_p_rms", "delta_p_rms_unc",
                   "same_top_p", "same_top_p_unc"}, ...]
    }
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

SPEC_SHA256 = "0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b"

DEFAULT_BINARY_CANDIDATES = (
    os.environ.get("LLAMA_PERPLEXITY", ""),
    str(Path.home() / ".unsloth" / "llama.cpp" / "build" / "bin" / "llama-perplexity"),
)
INSTALL_HINT = (
    "llama-perplexity not found. Build it from the local checkout "
    "(cmake --build ~/.unsloth/llama.cpp/build --target llama-perplexity) "
    "or set LLAMA_PERPLEXITY=/path/to/llama-perplexity."
)

_SECTION_PERPLEXITY = "====== Perplexity statistics ======"
_SECTION_KLD = "====== KL divergence statistics ======"
_SECTION_TOKEN = "====== Token probability statistics ======"

_NUM = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
_PM_RE = re.compile(rf"({_NUM})\s*(?:±\s*({_NUM}))?")
_FINAL_PPL_RE = re.compile(rf"Final estimate:\s*PPL\s*=\s*({_NUM})")


def find_perplexity_binary(binary: str | Path | None = None) -> Path:
    candidates: list[str] = []
    if binary is not None:
        candidates.append(str(binary))
    candidates.extend(candidate for candidate in DEFAULT_BINARY_CANDIDATES if candidate)
    found = shutil.which("llama-perplexity")
    if found:
        candidates.append(found)
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError(INSTALL_HINT)


def _common_args(
    model: str | Path,
    corpus: str | Path,
    *,
    chunks: int | None = None,
    ctx: int | None = None,
    n_gpu_layers: int | None = None,
) -> list[str]:
    args = ["-m", str(model), "-f", str(corpus)]
    if chunks is not None:
        args += ["--chunks", str(int(chunks))]
    if ctx is not None:
        args += ["-c", str(int(ctx))]
    if n_gpu_layers is not None:
        args += ["-ngl", str(int(n_gpu_layers))]
    return args


def build_save_logits_command(
    binary: str | Path,
    model: str | Path,
    corpus: str | Path,
    logits_file: str | Path,
    **common,
) -> list[str]:
    return [
        str(binary),
        *_common_args(model, corpus, **common),
        "--kl-divergence-base",
        str(logits_file),
    ]


def build_kld_command(
    binary: str | Path,
    model: str | Path,
    corpus: str | Path,
    base_logits: str | Path,
    **common,
) -> list[str]:
    return [
        str(binary),
        *_common_args(model, corpus, **common),
        "--kl-divergence",
        "--kl-divergence-base",
        str(base_logits),
    ]


def build_ppl_command(binary: str | Path, model: str | Path, corpus: str | Path, **common) -> list[str]:
    return [str(binary), *_common_args(model, corpus, **common)]


def _normalize_label(label: str) -> str:
    label = label.strip().replace("Δp", "delta_p").replace("δp", "delta_p")
    label = label.replace("Δ", "delta").replace("δ", "delta").lower()
    label = label.replace("%", " ")
    label = re.sub(r"\s+", " ", label).strip()
    return label


def _numbers(line: str) -> tuple[float | None, float | None]:
    match = _PM_RE.search(line)
    if not match:
        return None, None
    value = float(match.group(1))
    spread = float(match.group(2)) if match.group(2) is not None else None
    return value, spread


def _split_label(line: str) -> tuple[str, str]:
    head, separator, tail = line.partition(":")
    return (_normalize_label(head), tail) if separator else ("", line)


def parse_perplexity_output(text: str) -> dict:
    """Parse `llama-perplexity` text into the stable schema above."""
    result: dict = {
        "ppl": {},
        "kld": {"percentiles": {}},
        "token": {"percentiles": {}},
        "final_ppl": None,
        "chunks": [],
    }
    section = ""
    in_chunks = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == _SECTION_PERPLEXITY:
            section = "ppl"
            continue
        if line == _SECTION_KLD:
            section = "kld"
            continue
        if line == _SECTION_TOKEN:
            section = "token"
            continue
        if line.startswith("Final estimate:"):
            match = _FINAL_PPL_RE.search(line)
            if match:
                result["final_ppl"] = float(match.group(1))
            continue
        if re.match(r"^chunk\s+PPL\b", line):
            in_chunks = True
            continue
        if in_chunks:
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[0].isdigit():
                row = _PM_RE.findall(parts[1])
                if row and all(spread for _, spread in row):
                    means = [float(value) for value, _ in row]
                    spreads = [float(spread) for _, spread in row]
                    keys = ("ppl", "log_ratio", "kld", "delta_p_rms", "same_top_p")
                    chunk = {"chunk": int(parts[0])}
                    for key, mean, spread in zip(keys, means, spreads):
                        chunk[key] = mean
                        chunk[f"{key}_unc"] = spread
                    result["chunks"].append(chunk)
                    continue
            in_chunks = False
        if not line or ":" not in line:
            continue

        label, tail = _split_label(line)
        value, spread = _numbers(tail)
        if value is None:
            continue

        if label == "mean ppl(q)":
            result["ppl"].update(q=value, q_std=spread)
        elif label == "mean ppl(base)":
            result["ppl"].update(base=value, base_std=spread)
        elif label == "mean ppl(q)/ppl(base)":
            result["ppl"].update(ratio=value, ratio_std=spread)
        elif label == "mean ln(ppl(q)/ppl(base))":
            result["ppl"].update(log_ratio=value, log_ratio_std=spread)
        elif label.startswith("cor(ln(ppl(q))"):
            result["ppl"]["correlation"] = value
        elif label == "mean ppl(q)-ppl(base)":
            result["ppl"].update(diff=value, diff_std=spread)
        elif label.endswith("kld"):
            key = label[: -len("kld")].strip().replace(" ", "_") or "mean"
            if section == "kld":
                if key in ("mean",):
                    result["kld"].update(mean=value, std=spread)
                elif key in ("maximum",):
                    result["kld"]["max"] = value
                elif key in ("minimum",):
                    result["kld"]["min"] = value
                elif key in ("median",):
                    result["kld"]["median"] = value
                else:
                    result["kld"]["percentiles"][key] = value
        elif label.endswith("delta_p"):
            key = label[: -len("delta_p")].strip().replace(" ", "_") or "mean"
            if section == "token":
                if key == "mean":
                    result["token"].update(mean_delta_p=value, std_delta_p=spread)
                elif key == "rms":
                    result["token"]["rms_delta_p"] = value
                elif key == "maximum":
                    result["token"]["max_delta_p"] = value
                elif key == "minimum":
                    result["token"]["min_delta_p"] = value
                elif key == "median":
                    result["token"]["median_delta_p"] = value
                else:
                    result["token"]["percentiles"][key] = value
        elif label == "same top p":
            result["token"].update(same_top_p=value, same_top_p_std=spread)
    return result


def run_command(command: Sequence[str], *, timeout: float | None = None) -> tuple[int, str]:
    completed = subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    return completed.returncode, completed.stdout


def run_wrapper(
    command: Sequence[str],
    *,
    timeout: float | None = None,
    check: bool = True,
) -> dict:
    returncode, output = run_command(command, timeout=timeout)
    if check and returncode != 0:
        raise RuntimeError(
            f"llama-perplexity exited with {returncode}: {output.strip()[-2000:]}"
        )
    parsed = parse_perplexity_output(output)
    parsed["command"] = [str(part) for part in command]
    parsed["returncode"] = returncode
    return parsed


def save_base_logits(
    model: str | Path,
    corpus: str | Path,
    logits_file: str | Path,
    *,
    binary: str | Path | None = None,
    timeout: float | None = None,
    **common,
) -> dict:
    command = build_save_logits_command(find_perplexity_binary(binary), model, corpus, logits_file, **common)
    return run_wrapper(command, timeout=timeout)


def compute_kld(
    model: str | Path,
    corpus: str | Path,
    base_logits: str | Path,
    *,
    binary: str | Path | None = None,
    timeout: float | None = None,
    **common,
) -> dict:
    command = build_kld_command(find_perplexity_binary(binary), model, corpus, base_logits, **common)
    return run_wrapper(command, timeout=timeout)


def compute_ppl(
    model: str | Path,
    corpus: str | Path,
    *,
    binary: str | Path | None = None,
    timeout: float | None = None,
    **common,
) -> dict:
    command = build_ppl_command(find_perplexity_binary(binary), model, corpus, **common)
    return run_wrapper(command, timeout=timeout)


def within_budget(kld_mean: float, reference_kld: float, ratio: float = 2.0) -> bool:
    """T20 gate: KLD ≤ `ratio` × the reference (Bonsai 2 PQ2_0) KLD."""
    if reference_kld <= 0:
        raise ValueError("reference KLD must be positive")
    return kld_mean <= ratio * reference_kld


def to_json(parsed: dict) -> str:
    return json.dumps(parsed, sort_keys=True, indent=2)
