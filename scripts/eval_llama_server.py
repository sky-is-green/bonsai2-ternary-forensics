"""Standalone evaluation for a GGUF with llama.cpp (no harness dependency).

Two modes:

  smoke       start ``llama-server``, wait for ``/health``, run a set of chat
              prompts, record the responses and per-reply latency, shut the
              server down, and write a JSON report.
  perplexity  run ``llama-perplexity`` directly on a corpus and record PPL.

This replaces the hivebench eval driver (which depended on the harness model
manager). It is intentionally stdlib-only apart from the model binaries.

Examples::

    python scripts/eval_llama_server.py smoke \
        --server-bin /path/to/llama-server \
        --gguf /path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf \
        --ngl 99 --ctx 4096 --no-thinking \
        --out artifacts/eval/smoke.json

    python scripts/eval_llama_server.py perplexity \
        --perplexity-bin /path/to/llama-perplexity \
        --gguf /path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf \
        --corpus artifacts/corpus/tinyshakespeare.txt \
        --ctx 512 --chunks 8 --ngl 99 \
        --out artifacts/eval/ppl.json

Remember: run one heavy ROCm process at a time, and pin the device with
``HIP_VISIBLE_DEVICES`` / ``CUDA_VISIBLE_DEVICES`` if the box is shared.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_PROMPTS = [
    "Reply with the single word: ok",
    "Name the capital of France.",
    "What is 17 times 3? Answer with the number only.",
    "Write one short sentence about the sea.",
    "Repeat exactly: ternary.",
]

PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.eE+-]+)\s*(?:\+/?-\s*([0-9.eE+-]+))?")


def _post_json(url: str, payload: dict, timeout: float = 600.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError):
        return False


def _wait_healthy(base_url: str, timeout: float, proc: subprocess.Popen) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {proc.returncode}")
        if _get(f"{base_url}/health"):
            return
        time.sleep(1.0)
    raise TimeoutError(f"llama-server not healthy after {timeout:.0f}s")


def run_smoke(args: argparse.Namespace) -> dict:
    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        prompts = [
            line.strip()
            for line in Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    log_path = Path(args.log) if args.log else None
    log_handle = open(log_path, "w", encoding="utf-8") if log_path else subprocess.DEVNULL
    cmd = [
        args.server_bin,
        "-m", args.gguf,
        "--host", args.host,
        "--port", str(args.port),
        "-ngl", str(args.ngl),
        "-c", str(args.ctx),
    ]
    if args.extra:
        cmd += args.extra

    base_url = f"http://{args.host}:{args.port}"
    print(f"[eval] starting: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT)
    started = time.time()
    rows = []
    try:
        _wait_healthy(base_url, args.startup_timeout, proc)
        print(f"[eval] healthy after {time.time() - started:.1f}s", flush=True)
        for prompt in prompts:
            payload = {
                "model": "local",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
            }
            if args.no_thinking:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            t0 = time.time()
            try:
                resp = _post_json(f"{base_url}/v1/chat/completions", payload)
                content = resp["choices"][0]["message"].get("content") or ""
                status = "PASS" if content.strip() else "EMPTY"
                tokens = (resp.get("usage") or {}).get("completion_tokens")
            except Exception as exc:  # noqa: BLE001 - report, do not abort the run
                content, status, tokens = f"<error: {exc}>", "ERROR", None
            rows.append({
                "prompt": prompt,
                "response": content,
                "status": status,
                "completion_tokens": tokens,
                "seconds": round(time.time() - t0, 3),
            })
            print(f"[eval] {status}: {prompt!r} -> {content[:80]!r}", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        if log_path:
            log_handle.close()

    passed = sum(1 for r in rows if r["status"] == "PASS")
    return {
        "mode": "smoke",
        "gguf": args.gguf,
        "server_bin": args.server_bin,
        "ngl": args.ngl,
        "ctx": args.ctx,
        "no_thinking": args.no_thinking,
        "prompts": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "seconds": round(time.time() - started, 3),
        "rows": rows,
    }


def run_perplexity(args: argparse.Namespace) -> dict:
    cmd = [
        args.perplexity_bin,
        "-m", args.gguf,
        "-f", args.corpus,
        "-ngl", str(args.ngl),
        "-c", str(args.ctx),
        "--chunks", str(args.chunks),
    ]
    if args.extra:
        cmd += args.extra

    print(f"[eval] starting: {' '.join(cmd)}", flush=True)
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    output = (proc.stdout or "") + (proc.stderr or "")
    match = PPL_RE.search(output)
    if not match:
        sys.stderr.write(output[-2000:])
        raise RuntimeError("could not parse perplexity from llama-perplexity output")
    ppl = float(match.group(1))
    stderr = float(match.group(2)) if match.group(2) else None
    return {
        "mode": "perplexity",
        "gguf": args.gguf,
        "perplexity_bin": args.perplexity_bin,
        "corpus": args.corpus,
        "ngl": args.ngl,
        "ctx": args.ctx,
        "chunks": args.chunks,
        "ppl": ppl,
        "stderr": stderr,
        "seconds": round(time.time() - started, 3),
        "returncode": proc.returncode,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    smoke = sub.add_parser("smoke", help="serve the GGUF and run chat prompts")
    smoke.add_argument("--server-bin", required=True, help="path to llama-server")
    smoke.add_argument("--gguf", required=True)
    smoke.add_argument("--host", default="127.0.0.1")
    smoke.add_argument("--port", type=int, default=8090)
    smoke.add_argument("--ngl", default=99, help="GPU layers (int) or 'all'")
    smoke.add_argument("--ctx", type=int, default=4096)
    smoke.add_argument("--max-tokens", type=int, default=64)
    smoke.add_argument("--temperature", type=float, default=0.0)
    smoke.add_argument("--no-thinking", action="store_true", help="pass enable_thinking=false")
    smoke.add_argument("--prompts-file", default="", help="one prompt per line")
    smoke.add_argument("--startup-timeout", type=float, default=900.0)
    smoke.add_argument("--log", default="", help="server log path")
    smoke.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra flags passed to llama-server")
    smoke.add_argument("--out", required=True)
    smoke.set_defaults(func=run_smoke)

    ppl = sub.add_parser("perplexity", help="run llama-perplexity on a corpus")
    ppl.add_argument("--perplexity-bin", required=True, help="path to llama-perplexity")
    ppl.add_argument("--gguf", required=True)
    ppl.add_argument("--corpus", required=True)
    ppl.add_argument("--ngl", default=99, help="GPU layers (int) or 'all'")
    ppl.add_argument("--ctx", type=int, default=512)
    ppl.add_argument("--chunks", type=int, default=8)
    ppl.add_argument("--timeout", type=float, default=3600.0)
    ppl.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra flags passed to llama-perplexity")
    ppl.add_argument("--out", required=True)
    ppl.set_defaults(func=run_perplexity)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if isinstance(args.ngl, str) and args.ngl.lower() == "all":
        args.ngl = 999
    result = args.func(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[eval] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
