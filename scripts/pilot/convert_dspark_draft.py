#!/usr/bin/env python
"""Convert a Qwen3DSparkModel draft to a proper dspark-arch GGUF.

The PrismML fork's ``conversion/__init__.py::TEXT_MODEL_MAP`` contains a
duplicate ``"Qwen3DSparkModel"`` key — first ``"dspark"`` then ``"qwen"`` — and
the later value wins, so the stock ``convert_hf_to_gguf.py`` resolves the
architecture to ``conversion.qwen.DSparkModel`` (a DFlash subclass) and emits a
``dflash``-arch GGUF.  The DSpark speculative runtime
(``common_speculative_impl_draft_dspark``) and ``tests/test-dspark-real-eval``
require ``arch = dspark`` with ``dspark.*`` KVs, which only
``conversion.dspark.DSparkModel`` writes.

This wrapper imports ``conversion.qwen`` first and ``conversion.dspark`` second,
so the DSpark registration wins, then runs the stock converter unchanged.

Usage (same args as convert_hf_to_gguf.py):

    python scripts/pilot/convert_dspark_draft.py <draft_dir> \
        --outfile draft-dspark-f16.gguf --outtype f16
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

LLAMA = Path(os.environ.get("LLAMA_CPP", Path.home() / "llama.cpp"))
if not (LLAMA / "convert_hf_to_gguf.py").is_file():
    raise SystemExit(f"convert_hf_to_gguf.py not found under {LLAMA} (set LLAMA_CPP)")
sys.path.insert(0, str(LLAMA / "gguf-py"))
sys.path.insert(0, str(LLAMA))

import conversion.qwen    # noqa: E402,F401  (registers the name first)
import conversion.dspark  # noqa: E402,F401  (re-registers it, so this one wins)

sys.argv = ["convert_hf_to_gguf.py", *sys.argv[1:]]
runpy.run_path(str(LLAMA / "convert_hf_to_gguf.py"), run_name="__main__")
