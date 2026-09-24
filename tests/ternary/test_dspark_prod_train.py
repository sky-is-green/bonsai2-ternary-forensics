"""Tests for the DSpark block-diffusion batching (`dspark_prod_train.make_block`)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "pilot" / "dspark_prod_train.py"
_spec = importlib.util.spec_from_file_location("dspark_prod_train", _PATH)
mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mod
_spec.loader.exec_module(mod)


def test_make_block_context_labels_and_positions() -> None:
    tokens = np.arange(10, dtype=np.int64)
    taps = np.arange(10 * 4, dtype=np.float32).reshape(10, 4)
    ctx, draft, labels, prev, pos = mod.make_block(tokens, taps, start=3, block_size=3,
                                                   window=2, mask_token_id=99)
    assert ctx.shape == (2, 4)
    assert np.array_equal(ctx, taps[1:3])                 # last `window` rows before start
    assert np.array_equal(draft, [99, 99, 99])
    assert np.array_equal(labels, [3, 4, 5])
    assert np.array_equal(prev, [2, 3, 4])                 # anchor then preceding slot
    assert np.array_equal(pos, [1, 2, 3, 4, 5])


def test_make_block_at_start_has_empty_context_and_mask_anchor() -> None:
    tokens = np.arange(6, dtype=np.int64)
    taps = np.zeros((6, 4), dtype=np.float32)
    ctx, draft, labels, prev, pos = mod.make_block(tokens, taps, start=0, block_size=2,
                                                   window=4, mask_token_id=99)
    assert ctx.shape == (0, 4)
    assert np.array_equal(labels, [0, 1])
    assert prev[0] == 99 and prev[1] == 0
    assert np.array_equal(pos, [0, 1])
