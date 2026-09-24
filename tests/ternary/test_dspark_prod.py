"""Tests for the fork-compatible DSpark trunk (`bonsai_forensics.dspark_prod`)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from bonsai_forensics.dspark_prod import DSparkProdConfig, DSparkProdDraft, count_parameters

_check_path = Path(__file__).resolve().parents[2] / "scripts" / "pilot" / "dspark_export_check.py"
_spec = importlib.util.spec_from_file_location("dspark_export_check", _check_path)
_check = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _check  # dataclasses resolve fields via sys.modules[__module__]
_spec.loader.exec_module(_check)


def _tiny() -> DSparkProdDraft:
    return DSparkProdDraft(DSparkProdConfig(
        hidden_size=32, vocab_size=64, num_layers=2, num_heads=4, num_kv_heads=2,
        head_dim=8, intermediate_size=64, n_capture=3, markov_rank=8,
        block_size=4, mask_token_id=63, confidence_head=True,
        confidence_head_with_markov=True))


def _fork_spec(cfg: DSparkProdConfig) -> "_check.ForkDSparkSpec":
    return _check.ForkDSparkSpec(
        hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, num_heads=cfg.num_heads,
        num_kv_heads=cfg.num_kv_heads, head_dim=cfg.head_dim,
        intermediate_size=cfg.intermediate_size, vocab_size=cfg.vocab_size,
        n_capture=cfg.n_capture, markov_rank=cfg.markov_rank,
        hidden_correction=False, confidence_head=cfg.confidence_head,
        confidence_with_markov=cfg.confidence_head_with_markov, log_snr=False)


def test_inventory_matches_fork_spec() -> None:
    model = _tiny()
    report = _check.audit({n: tuple(p.shape) for n, p in model.named_parameters()},
                          _fork_spec(model.config))
    assert report["ok"], report
    assert not report["missing"] and not report["extra"] and not report["shape_mismatch"]


def test_hf_name_map_covers_every_parameter() -> None:
    model = _tiny()
    mapping = model.hf_name_map()
    assert set(mapping) == {n for n, _ in model.named_parameters()}
    assert len(set(mapping.values())) == len(mapping)  # one-to-one
    assert mapping["fc.weight"] == "dspark.fc.weight"
    assert mapping["layers.1.self_attn.q_norm.weight"] == "blk.1.attn_q_norm.weight"
    assert mapping["markov_head.markov_w1.weight"] == "dspark.markov_head_a.weight"


def test_forward_and_backward() -> None:
    model = _tiny()
    cfg = model.config
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    taps = torch.randn(2, 5, cfg.n_embd_cap)          # 5 context rows
    prev = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
    logits, conf = model(tokens, taps, prev_tokens=prev)
    assert logits.shape == (2, cfg.block_size, cfg.vocab_size)
    assert conf is not None and conf.shape == (2, cfg.block_size)
    assert torch.isfinite(logits).all() and torch.isfinite(conf).all()
    logits.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_forward_without_markov_or_context_is_valid() -> None:
    model = DSparkProdDraft(DSparkProdConfig(
        hidden_size=32, vocab_size=64, num_layers=2, num_heads=4, num_kv_heads=2,
        head_dim=8, intermediate_size=64, n_capture=3, markov_rank=0,
        block_size=4, confidence_head=False))
    tokens = torch.randint(0, model.config.vocab_size, (1, 4))
    taps = torch.randn(1, 1, model.config.n_embd_cap)
    logits, conf = model(tokens, taps)  # no markov head, no confidence head
    assert logits.shape == (1, 4, model.config.vocab_size)
    assert conf is None


def test_forward_bf16_mixed_dtype() -> None:
    """SDPA requires q/k/v to share a dtype; rope must not promote q/k to fp32."""
    model = _tiny().to(torch.bfloat16).eval()
    cfg = model.config
    tokens = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    taps = torch.randn(1, 3, cfg.n_embd_cap).to(torch.bfloat16)
    prev = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    with torch.no_grad():
        logits, conf = model(tokens, taps, prev_tokens=prev)
    assert logits.dtype == torch.bfloat16 and torch.isfinite(logits.float()).all()


def test_default_config_has_the_bonsai_assumptions() -> None:
    cfg = DSparkProdConfig()
    assert cfg.n_embd_cap == 5 * 5120 == 25600  # matches the collected taps
    assert cfg.block_size == 7 and cfg.markov_rank == 256
    assert count_parameters(DSparkProdDraft(cfg)) > 0
