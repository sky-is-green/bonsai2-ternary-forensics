"""Offline tests for the minimal DSpark draft model (no GPU)."""

from __future__ import annotations

import torch

from bonsai_forensics.dspark import DSparkDraft, count_parameters


def _model(vocab: int = 64) -> DSparkDraft:
    return DSparkDraft(
        hidden_size=32, vocab_size=vocab, num_layers=2, num_heads=4, num_kv_heads=2,
        head_dim=8, intermediate_size=64, target_hidden_size=16, num_target_layers=2,
        markov_rank=8, correction_size=32, mask_token_id=vocab - 1,
    )


def test_forward_shapes() -> None:
    model = _model()
    tokens = torch.randint(0, 64, (2, 10))
    target_hidden = torch.randn(2, 10, 32)  # 2 target layers * 16
    logits, confidence = model(tokens, target_hidden)
    assert logits.shape == (2, 10, 64)
    assert confidence.shape == (2, 10)
    assert torch.isfinite(logits).all() and torch.isfinite(confidence).all()


def test_markov_bias_depends_on_previous_token_not_current() -> None:
    model = _model()
    model.eval()
    tokens = torch.randint(0, 60, (1, 8))
    bias = model.markov_bias(tokens)
    # position 0 uses the mask token; changing token[0] must not change bias[0]
    changed = tokens.clone()
    changed[0, 0] = (changed[0, 0] + 1) % 60
    bias_changed = model.markov_bias(changed)
    assert torch.allclose(bias[0, 0], bias_changed[0, 0])
    # but changing token[0] *does* change bias[1] (it keys on the previous token)
    assert not torch.allclose(bias[0, 1], bias_changed[0, 1])


def test_hf_name_map_covers_head_and_correction() -> None:
    mapped = _model().hf_name_map()
    for expected in ("fc.weight", "markov_w1.weight", "markov_w2.weight",
                     "confidence.weight", "hidden_correction.gate_proj.weight",
                     "hidden_correction.down_proj.weight"):
        assert expected in mapped
        assert mapped[expected].startswith(("dspark.", "hidden_correction.",
                                            "markov_head.", "confidence_head.")) \
            or mapped[expected] in ("fc.weight", "lm_head.weight", "embed_tokens.weight",
                                    "norm.weight")


def test_seed_config_is_trainable() -> None:
    model = _model()
    assert count_parameters(model) > 0
    tokens = torch.randint(0, 64, (1, 6))
    target_hidden = torch.randn(1, 6, 32)
    logits, _ = model(tokens, target_hidden)
    logits.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_sliding_window_log_snr_and_tied_embeddings() -> None:
    model = DSparkDraft(
        hidden_size=32, vocab_size=64, num_layers=2, num_heads=4, num_kv_heads=2,
        head_dim=8, intermediate_size=64, target_hidden_size=16, num_target_layers=2,
        markov_rank=8, correction_size=32, mask_token_id=63,
        sliding_window=4, log_snr=True, log_snr_min=-10.0, log_snr_max=10.0,
        tie_embeddings=True)
    tokens = torch.randint(0, 64, (2, 12))
    target_hidden = torch.randn(2, 12, 32)
    logits, confidence = model(tokens, target_hidden, log_snr=torch.full((2, 12, 1), 3.0))
    assert logits.shape == (2, 12, 64) and confidence.shape == (2, 12)
    assert model.lm_head.weight is model.embed.weight
    other, _ = model(tokens, target_hidden, log_snr=torch.full((2, 12, 1), -5.0))
    assert not torch.allclose(logits, other)


def test_tie_to_target_copies_and_ties() -> None:
    model = _model()
    weight = torch.randn(64, 32)
    model.tie_to_target(weight, tie=True)
    assert torch.allclose(model.embed.weight, weight)
    assert model.lm_head.weight is model.embed.weight
