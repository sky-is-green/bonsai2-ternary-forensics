"""Minimal DSpark draft model for local smoke training and export plumbing.

Not the production DeepSpec implementation — a faithful reconstruction of the
component set the PrismML-Eng fork's ``conversion/dspark.py`` expects:

    input = concat(embed(x), corrected_target_hidden)
        -> fc -> trunk(N x [GQA + SwiGLU]) -> norm -> logits
    logits += Markov bias  (low-rank term keyed on the previous token)
    confidence head -> per-position predicted acceptance

The real draft ties ``embed``/``lm_head`` to the *target* model's vocabulary and
token embeddings; here ``vocab_size`` is configurable so a smoke model can be
tiny.  This module is for **smoke training only**; production weights come from
DeepSpec and are converted by the fork.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Attention(nn.Module):
    """Grouped-query causal attention with an optional sliding window (no cache)."""

    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int,
                 sliding_window: int | None = None) -> None:
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        k = k.repeat_interleave(self.heads // self.kv_heads, dim=1)
        v = v.repeat_interleave(self.heads // self.kv_heads, dim=1)
        if self.sliding_window is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            idx = torch.arange(t, device=x.device)
            mask = ((idx[None, :] <= idx[:, None])
                    & (idx[None, :] > idx[:, None] - self.sliding_window))
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(b, t, self.heads * self.head_dim)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int,
                 intermediate: int, sliding_window: int | None = None) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(hidden)
        self.self_attn = Attention(hidden, heads, kv_heads, head_dim,
                                   sliding_window=sliding_window)
        self.post_attention_layernorm = RMSNorm(hidden)
        self.mlp = MLP(hidden, intermediate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class HiddenCorrection(nn.Module):
    """Gated MLP correction on the concatenated target hidden states."""

    def __init__(self, in_dim: int, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.embed_norm = RMSNorm(in_dim)
        self.hidden_norm = RMSNorm(hidden)
        self.gate_proj = nn.Linear(in_dim, intermediate, bias=False)
        self.up_proj = nn.Linear(in_dim, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, target_hidden: torch.Tensor) -> torch.Tensor:
        h = self.embed_norm(target_hidden)
        return self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


class DSparkDraft(nn.Module):
    """Trunk + Markov head + hidden correction + confidence head."""

    def __init__(self, *, hidden_size: int, vocab_size: int, num_layers: int = 5,
                 num_heads: int = 8, num_kv_heads: int = 2, head_dim: int = 64,
                 intermediate_size: int | None = None, target_hidden_size: int | None = None,
                 num_target_layers: int = 1, markov_rank: int = 64,
                 correction_size: int | None = None, mask_token_id: int = 0,
                 sliding_window: int | None = None, log_snr: bool = False,
                 log_snr_min: float = -10.0, log_snr_max: float = 10.0,
                 tie_embeddings: bool = False) -> None:
        super().__init__()
        intermediate_size = intermediate_size or 4 * hidden_size
        target_hidden_size = target_hidden_size or hidden_size
        correction_size = correction_size or hidden_size
        in_dim = num_target_layers * target_hidden_size
        self.mask_token_id = int(mask_token_id)
        self.markov_rank = int(markov_rank)
        self.sliding_window = sliding_window
        self.log_snr_min = float(log_snr_min)
        self.log_snr_max = float(log_snr_max)

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.pre_fc_norm_embedding = RMSNorm(hidden_size)
        self.pre_fc_norm_hidden = RMSNorm(correction_size)
        self.fc = nn.Linear(hidden_size + correction_size, hidden_size, bias=False)
        self.hidden_correction = HiddenCorrection(in_dim, correction_size, intermediate_size)
        self.layers = nn.ModuleList([
            Block(hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size,
                  sliding_window=sliding_window)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.log_snr_embed = (nn.Sequential(nn.Linear(1, hidden_size), nn.SiLU(),
                                            nn.Linear(hidden_size, hidden_size))
                              if log_snr else None)

        # Markov head: prev-token low-rank logit bias  (fork names markov_w1/w2).
        self.markov_w1 = nn.Embedding(vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, vocab_size, bias=False)
        # Confidence head: per-position predicted acceptance.
        self.confidence = nn.Linear(hidden_size, 1, bias=True)

        if tie_embeddings:
            self.lm_head.weight = self.embed.weight

    def tie_to_target(self, embed_weight: torch.Tensor, lm_head_weight: torch.Tensor | None = None,
                      *, tie: bool = False) -> None:
        """Initialise (and optionally tie) the draft's vocab weights from the target."""
        with torch.no_grad():
            self.embed.weight.copy_(embed_weight.to(self.embed.weight.dtype))
            if lm_head_weight is not None and not tie:
                self.lm_head.weight.copy_(lm_head_weight.to(self.lm_head.weight.dtype))
        if tie:
            self.lm_head.weight = self.embed.weight

    def log_snr_features(self, log_snr: torch.Tensor) -> torch.Tensor:
        """Normalise a log-SNR scalar to [-1, 1] before the MLP."""
        lo, hi = self.log_snr_min, self.log_snr_max
        return 2.0 * (log_snr - lo) / (hi - lo) - 1.0

    def markov_bias(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Low-rank bias from the previous token, chained across the sequence."""
        prev = input_ids.roll(1, dims=1)
        prev[:, 0] = self.mask_token_id
        return self.markov_w2(self.markov_w1(prev))

    def forward(self, input_ids: torch.Tensor, target_hidden: torch.Tensor,
                log_snr: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.embed(input_ids)
        correction = self.hidden_correction(target_hidden)
        x = self.fc(torch.cat([
            self.pre_fc_norm_embedding(embedding),
            self.pre_fc_norm_hidden(correction),
        ], dim=-1))
        if self.log_snr_embed is not None and log_snr is not None:
            snr = log_snr if log_snr.dim() == 3 else log_snr.reshape(-1, 1, 1)
            x = x + self.log_snr_embed(self.log_snr_features(snr))
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x) + self.markov_bias(input_ids)
        return logits, self.confidence(x).squeeze(-1)

    def hf_name_map(self) -> dict[str, str]:
        """Map local parameter names to the fork's HF/DeepSpec names."""
        mapping = {
            "fc.weight": "fc.weight",
            "hidden_correction.gate_proj.weight": "hidden_correction.gate_proj.weight",
            "hidden_correction.up_proj.weight": "hidden_correction.up_proj.weight",
            "hidden_correction.down_proj.weight": "hidden_correction.down_proj.weight",
            "hidden_correction.embed_norm.weight": "hidden_correction.embed_norm.weight",
            "hidden_correction.hidden_norm.weight": "hidden_correction.hidden_norm.weight",
            "markov_w1.weight": "markov_head.markov_w1.weight",
            "markov_w2.weight": "markov_head.markov_w2.weight",
            "confidence.weight": "confidence_head.proj.weight",
            "confidence.bias": "confidence_head.proj.bias",
            "norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
            "embed.weight": "embed_tokens.weight",
        }
        if self.log_snr_embed is not None:
            mapping["log_snr_embed.0.weight"] = "log_snr_embed.fc1.weight"
            mapping["log_snr_embed.0.bias"] = "log_snr_embed.fc1.bias"
            mapping["log_snr_embed.2.weight"] = "log_snr_embed.fc2.weight"
            mapping["log_snr_embed.2.bias"] = "log_snr_embed.fc2.bias"
        return mapping


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


__all__ = ["Block", "DSparkDraft", "HiddenCorrection", "count_parameters"]
