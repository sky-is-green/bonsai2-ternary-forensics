"""Fork-compatible DSpark drafter for Bonsai 2 (Path 1).

Mirrors the PrismML-Eng fork's ``dspark`` runtime topology
(``src/models/dspark.cpp``) so trained weights convert and load:

    ctx   = RMSNorm(fc(target_taps))                     # [T_ctx, n_embd]
    x     = token_embd(draft_tokens)                     # [T_draft, n_embd]
    layer:
        seq = concat(ctx, RMSNorm_attn(x))               # sequence axis
        q,k,v = proj(seq);  RMSNorm over head_dim; RoPE
        attn = softmax(q kᵀ/√d) v   (NON-causal);  o_proj
        x = x + o_proj(attn)[draft rows]
        x = x + SwiGLU(RMSNorm_ffn(x))
    logits = output(RMSNorm_out(x)) + markov_bias(prev_token)

Unlike the smoke model in :mod:`bonsai_forensics.dspark` (concat embedding +
corrected hidden into ``fc``, no qk-norm), this matches the runtime's tensor
inventory and forward; ``fork_spec()`` + ``scripts/pilot/dspark_export_check.py``
verify the inventory, and ``hf_name_map()`` matches ``conversion/dspark.py``.

Scope note: this is the trunk.  The Markov semi-autoregressive resample and the
optional host-side hidden-correction/log-SNR heads are training stages
(`docs/DSPARK-PATH1-PLAN.md`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DSparkProdConfig:
    hidden_size: int = 5120
    vocab_size: int = 248320
    num_layers: int = 5
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 9728
    n_capture: int = 5
    markov_rank: int = 256
    block_size: int = 7
    mask_token_id: int = 0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    confidence_head: bool = True
    confidence_head_with_markov: bool = True
    tie_word_embeddings: bool = False

    @property
    def n_embd_cap(self) -> int:
        return self.n_capture * self.hidden_size


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, position_ids: torch.Tensor, theta: float) -> torch.Tensor:
    """NeoX-style RoPE on ``[B, H, T, head_dim]`` (matches the converter's default)."""
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    freqs = position_ids.float()[:, None, :, None] * inv[None, None, None, :]  # [B, 1, T, d/2]
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).to(x.dtype)           # [B, 1, T, d]
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).to(x.dtype)
    return x * cos + _rotate_half(x) * sin


class Attention(nn.Module):
    """Non-causal GQA over ``concat(target_ctx, norm(x))`` with qk-norm."""

    def __init__(self, cfg: DSparkProdConfig) -> None:
        super().__init__()
        self.h = cfg.num_heads
        self.kv = cfg.num_kv_heads
        self.hd = cfg.head_dim
        self.theta = cfg.rope_theta
        self.q_proj = nn.Linear(cfg.hidden_size, self.h * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.kv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.kv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.h * self.hd, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.hd, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.hd, cfg.rms_norm_eps)

    def forward(self, ctx: torch.Tensor, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        b, tc, _ = ctx.shape
        td = x.shape[1]
        seq = torch.cat([ctx, x], dim=1)
        q = self.q_proj(seq).view(b, tc + td, self.h, self.hd).transpose(1, 2)
        k = self.k_proj(seq).view(b, tc + td, self.kv, self.hd).transpose(1, 2)
        v = self.v_proj(seq).view(b, tc + td, self.kv, self.hd).transpose(1, 2)
        q = apply_rope(self.q_norm(q), position_ids, self.theta)
        k = apply_rope(self.k_norm(k), position_ids, self.theta)
        k = k.repeat_interleave(self.h // self.kv, dim=1)
        v = v.repeat_interleave(self.h // self.kv, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).reshape(b, tc + td, self.h * self.hd)
        return self.o_proj(out)[:, tc:, :]


class MLP(nn.Module):
    def __init__(self, cfg: DSparkProdConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: DSparkProdConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, ctx: torch.Tensor, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(ctx, self.input_layernorm(x), position_ids)
        return x + self.mlp(self.post_attention_layernorm(x))


class DSparkProdDraft(nn.Module):
    """Fork-compatible DSpark trunk (see module docstring)."""

    def __init__(self, cfg: DSparkProdConfig | None = None) -> None:
        super().__init__()
        self.config = cfg or DSparkProdConfig()
        c = self.config
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.fc = nn.Linear(c.n_embd_cap, c.hidden_size, bias=False)
        self.hidden_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.layers = nn.ModuleList([Block(c) for _ in range(c.num_layers)])
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        self.markov_head = _Markov(c) if c.markov_rank > 0 else None
        self.confidence_head = _Confidence(c) if c.confidence_head else None
        if c.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, tokens: torch.Tensor, target_taps: torch.Tensor,
                position_ids: torch.Tensor | None = None,
                prev_tokens: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, td = tokens.shape
        ctx = self.hidden_norm(self.fc(target_taps))
        tc = ctx.shape[1]
        if position_ids is None:
            position_ids = torch.arange(tc + td, device=tokens.device).unsqueeze(0).expand(b, -1)
        x = self.embed_tokens(tokens)
        for layer in self.layers:
            x = layer(ctx, x, position_ids)
        x = self.norm(x)
        logits = self.lm_head(x)
        markov_feat = None
        if self.markov_head is not None and prev_tokens is not None:
            markov_feat = self.markov_head.markov_w1(prev_tokens)
            logits = logits + self.markov_head.markov_w2(markov_feat)
        conf = None
        if self.confidence_head is not None:
            conf = self.confidence_head(x, markov_feat).squeeze(-1)
        return logits, conf

    def hf_name_map(self) -> dict[str, str]:
        """Local parameter name -> the fork's GGUF tensor name (conversion/dspark.py)."""
        mapping = {
            "embed_tokens.weight": "token_embd.weight",
            "norm.weight": "output_norm.weight",
            "lm_head.weight": "output.weight",
            "fc.weight": "dspark.fc.weight",
            "hidden_norm.weight": "dspark.hidden_norm.weight",
        }
        if self.markov_head is not None:
            mapping["markov_head.markov_w1.weight"] = "dspark.markov_head_a.weight"
            mapping["markov_head.markov_w2.weight"] = "dspark.markov_head_b.weight"
        if self.confidence_head is not None:
            mapping["confidence_head.proj.weight"] = "dspark.confidence_head.weight"
            mapping["confidence_head.proj.bias"] = "dspark.confidence_head.bias"
        layer = {
            "input_layernorm.weight": "attn_norm.weight",
            "post_attention_layernorm.weight": "ffn_norm.weight",
            "self_attn.q_proj.weight": "attn_q.weight",
            "self_attn.k_proj.weight": "attn_k.weight",
            "self_attn.v_proj.weight": "attn_v.weight",
            "self_attn.o_proj.weight": "attn_output.weight",
            "self_attn.q_norm.weight": "attn_q_norm.weight",
            "self_attn.k_norm.weight": "attn_k_norm.weight",
            "mlp.gate_proj.weight": "ffn_gate.weight",
            "mlp.up_proj.weight": "ffn_up.weight",
            "mlp.down_proj.weight": "ffn_down.weight",
        }
        for i in range(self.config.num_layers):
            for src, dst in layer.items():
                mapping[f"layers.{i}.{src}"] = f"blk.{i}.{dst}"
        return mapping


class _Markov(nn.Module):
    """Markov logit bias: ``markov_w2(markov_w1(prev_token))``."""

    def __init__(self, cfg: DSparkProdConfig) -> None:
        super().__init__()
        self.markov_w1 = nn.Embedding(cfg.vocab_size, cfg.markov_rank)
        self.markov_w2 = nn.Linear(cfg.markov_rank, cfg.vocab_size, bias=False)


class _Confidence(nn.Module):
    def __init__(self, cfg: DSparkProdConfig) -> None:
        super().__init__()
        self.in_dim = cfg.hidden_size + (cfg.markov_rank if cfg.confidence_head_with_markov else 0)
        # stored as [1, in_dim] in the GGUF; a projection to 1 keeps that shape
        self.proj = nn.Linear(self.in_dim, 1)

    def forward(self, x: torch.Tensor, markov_feat: torch.Tensor | None = None) -> torch.Tensor:
        if self.in_dim != x.shape[-1]:
            if markov_feat is None:
                raise ValueError("confidence_head_with_markov requires the markov feature")
            x = torch.cat([x, markov_feat], dim=-1)
        return self.proj(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


__all__ = [
    "DSparkProdConfig",
    "DSparkProdDraft",
    "RMSNorm",
    "apply_rope",
    "count_parameters",
]
