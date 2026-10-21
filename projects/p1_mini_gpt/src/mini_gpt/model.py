"""A decoder-only transformer with the design choices of current open LLMs.

Compared with the 2017 "Attention Is All You Need" decoder, this model uses:

* pre-norm residual blocks with RMSNorm instead of post-norm LayerNorm
* rotary position embeddings (RoPE) instead of absolute position embeddings
* RMSNorm on queries and keys (QK-norm), which keeps attention logits bounded during training
* a SwiGLU feed-forward block instead of a ReLU/GELU MLP
* no bias terms, and input/output embeddings tied together
* optional grouped-query attention (``n_kv_head < n_head``), explored in detail in P2

``pos_emb="learned"`` and ``qk_norm=False`` exist for ablations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from mini_gpt.sampling import sample_next


@dataclass
class GPTConfig:
    vocab_size: int = 16384
    max_seq_len: int = 512
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int = 6
    """Fewer KV heads than query heads gives grouped-query attention (GQA)."""
    d_model: int = 384
    mlp_hidden: int | None = None
    """SwiGLU hidden size. Defaults to 8/3 * d_model rounded up to a multiple of 64."""
    pos_emb: Literal["rope", "learned"] = "rope"
    rope_theta: float = 10_000.0
    qk_norm: bool = True
    tie_embeddings: bool = True
    dropout: float = 0.0
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        if self.mlp_hidden is None:
            self.mlp_hidden = 64 * math.ceil(8 * self.d_model / 3 / 64)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


class RMSNorm(nn.Module):
    """Normalise by the root mean square (no mean subtraction, no bias), then scale."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()  # compute the statistic in fp32 even under bf16 autocast
        x_f = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x_f.type_as(x) * self.weight


def rope_cache(head_dim: int, max_seq_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape (max_seq_len, head_dim) in the "rotate half" layout.

    Dimension pair (i, i + head_dim/2) is rotated by angle ``pos * theta^(-2i/head_dim)``.
    This is the layout used by Llama, Qwen and Hugging Face, which matters when loading their
    weights later (P2).
    """
    inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    pos = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` of shape (B, H, T, head_dim) by position. cos/sin: (T, head_dim)."""
    return (x * cos + rotate_half(x) * sin).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.use_rope = cfg.pos_emb == "rope"
        self.dropout = cfg.dropout
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, cfg.norm_eps) if cfg.qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.use_rope:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # Fused attention kernel (FlashAttention / memory-efficient backend when available).
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=self.n_kv_head != self.n_head,
        )
        return self.o_proj(y.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    """FFN(x) = W_down (SiLU(W_gate x) * W_up x)."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.mlp_hidden is not None
        self.gate_proj = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.down_proj = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.attn_norm(x), cos, sin))
        x = x + self.drop(self.mlp(self.mlp_norm(x)))
        return x


class GPT(nn.Module):
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = (
            nn.Embedding(cfg.max_seq_len, cfg.d_model) if cfg.pos_emb == "learned" else None
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale the projections that write into the residual stream so its variance does
        # not grow with depth (GPT-2 style).
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if self.pos_emb is not None:
                n -= self.pos_emb.weight.numel()
        return n

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, T = idx.shape
        if self.cfg.max_seq_len < T:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}")
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(x)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        eos_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling. Recomputes the full context each step; the KV cache
        that avoids this is built in P5."""
        for _ in range(max_new_tokens):
            context = idx[:, -self.cfg.max_seq_len :]
            logits, _ = self(context)
            next_id = sample_next(
                logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                generator=generator,
            )
            idx = torch.cat([idx, next_id[:, None]], dim=1)
            if eos_id is not None and bool((next_id == eos_id).all()):
                break
        return idx
