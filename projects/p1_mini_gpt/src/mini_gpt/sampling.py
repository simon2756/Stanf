"""Next-token sampling: temperature, top-k, top-p (nucleus) and min-p.

Filters are applied in the same order as Hugging Face ``generate``: temperature first, then
top-k, top-p and min-p. Every filter keeps at least one token.
"""

from __future__ import annotations

import torch


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the ``k`` highest logits."""
    k = min(k, logits.size(-1))
    kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Keep the smallest set of tokens whose probabilities sum to at least ``p``."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = sorted_logits.softmax(dim=-1)
    # Drop a token if the tokens ranked above it already cover ``p``.
    drop_sorted = (probs.cumsum(dim=-1) - probs) >= p
    drop = drop_sorted.scatter(-1, sorted_idx, drop_sorted)
    return logits.masked_fill(drop, float("-inf"))


def min_p_filter(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    """Keep tokens with probability >= ``min_p`` times the top token's probability.

    Unlike top-p, the cut-off adapts to the model's confidence: a peaked distribution keeps
    few tokens, a flat one keeps many (Nguyen et al., 2024).
    """
    probs = logits.softmax(dim=-1)
    threshold = min_p * probs.max(dim=-1, keepdim=True).values
    return logits.masked_fill(probs < threshold, float("-inf"))


def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token id per row from logits of shape (B, vocab). Temperature 0 is greedy."""
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    logits = logits.float() / temperature
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)
    if min_p is not None:
        logits = min_p_filter(logits, min_p)
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
