"""Optimizers and learning-rate schedules.

* AdamW with weight decay only on matrices (not on norm gains).
* Muon (Jordan et al., 2024): momentum SGD whose update for each weight matrix is replaced by
  its nearest semi-orthogonal matrix, computed with a few Newton-Schulz iterations. It is used
  for the 2-D weights inside the transformer blocks; embeddings, the output head and norm
  gains stay on AdamW.
* Schedules: cosine decay, or warmup-stable-decay (WSD), which keeps the learning rate
  constant and decays only at the end, so a run can be extended without restarting.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn


def lr_multiplier(
    step: int,
    max_steps: int,
    warmup_steps: int,
    schedule: Literal["cosine", "wsd"] = "wsd",
    decay_frac: float = 0.2,
    min_lr_ratio: float = 0.0,
) -> float:
    """Factor in [min_lr_ratio, 1] that multiplies each parameter group's base learning rate."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "cosine":
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    decay_start = int(max_steps * (1.0 - decay_frac))
    if step < decay_start:
        return 1.0
    progress = (step - decay_start) / max(1, max_steps - decay_start)
    return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - min(progress, 1.0))


@torch.no_grad()
def newton_schulz_orthogonalize(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate U V^T for G = U S V^T with a quintic Newton-Schulz iteration.

    The coefficients are tuned for speed rather than exact convergence: singular values end up
    roughly in [0.7, 1.2], which works as well as exact orthogonalization in practice.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                buf.lerp_(p.grad, 1 - group["momentum"])
                g = p.grad.lerp(buf, group["momentum"]) if group["nesterov"] else buf
                update = newton_schulz_orthogonalize(g, steps=group["ns_steps"])
                # Scale so that tall and wide matrices get comparable per-element updates.
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.to(p.dtype), alpha=-group["lr"] * scale)
        return loss


def build_optimizers(
    model: nn.Module,
    kind: Literal["adamw", "muon"],
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    muon_lr: float = 0.02,
) -> list[torch.optim.Optimizer]:
    """Return the optimizers for ``model``. Each param group stores its base lr as
    ``initial_lr`` so the schedule can scale it."""
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    fused = torch.cuda.is_available()

    if kind == "adamw":
        decay = [p for p in params.values() if p.dim() >= 2]
        no_decay = [p for p in params.values() if p.dim() < 2]
        opt = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=betas,
            fused=fused,
        )
        optimizers: list[torch.optim.Optimizer] = [opt]
    else:
        hidden = [p for n, p in params.items() if p.dim() == 2 and n.startswith("blocks.")]
        hidden_ids = {id(p) for p in hidden}
        rest_matrices = [p for p in params.values() if p.dim() >= 2 and id(p) not in hidden_ids]
        gains = [p for p in params.values() if p.dim() < 2]
        optimizers = [
            Muon(hidden, lr=muon_lr, weight_decay=0.0),
            torch.optim.AdamW(
                [
                    {"params": rest_matrices, "weight_decay": weight_decay},
                    {"params": gains, "weight_decay": 0.0},
                ],
                lr=lr,
                betas=betas,
                fused=fused,
            ),
        ]

    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return optimizers
