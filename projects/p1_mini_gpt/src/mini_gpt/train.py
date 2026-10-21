"""Training loop for the mini-GPT.

    python -m mini_gpt.train --out-dir runs/baseline
    python -m mini_gpt.train --out-dir runs/muon --optimizer muon
    python -m mini_gpt.train --out-dir runs/no-qknorm --model.no-qk-norm

Logs go to ``<out_dir>/metrics.jsonl`` (and to Weights & Biases with ``--wandb-project``).
"""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro

from mini_gpt.model import GPT, GPTConfig
from mini_gpt.optim import build_optimizers, lr_multiplier


@dataclass
class TrainConfig:
    data_dir: Path = Path("data/tokens")
    out_dir: Path = Path("runs/default")
    model: GPTConfig = field(default_factory=GPTConfig)

    batch_size: int = 32
    grad_accum_steps: int = 1
    max_steps: int = 5000

    optimizer: Literal["adamw", "muon"] = "adamw"
    lr: float = 1e-3
    """AdamW learning rate (for Muon runs: embeddings, head and norm gains)."""
    muon_lr: float = 0.02
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    schedule: Literal["cosine", "wsd"] = "wsd"
    warmup_steps: int = 200
    decay_frac: float = 0.2
    min_lr_ratio: float = 0.0

    eval_interval: int = 250
    eval_batches: int = 50
    log_interval: int = 10

    device: str = "auto"
    dtype: Literal["auto", "float32", "bfloat16"] = "auto"
    compile: bool = False
    seed: int = 1337
    wandb_project: str | None = None


class TokenLoader:
    """Samples random windows from a flat uint16 token file (memory-mapped)."""

    def __init__(self, path: Path, batch_size: int, seq_len: int, device: torch.device, seed: int):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        if len(self.data) <= seq_len + 1:
            raise ValueError(f"{path} has only {len(self.data)} tokens")
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.rng = np.random.default_rng(seed)

    def batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        starts = self.rng.integers(0, len(self.data) - self.seq_len - 1, size=self.batch_size)
        windows = np.stack([self.data[s : s + self.seq_len + 1] for s in starts]).astype(np.int64)
        xy = torch.from_numpy(windows)
        if self.device.type == "cuda":
            xy = xy.pin_memory().to(self.device, non_blocking=True)
        else:
            xy = xy.to(self.device)
        return xy[:, :-1], xy[:, 1:]


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: GPT, loaders: dict[str, TokenLoader], n_batches: int, autocast) -> dict:
    model.eval()
    out = {}
    for split, loader in loaders.items():
        losses = []
        for _ in range(n_batches):
            x, y = loader.batch()
            with autocast:
                _, loss = model(x, y)
            assert loss is not None
            losses.append(loss.item())
        out[f"{split}_loss"] = float(np.mean(losses))
    model.train()
    return out


def train(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    meta = json.loads((cfg.data_dir / "meta.json").read_text())
    cfg.model.vocab_size = meta["vocab_size"]
    bytes_per_token = meta.get("val_bytes_per_token")

    use_bf16 = cfg.dtype == "bfloat16" or (
        cfg.dtype == "auto" and device.type == "cuda" and torch.cuda.is_bf16_supported()
    )
    autocast = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16) if use_bf16 else nullcontext()
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = GPT(cfg.model).to(device)
    print(f"model: {model.num_params():,} non-embedding params, {model.num_params(False):,} total")
    optimizers = build_optimizers(
        model, cfg.optimizer, cfg.lr, cfg.weight_decay, (cfg.beta1, cfg.beta2), cfg.muon_lr
    )
    fwd = torch.compile(model) if cfg.compile else model

    seq_len = cfg.model.max_seq_len
    train_loader = TokenLoader(
        cfg.data_dir / "train.bin", cfg.batch_size, seq_len, device, cfg.seed
    )
    loaders = {
        "train": TokenLoader(
            cfg.data_dir / "train.bin", cfg.batch_size, seq_len, device, cfg.seed + 1
        ),
        "val": TokenLoader(cfg.data_dir / "val.bin", cfg.batch_size, seq_len, device, cfg.seed + 2),
    }

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
    log_file = (cfg.out_dir / "metrics.jsonl").open("a")
    run = None
    if cfg.wandb_project:
        import wandb  # pyright: ignore[reportMissingImports]

        run = wandb.init(project=cfg.wandb_project, name=cfg.out_dir.name, config=asdict(cfg))

    def log(record: dict) -> None:
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()
        if run is not None:
            run.log(record, step=record["step"])

    tokens_per_step = cfg.batch_size * cfg.grad_accum_steps * seq_len
    best_val = math.inf
    train_time = 0.0  # excludes evaluation and checkpointing
    steps_since_log = 0
    for step in range(cfg.max_steps + 1):
        last = step == cfg.max_steps
        if step % cfg.eval_interval == 0 or last:
            metrics = evaluate(model, loaders, cfg.eval_batches, autocast)
            if bytes_per_token:
                metrics["val_bpb"] = metrics["val_loss"] / math.log(2) / bytes_per_token
            log({"step": step, **metrics})
            print(f"step {step}: " + ", ".join(f"{k} {v:.4f}" for k, v in metrics.items()))
            if metrics["val_loss"] < best_val:
                best_val = metrics["val_loss"]
                torch.save(
                    {"model": model.state_dict(), "model_config": asdict(cfg.model), "step": step},
                    cfg.out_dir / "ckpt.pt",
                )
        if last:
            break

        mult = lr_multiplier(
            step, cfg.max_steps, cfg.warmup_steps, cfg.schedule, cfg.decay_frac, cfg.min_lr_ratio
        )
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * mult

        t0 = time.perf_counter()
        loss_sum = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = train_loader.batch()
            with autocast:
                _, loss = fwd(x, y)
            assert loss is not None
            (loss / cfg.grad_accum_steps).backward()
            loss_sum += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)
        train_time += time.perf_counter() - t0  # loss.item() above already synchronised
        steps_since_log += 1

        if step % cfg.log_interval == 0:
            record = {
                "step": step,
                "train_loss_step": loss_sum / cfg.grad_accum_steps,
                "lr_mult": mult,
                "grad_norm": grad_norm.item(),
                "tokens_per_sec": tokens_per_step * steps_since_log / train_time,
            }
            train_time, steps_since_log = 0.0, 0
            log(record)
            print(
                f"step {step}: loss {record['train_loss_step']:.4f}, "
                f"grad_norm {record['grad_norm']:.2f}, {record['tokens_per_sec']:,.0f} tok/s"
            )

    log_file.close()
    if run is not None:
        run.finish()
    return {"best_val_loss": best_val}


def main() -> None:
    train(tyro.cli(TrainConfig))


if __name__ == "__main__":
    main()
