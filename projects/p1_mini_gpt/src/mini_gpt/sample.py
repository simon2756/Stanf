"""Generate text from a trained checkpoint.

python -m mini_gpt.sample --ckpt runs/baseline/ckpt.pt
python -m mini_gpt.sample --ckpt runs/baseline/ckpt.pt --prompts "Die Schweizerische Nationalbank"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

from mini_gpt.model import GPT, GPTConfig
from mini_gpt.tokenizer import BPETokenizer
from mini_gpt.train import resolve_device


@dataclass
class SampleConfig:
    ckpt: Path = Path("runs/default/ckpt.pt")
    tokenizer: Path = Path("data/tokenizer.json")
    prompts: tuple[str, ...] = (
        "Die Schweizerische Nationalbank",
        "La Banque nationale suisse",
        "La Banca nazionale svizzera",
        "The Swiss National Bank",
    )
    max_new_tokens: int = 80
    temperature: float = 0.8
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = 0.05
    seed: int = 0
    device: str = "auto"


def main() -> None:
    cfg = tyro.cli(SampleConfig)
    device = resolve_device(cfg.device)
    ckpt = torch.load(cfg.ckpt, map_location=device)
    model = GPT(GPTConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = BPETokenizer.load(cfg.tokenizer)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

    for prompt in cfg.prompts:
        idx = torch.tensor([tok.encode(prompt)], device=device)
        out = model.generate(
            idx,
            cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            min_p=cfg.min_p,
            eos_id=tok.eot_id,
            generator=gen,
        )
        print(f"--- {prompt}\n{tok.decode(out[0].tolist())}\n")


if __name__ == "__main__":
    main()
