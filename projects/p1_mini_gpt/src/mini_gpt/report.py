"""Summarise a set of training runs: results table (Markdown) and validation-loss plot.

python -m mini_gpt.report runs/p1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_evals(run_dir: Path) -> list[dict]:
    lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    return [r for r in map(json.loads, lines) if "val_loss" in r]


def main(root: Path, out_dir: Path | None = None) -> str:
    out_dir = out_dir or root
    runs = {p.parent.name: load_evals(p.parent) for p in sorted(root.glob("*/metrics.jsonl"))}
    if not runs:
        raise SystemExit(f"no runs with metrics.jsonl under {root}")

    lines = [
        "| run | steps | best val loss | best val bits/byte | final train loss |",
        "|---|---|---|---|---|",
    ]
    for name, evals in runs.items():
        best = min(evals, key=lambda r: r["val_loss"])
        bpb = f"{best['val_bpb']:.4f}" if "val_bpb" in best else "–"
        lines.append(
            f"| {name} | {evals[-1]['step']} | {best['val_loss']:.4f} | {bpb} "
            f"| {evals[-1]['train_loss']:.4f} |"
        )
    table = "\n".join(lines) + "\n"
    (out_dir / "results.md").write_text(table)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, evals in runs.items():
        ax.plot([r["step"] for r in evals], [r["val_loss"] for r in evals], label=name)
    ax.set_xlabel("step")
    ax.set_ylabel("validation loss (nats/token)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "val_loss.png", dpi=150)
    print(table)
    return table


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/p1"))
