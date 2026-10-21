# Stanf: LLM engineering portfolio

Hands-on projects that follow Stanford CME 295 (Transformers & Large Language Models), with a focus on
multilingual, regulated use cases in Swiss finance.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full roadmap, timeline and skill matrix.

| # | Project | Status |
|---|---|---|
| P1 | [Swiss-multilingual mini-GPT from scratch](projects/p1_mini_gpt/) | in progress |
| P2 | LLM internals lab (GQA, MLA, RoPE/YaRN, MoE, sampling) | planned |
| P3 | Domain adaptation: financial document extraction (LoRA, DPO, distillation) | planned |
| P4 | RLVR for financial numeric reasoning (GRPO family) | planned |
| P5 | Inference performance lab (KV cache, speculative decoding, vLLM, Triton) | planned |
| P6 | Compliant research agent with MCP | planned |
| P7 | Evaluation harness (calibrated LLM-as-a-judge, agent evals) | planned |
| P8 | Masked diffusion LM vs autoregressive | planned |
| P9 | Charts and tables in PDFs (visual retrieval, VLMs) | planned |
| ★ | Flagship: HelvetiQ | planned |

## Setup

The repository is a [uv](https://docs.astral.sh/uv/) workspace; each project in `projects/` is a
package.

```bash
# NVIDIA GPU machine or macOS: torch from PyPI
uv sync --extra data

# Linux without a GPU (and CI): the much smaller CPU-only torch build.
# Pass the same extras to every `uv run`, otherwise uv switches torch back.
uv sync --extra cpu --extra data
uv run --extra cpu --extra data pytest
```

Checks that run in CI on every push and pull request:

```bash
uv run ruff check .          # lint
uv run ruff format --check . # formatting
uv run pyright               # types
uv run pytest                # tests
```

`uv run pre-commit install` runs ruff on every commit. New projects start from
[docs/PROJECT_README_TEMPLATE.md](docs/PROJECT_README_TEMPLATE.md).

Only public data is used in this repository.
