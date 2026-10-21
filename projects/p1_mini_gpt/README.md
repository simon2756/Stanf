# P1: Swiss-multilingual mini-GPT from scratch

> A byte-level BPE tokenizer and a modern decoder-only transformer, written from scratch in PyTorch
> and trained on German, French, Italian and English web text. A tokenizer trained on balanced
> data cuts the "non-English token premium" from about 1.6× (GPT-4, Qwen3) to 1.2×.

Part of the [LLM portfolio](../../README.md) · follows CME 295 Lecture 1 (Transformers).

## Problem

1. **Understand every line** of a current LLM architecture by building and training one.
2. **Measure what tokenization costs multilingual users.** Swiss companies work in German, French
   and Italian. If a tokenizer needs 60% more tokens for the same content in French than in English,
   French documents cost 60% more in API fees and latency and fill the context window 60% faster.

## Approach

### Tokenizer ([`tokenizer.py`](src/mini_gpt/tokenizer.py))

- Byte-level BPE as in GPT-2/GPT-4: GPT-4 regex pre-tokenization, 256 byte base tokens (no unknown
  token), merges learned by frequency, `<|endoftext|>` special token.
- Incremental training: pair counts are updated only for chunks that contain the merged pair, with a
  lazy max-heap. **16k merges on 20M characters take about 1 minute on one CPU core.**
- Trained on an equal share of each language (5M characters each).

### Model ([`model.py`](src/mini_gpt/model.py))

The current open-model standard rather than the 2017 original:

| Component | 2017 Transformer | This model (as in Llama/Qwen-style LLMs) |
|---|---|---|
| Normalisation | post-norm LayerNorm | pre-norm **RMSNorm** |
| Positions | sinusoidal, added to input | **RoPE** on queries and keys |
| Attention logits | unnormalised | **QK-norm** (RMSNorm on q and k) |
| Feed-forward | ReLU MLP | **SwiGLU** |
| Biases | yes | none |
| Embeddings | separate | input/output **tied** |
| KV heads | = query heads | configurable (**GQA** ready for P2) |

### Training ([`train.py`](src/mini_gpt/train.py), [`optim.py`](src/mini_gpt/optim.py))

- AdamW (decay on matrices only) or **Muon** (Newton-Schulz orthogonalised momentum for hidden
  matrices, AdamW for the rest)
- **Warmup-stable-decay** (WSD) or cosine schedule
- bf16 autocast, `torch.compile`, gradient accumulation, gradient clipping
- Reports **bits per byte** besides loss per token, so models with different tokenizers can be
  compared fairly

### Sampling ([`sampling.py`](src/mini_gpt/sampling.py))

Temperature, top-k, top-p and **min-p**, each covered by unit tests.

### Tests ([`tests/`](tests/))

Tokenizer round-trips on unseen text (Swiss German, accents, emoji, CJK), causality (a future token
never changes past logits, for every model variant), RoPE properties (norm preservation, scores
depend only on relative position), GQA parameter counts, overfitting a batch, sampling filters,
schedules, Newton-Schulz orthogonality, and an end-to-end training run for both optimizers.

## Data

| Language | Source | Train | Validation |
|---|---|---|---|
| German | FineWeb-2 `deu_Latn` | 6M chars | 0.5M chars |
| French | FineWeb-2 `fra_Latn` | 6M chars | 0.5M chars |
| Italian | FineWeb-2 `ita_Latn` | 6M chars | 0.5M chars |
| English | FineWeb-Edu `sample-10BT` | 6M chars | 0.5M chars |
| Swiss German, Romansh | FineWeb-2 `gsw_Latn`, `roh_Latn` | tokenizer study only | 0.5M chars |

That is about 6.5M training tokens with the 16k tokenizer, enough for the CPU smoke run and a
small GPU run. For the full GPU runs, download more (`--chars-per-lang 250_000_000`).

## Results

### Tokenizer study ([`results/tokenizer_study.md`](results/tokenizer_study.md))

**Token premium vs English on parallel text** (Belebele/FLORES: 488 professionally translated
passages, so the content is identical across languages):

| Tokenizer | Vocab | DE | FR | IT |
|---|---|---|---|---|
| **mini-gpt (this project)** | 16k | **1.22** | **1.25** | **1.20** |
| Apertus-8B (Swiss AI) | 131k | 1.30 | 1.31 | 1.39 |
| o200k_base (GPT-4o) | 200k | 1.31 | 1.36 | 1.47 |
| Qwen3 | 151k | 1.56 | 1.57 | 1.63 |
| SmolLM3 | 128k | 1.58 | 1.59 | 1.64 |
| cl100k_base (GPT-4) | 100k | 1.58 | 1.59 | 1.64 |

**Bytes per token on web text** (higher = more text per token):

| Tokenizer | EN | DE | FR | IT | Swiss German | Romansh |
|---|---|---|---|---|---|---|
| mini-gpt (16k) | 3.82 | 3.67 | 3.72 | 3.74 | 2.69 | 2.86 |
| Apertus-8B | 4.64 | 4.03 | 4.26 | 3.90 | 2.97 | 3.23 |
| o200k_base | 4.82 | 4.31 | 4.32 | 3.94 | 3.15 | 3.51 |
| Qwen3 | 4.67 | 3.56 | 3.73 | 3.48 | 2.81 | 3.05 |

Findings:

- With GPT-4's and Qwen3's tokenizers the same content costs **56–64% more tokens** in German,
  French or Italian than in English. With GPT-4o's tokenizer and Swiss Apertus it is 30–47%.
- A tokenizer trained on balanced data almost removes the premium (1.20–1.25×) **with an eight to
  twelve times smaller vocabulary**. The price is worse absolute compression in English, since the
  vocabulary is no longer spent mainly on English.
- **Swiss German and Romansh are expensive for every tokenizer** (2.7–3.5 bytes/token, against 4.6–4.8 for
  English with the large tokenizers), most likely because there is little of either language in
  their training data.
- Practical consequence: cost and context-length estimates for a Swiss multilingual application
  should be made per language, and the tokenizer is a real factor when choosing a model.

### Training

**CPU smoke run** (7.4M parameters, 3.3M tokens, about 12 minutes on 4 CPU cores): validation loss
falls from 9.76 to 5.88 (2.27 bits/byte), and samples stay in the prompt's language with plausible
word forms but no meaning yet. Details, loss curve and samples:
[`results/cpu_smoke.md`](results/cpu_smoke.md). This only confirms that the pipeline works end to
end; the real experiments need a GPU.

**To do (GPU):** baseline vs Muon vs cosine vs learned positions vs no QK-norm vs GQA, using
[`scripts/ablations.sh`](scripts/ablations.sh). Results table and loss curves come from
`python -m mini_gpt.report runs/p1`.

## What did not work / surprises

- A first sanity test expected the initial loss to be ln(vocab) with `targets = inputs`. It was
  noticeably lower: with tied embeddings, an untrained model already favours predicting the
  input token itself, because `h · e_token` is largest for the token's own embedding. The test now
  uses random targets.
- Streaming datasets left half-read Parquet streams open, and the download script hung at
  interpreter exit. Closing each stream explicitly fixed it.

## Reproduce

From the repository root:

```bash
uv sync --extra data                      # add --extra cpu on a machine without a GPU
uv run python -m mini_gpt.data download --chars-per-lang 6_000_000
uv run python -m mini_gpt.data download --langs gsw rm --chars-per-lang 100_000
uv run python -m mini_gpt.data train-tokenizer --vocab-size 16384
uv run python -m mini_gpt.data tokenize
uv run python -m mini_gpt.tokenizer_study
uv run python -m mini_gpt.train --out-dir runs/p1/baseline
uv run python -m mini_gpt.sample --ckpt runs/p1/baseline/ckpt.pt
```

All commands take `--help`.

## Next steps

- [ ] GPU runs and ablations (Kaggle / Colab are enough at this size)
- [ ] Word2vec warm-up notebook (skip-gram with negative sampling) for the embeddings part of L1
- [ ] Add Swiss sources (SNB, admin.ch) to the corpus
- [ ] Compare the 16k tokenizer with 32k and 64k: does the premium stay low as English compression
      improves?

## Versions

Python 3.13, PyTorch 2.14, run on 2026-10-21. Tokenizers: Hugging Face Hub versions on that date.
