"""How much does each language cost to process? A tokenizer comparison.

Two measurements per tokenizer and language:

* **Parallel token premium** on Belebele/FLORES passages, which are professional translations of
  the same English texts: tokens for the translation divided by tokens for the English original.
  1.30 means the same content costs 30% more (in API price, latency and context length).
* **Fertility** (tokens per whitespace-separated word) and **bytes per token** on web text from
  the validation split, including Swiss German and Romansh where available.

    python -m mini_gpt.tokenizer_study --ours data/tokenizer.json
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import tyro

from mini_gpt.data import read_docs
from mini_gpt.tokenizer import BPETokenizer

Encoder = Callable[[str], list[int]]

BELEBELE_CONFIGS = {"en": "eng_Latn", "de": "deu_Latn", "fr": "fra_Latn", "it": "ita_Latn"}


@dataclass
class StudyConfig:
    ours: Path | None = Path("data/tokenizer.json")
    tiktoken_encodings: tuple[str, ...] = ("cl100k_base", "o200k_base")
    """OpenAI encodings: cl100k_base (GPT-4), o200k_base (GPT-4o and later)."""
    hf_tokenizers: tuple[str, ...] = (
        "Qwen/Qwen3-0.6B",
        "swiss-ai/Apertus-8B-2509",
        "HuggingFaceTB/SmolLM3-3B",
    )
    raw_dir: Path = Path("data/raw")
    web_langs: tuple[str, ...] = ("en", "de", "fr", "it", "gsw", "rm")
    web_chars: int = 500_000
    out_dir: Path = Path("projects/p1_mini_gpt/results")


def load_encoders(cfg: StudyConfig) -> dict[str, Encoder]:
    encoders: dict[str, Encoder] = {}
    if cfg.ours is not None and cfg.ours.exists():
        ours = BPETokenizer.load(cfg.ours)
        encoders[f"mini-gpt ({ours.vocab_size // 1000}k)"] = ours.encode_ordinary
    for name in cfg.tiktoken_encodings:
        import tiktoken

        enc = tiktoken.get_encoding(name)
        encoders[name] = enc.encode_ordinary
    for repo in cfg.hf_tokenizers:
        try:
            from transformers import AutoTokenizer

            hf = AutoTokenizer.from_pretrained(repo)
        except Exception as e:  # gated, offline, renamed, ...
            print(f"skipping {repo}: {type(e).__name__}: {e}")
            continue
        encoders[repo.split("/")[-1]] = lambda s, hf=hf: hf.encode(s, add_special_tokens=False)
    return encoders


def belebele_passages() -> dict[str, list[str]]:
    """Unique FLORES passages per language, aligned by their source link."""
    from datasets import load_dataset

    by_lang: dict[str, dict[str, str]] = {}
    for lang, config in BELEBELE_CONFIGS.items():
        ds = load_dataset("facebook/belebele", config, split="test")
        by_lang[lang] = dict(zip(ds["link"], ds["flores_passage"], strict=True))
    common = sorted(set.intersection(*(set(d) for d in by_lang.values())))
    return {lang: [d[link] for link in common] for lang, d in by_lang.items()}


def web_sample(path: Path, n_chars: int) -> str:
    parts, total = [], 0
    for text in read_docs(path):
        parts.append(text)
        total += len(text)
        if total >= n_chars:
            break
    return "\n\n".join(parts)


def main() -> None:
    cfg = tyro.cli(StudyConfig)
    encoders = load_encoders(cfg)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    passages = belebele_passages()
    print(f"parallel set: {len(passages['en'])} passages per language")
    for name, enc in encoders.items():
        en_tokens = sum(len(enc(p)) for p in passages["en"])
        for lang, texts in passages.items():
            n_tokens = sum(len(enc(p)) for p in texts)
            rows.append(
                {
                    "tokenizer": name,
                    "lang": lang,
                    "source": "belebele",
                    "tokens": n_tokens,
                    "premium_vs_en": n_tokens / en_tokens,
                    "fertility": n_tokens / sum(len(p.split()) for p in texts),
                    "bytes_per_token": sum(len(p.encode()) for p in texts) / n_tokens,
                }
            )

    for lang in cfg.web_langs:
        path = cfg.raw_dir / f"{lang}_val.jsonl"
        if not path.exists():
            print(f"no web sample for {lang} ({path} missing)")
            continue
        text = web_sample(path, cfg.web_chars)
        for name, enc in encoders.items():
            n_tokens = len(enc(text))
            rows.append(
                {
                    "tokenizer": name,
                    "lang": lang,
                    "source": "web",
                    "tokens": n_tokens,
                    "premium_vs_en": None,
                    "fertility": n_tokens / len(text.split()),
                    "bytes_per_token": len(text.encode()) / n_tokens,
                }
            )

    with (cfg.out_dir / "tokenizer_study.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md = render_markdown(rows, list(encoders))
    (cfg.out_dir / "tokenizer_study.md").write_text(md)
    print(md)


def render_markdown(rows: list[dict], names: list[str]) -> str:
    def table(source: str, metric: str, fmt: str) -> str:
        langs = list(dict.fromkeys(r["lang"] for r in rows if r["source"] == source))
        if not langs:
            return "_no data_\n"
        lines = ["| tokenizer | " + " | ".join(langs) + " |", "|---" * (len(langs) + 1) + "|"]
        for name in names:
            vals = {
                r["lang"]: r[metric]
                for r in rows
                if r["source"] == source and r["tokenizer"] == name
            }
            cells = [format(vals[lang], fmt) if lang in vals else "–" for lang in langs]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"

    return (
        "## Token premium vs English (parallel Belebele/FLORES passages)\n\n"
        + table("belebele", "premium_vs_en", ".2f")
        + "\n## Fertility: tokens per word (web text)\n\n"
        + table("web", "fertility", ".2f")
        + "\n## Bytes per token (web text, higher = better compression)\n\n"
        + table("web", "bytes_per_token", ".2f")
    )


if __name__ == "__main__":
    main()
