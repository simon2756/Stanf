"""Corpus pipeline: download -> train tokenizer -> tokenize into binary shards.

    python -m mini_gpt.data download --chars-per-lang 25_000_000
    python -m mini_gpt.data train-tokenizer --vocab-size 16384
    python -m mini_gpt.data tokenize

Sources (all public, streamed from the Hugging Face Hub):
* German, French, Italian: FineWeb-2 (filtered Common Crawl, per-language subsets)
* English: FineWeb-Edu (educational-quality filtered Common Crawl)
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tyro

from mini_gpt.tokenizer import EOT, BPETokenizer

SOURCES: dict[str, tuple[str, str]] = {
    "de": ("HuggingFaceFW/fineweb-2", "deu_Latn"),
    "fr": ("HuggingFaceFW/fineweb-2", "fra_Latn"),
    "it": ("HuggingFaceFW/fineweb-2", "ita_Latn"),
    "en": ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
    # Not used for training by default, but available for the tokenizer study.
    "gsw": ("HuggingFaceFW/fineweb-2", "gsw_Latn"),  # Swiss German
    "rm": ("HuggingFaceFW/fineweb-2", "roh_Latn"),  # Romansh
}


def read_docs(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]


# ---------------------------------------------------------------------------- download


@dataclass
class Download:
    """Stream documents per language and write train/val JSONL files."""

    out_dir: Path = Path("data/raw")
    langs: tuple[str, ...] = ("de", "fr", "it", "en")
    chars_per_lang: int = 25_000_000
    val_chars_per_lang: int = 500_000
    min_doc_chars: int = 200

    def run(self) -> None:
        from datasets import load_dataset

        self.out_dir.mkdir(parents=True, exist_ok=True)
        for lang in self.langs:
            repo, config = SOURCES[lang]
            ds = load_dataset(repo, name=config, split="train", streaming=True)
            # Validation documents come first in the stream, so they are fixed across runs
            # regardless of how much training data is downloaded.
            budgets = [("val", self.val_chars_per_lang), ("train", self.chars_per_lang)]
            stream = iter(ds)
            for split, budget in budgets:
                path = self.out_dir / f"{lang}_{split}.jsonl"
                n_chars = n_docs = 0
                with path.open("w", encoding="utf-8") as f:
                    for row in stream:
                        text = row["text"].strip()
                        if len(text) < self.min_doc_chars:
                            continue
                        f.write(json.dumps({"text": text, "lang": lang}, ensure_ascii=False))
                        f.write("\n")
                        n_chars += len(text)
                        n_docs += 1
                        if n_chars >= budget:
                            break
                print(f"{lang} {split}: {n_docs:,} docs, {n_chars:,} chars -> {path}")
            # Close the half-consumed stream now; leaving it to interpreter shutdown can hang.
            stream.close()


# ---------------------------------------------------------------------------- tokenizer


@dataclass
class TrainTokenizer:
    """Train the byte-level BPE tokenizer on an equal-size sample of each language."""

    raw_dir: Path = Path("data/raw")
    out_path: Path = Path("data/tokenizer.json")
    langs: tuple[str, ...] = ("de", "fr", "it", "en")
    vocab_size: int = 16384
    sample_chars_per_lang: int = 5_000_000

    def run(self) -> None:
        def sample() -> Iterator[str]:
            for lang in self.langs:
                n = 0
                for text in read_docs(self.raw_dir / f"{lang}_train.jsonl"):
                    yield text
                    n += len(text)
                    if n >= self.sample_chars_per_lang:
                        break

        tok = BPETokenizer.train(sample(), vocab_size=self.vocab_size, verbose=True)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        tok.save(self.out_path)
        print(f"saved tokenizer with {tok.vocab_size:,} tokens -> {self.out_path}")


# ---------------------------------------------------------------------------- tokenize

_worker_tok: BPETokenizer | None = None


def _init_worker(path: str) -> None:
    global _worker_tok
    _worker_tok = BPETokenizer.load(path)


def _encode_doc(text: str) -> tuple[list[int], int]:
    assert _worker_tok is not None
    return _worker_tok.encode_ordinary(text), len(text.encode("utf-8"))


@dataclass
class Tokenize:
    """Encode all documents into flat uint16 token files (``train.bin``, ``val.bin``).

    Documents are separated by the end-of-text token. ``meta.json`` stores the vocab size and
    the bytes-per-token ratio of the validation set, which the trainer uses to report
    bits-per-byte (a loss that is comparable across tokenizers).
    """

    raw_dir: Path = Path("data/raw")
    tokenizer_path: Path = Path("data/tokenizer.json")
    out_dir: Path = Path("data/tokens")
    langs: tuple[str, ...] = ("de", "fr", "it", "en")
    workers: int = field(default_factory=lambda: os.cpu_count() or 1)

    def run(self) -> None:
        tok = BPETokenizer.load(self.tokenizer_path)
        if tok.vocab_size > 2**16:
            raise ValueError("uint16 storage needs vocab_size <= 65536")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        meta: dict = {"vocab_size": tok.vocab_size, "eot_id": tok.special_tokens[EOT]}

        with Pool(
            self.workers, initializer=_init_worker, initargs=(str(self.tokenizer_path),)
        ) as pool:
            for split in ("train", "val"):
                path = self.out_dir / f"{split}.bin"
                n_tokens = n_bytes = 0
                per_lang: dict[str, dict[str, int]] = {}
                with path.open("wb") as f:
                    for lang in self.langs:
                        docs = read_docs(self.raw_dir / f"{lang}_{split}.jsonl")
                        lang_tokens = lang_bytes = 0
                        for ids, nb in pool.imap(_encode_doc, docs, chunksize=64):
                            arr = np.array([*ids, tok.eot_id], dtype=np.uint16)
                            arr.tofile(f)
                            lang_tokens += len(arr)
                            lang_bytes += nb
                        per_lang[lang] = {"tokens": lang_tokens, "bytes": lang_bytes}
                        n_tokens += lang_tokens
                        n_bytes += lang_bytes
                meta[split] = {"tokens": n_tokens, "bytes": n_bytes, "per_lang": per_lang}
                print(f"{split}: {n_tokens:,} tokens, {n_bytes / n_tokens:.2f} bytes/token")

        meta["val_bytes_per_token"] = meta["val"]["bytes"] / meta["val"]["tokens"]
        (self.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    cmd = tyro.extras.subcommand_cli_from_dict(
        {"download": Download, "train-tokenizer": TrainTokenizer, "tokenize": Tokenize}
    )
    cmd.run()


if __name__ == "__main__":
    main()
