"""Byte-level BPE tokenizer, written from scratch.

The design follows GPT-2/GPT-4 style tokenizers:

1. Text is split into chunks with a regex (pre-tokenization), so merges never cross word,
   number or punctuation boundaries.
2. Each chunk is turned into its UTF-8 bytes. The 256 byte values are the base vocabulary,
   so any string can be encoded and there is no unknown token.
3. Training repeatedly merges the most frequent adjacent pair of tokens into a new token.
4. Special tokens such as ``<|endoftext|>`` are appended after the learned merges.

Training keeps pair counts up to date incrementally and only revisits the chunks that contain
the merged pair, which makes it fast enough for corpora of tens of millions of characters.
"""

from __future__ import annotations

import heapq
import itertools
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import regex

# Split pattern of OpenAI's cl100k_base tokenizer (GPT-4). It keeps letters, numbers (in
# groups of up to three digits), punctuation and whitespace apart. Note that the contraction
# rule only covers English ("'s", "'re", ...); French elisions such as "l'homme" are split
# at the apostrophe like any other punctuation.
GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*"""
    r"""|\s*[\r\n]|\s+(?!\S)|\s+"""
)

EOT = "<|endoftext|>"

Pair = tuple[int, int]


def _merge(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` in ``ids`` with ``new_id``."""
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(
        self,
        merges: dict[Pair, int] | None = None,
        special_tokens: dict[str, int] | None = None,
        pattern: str = GPT4_SPLIT_PATTERN,
    ) -> None:
        self.merges: dict[Pair, int] = dict(merges or {})
        self.special_tokens: dict[str, int] = dict(special_tokens or {})
        self.pattern = pattern
        self._compiled = regex.compile(pattern)
        self.vocab = self._build_vocab()
        self.inverse_special = {i: s for s, i in self.special_tokens.items()}
        self._special_re = (
            regex.compile("(" + "|".join(regex.escape(s) for s in self.special_tokens) + ")")
            if self.special_tokens
            else None
        )
        self._cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------------ training

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        special_tokens: Iterable[str] = (EOT,),
        pattern: str = GPT4_SPLIT_PATTERN,
        verbose: bool = False,
    ) -> BPETokenizer:
        special_tokens = list(special_tokens)
        num_merges = vocab_size - 256 - len(special_tokens)
        if num_merges < 0:
            raise ValueError(f"vocab_size must be at least {256 + len(special_tokens)}")

        # Count each distinct chunk once; training then works on (chunk, frequency) pairs.
        compiled = regex.compile(pattern)
        chunk_counts: Counter[str] = Counter()
        for text in texts:
            chunk_counts.update(compiled.findall(text))

        words: list[list[int]] = [list(c.encode("utf-8")) for c in chunk_counts]
        freqs: list[int] = list(chunk_counts.values())

        pair_counts: dict[Pair, int] = defaultdict(int)
        where: dict[Pair, set[int]] = defaultdict(set)
        for wi, (w, f) in enumerate(zip(words, freqs, strict=True)):
            for pair in itertools.pairwise(w):
                pair_counts[pair] += f
                where[pair].add(wi)

        # Max-heap with lazy deletion: entries are re-pushed whenever a count changes and
        # stale entries are skipped when popped. Ties go to the smallest pair (deterministic).
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        merges: dict[Pair, int] = {}
        for i in range(num_merges):
            best: Pair | None = None
            while heap:
                neg_count, pair = heapq.heappop(heap)
                if pair_counts.get(pair, 0) == -neg_count and neg_count < 0:
                    best = pair
                    break
            if best is None:  # nothing left to merge (tiny corpus)
                break
            new_id = 256 + i
            merges[best] = new_id

            changed: set[Pair] = set()
            for wi in list(where[best]):
                w = words[wi]
                f = freqs[wi]
                for pair in itertools.pairwise(w):
                    pair_counts[pair] -= f
                    changed.add(pair)
                w = _merge(w, best, new_id)
                words[wi] = w
                for pair in itertools.pairwise(w):
                    pair_counts[pair] += f
                    where[pair].add(wi)
                    changed.add(pair)
            del where[best]
            pair_counts.pop(best, None)
            changed.discard(best)
            for pair in changed:
                c = pair_counts[pair]
                if c > 0:
                    heapq.heappush(heap, (-c, pair))
                else:
                    pair_counts.pop(pair, None)

            if verbose and (i + 1) % 1000 == 0:
                print(f"merge {i + 1}/{num_merges}")

        first_special = 256 + len(merges)
        specials = {s: first_special + j for j, s in enumerate(special_tokens)}
        return cls(merges=merges, special_tokens=specials, pattern=pattern)

    # ------------------------------------------------------------------ encode / decode

    def _build_vocab(self) -> dict[int, bytes]:
        vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in self.merges.items():  # dicts keep insertion (= merge) order
            vocab[new_id] = vocab[a] + vocab[b]
        for s, i in self.special_tokens.items():
            vocab[i] = s.encode("utf-8")
        return vocab

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    @property
    def eot_id(self) -> int:
        return self.special_tokens[EOT]

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Apply the earliest-learned merge first, exactly as during training.
            pair = min(itertools.pairwise(ids), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        if len(self._cache) < 1_000_000:
            self._cache[chunk] = ids
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text, treating special-token strings as normal text."""
        ids: list[int] = []
        for chunk in self._compiled.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        if not allow_special or self._special_re is None:
            return self.encode_ordinary(text)
        ids: list[int] = []
        for part in self._special_re.split(text):
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def token_strings(self, ids: Iterable[int]) -> list[str]:
        """Human-readable pieces, handy for inspecting how a word gets split."""
        return [self.vocab[i].decode("utf-8", errors="replace") for i in ids]

    # ------------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> None:
        data = {
            "type": "byte_bpe",
            "pattern": self.pattern,
            "merges": [[a, b] for (a, b) in self.merges],
            "special_tokens": self.special_tokens,
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = {(a, b): 256 + i for i, (a, b) in enumerate(data["merges"])}
        return cls(merges=merges, special_tokens=data["special_tokens"], pattern=data["pattern"])
