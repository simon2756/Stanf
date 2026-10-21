import pytest

from mini_gpt.tokenizer import EOT, BPETokenizer

CORPUS = [
    "Die Schweizerische Nationalbank führt die Geldpolitik als unabhängige Zentralbank.",
    "La Banque nationale suisse mène la politique monétaire en tant que banque centrale.",
    "La Banca nazionale svizzera conduce la politica monetaria del Paese.",
    "The Swiss National Bank conducts the country's monetary policy. 1234567 CHF",
] * 20


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    return BPETokenizer.train(CORPUS, vocab_size=400)


def test_vocab_size_and_special_tokens(tok: BPETokenizer):
    assert tok.vocab_size == 400
    assert tok.eot_id == 399
    assert len(tok.merges) == 400 - 256 - 1


@pytest.mark.parametrize(
    "text",
    [
        "Grüezi mitenand! Chuchichäschtli.",
        "L'économie suisse, ça marche ?",
        "Perché è così?",
        "emoji 🚴‍♂️ and CJK 銀行 and tabs\tand\nnewlines  ",
        "",
    ],
)
def test_roundtrip_unseen_text(tok: BPETokenizer, text: str):
    assert tok.decode(tok.encode(text)) == text


def test_training_compresses(tok: BPETokenizer):
    text = CORPUS[0]
    assert len(tok.encode(text)) < len(text.encode("utf-8")) / 2


def test_first_merge_is_most_frequent_pair():
    tok = BPETokenizer.train(["aaab aaab aaab cd"], vocab_size=257, special_tokens=())
    # "aa" occurs twice per "aaab" chunk (overlapping count), more often than any other pair.
    assert list(tok.merges) == [(ord("a"), ord("a"))]


def test_training_is_deterministic():
    a = BPETokenizer.train(CORPUS, vocab_size=320)
    b = BPETokenizer.train(CORPUS, vocab_size=320)
    assert a.merges == b.merges


def test_special_tokens(tok: BPETokenizer):
    ids = tok.encode(f"Hallo{EOT}Salut")
    assert ids.count(tok.eot_id) == 1
    # When special tokens are not allowed, the string is encoded as plain text.
    assert tok.eot_id not in tok.encode(f"Hallo{EOT}", allow_special=False)


def test_save_load(tok: BPETokenizer, tmp_path):
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    text = "Die Nationalbank und la Banque nationale"
    assert loaded.encode(text) == tok.encode(text)
    assert loaded.special_tokens == tok.special_tokens


def test_merges_do_not_cross_word_boundaries(tok: BPETokenizer):
    for piece in tok.token_strings(tok.encode("Nationalbank Geldpolitik")):
        assert piece.strip() == piece or piece.startswith(" ")
        assert " " not in piece.strip()
