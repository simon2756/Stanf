import itertools
import json

import numpy as np
import pytest
import torch

from mini_gpt.model import GPTConfig
from mini_gpt.optim import lr_multiplier, newton_schulz_orthogonalize
from mini_gpt.sampling import min_p_filter, sample_next, top_k_filter, top_p_filter
from mini_gpt.train import TrainConfig, train

LOGITS = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))


def kept(logits: torch.Tensor) -> list[int]:
    return torch.isfinite(logits[0]).nonzero().flatten().tolist()


def test_top_k():
    assert kept(top_k_filter(LOGITS, 2)) == [0, 1]
    assert kept(top_k_filter(LOGITS, 10)) == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("p", "expected"), [(0.5, [0]), (0.7, [0, 1]), (0.9, [0, 1, 2]), (1.0, [0, 1, 2, 3])]
)
def test_top_p(p, expected):
    assert kept(top_p_filter(LOGITS, p)) == expected


def test_min_p_adapts_to_confidence():
    # Threshold = 0.2 * 0.5 = 0.1, so 0.05 is dropped.
    assert kept(min_p_filter(LOGITS, 0.2)) == [0, 1, 2]
    peaked = torch.log(torch.tensor([[0.97, 0.01, 0.01, 0.01]]))
    assert kept(min_p_filter(peaked, 0.2)) == [0]


def test_greedy_and_sampling_distribution():
    assert sample_next(LOGITS, temperature=0.0).item() == 0
    gen = torch.Generator().manual_seed(0)
    draws = sample_next(LOGITS.expand(20_000, 4), generator=gen)
    freq = torch.bincount(draws, minlength=4).float() / 20_000
    torch.testing.assert_close(freq, LOGITS.exp()[0], atol=0.02, rtol=0)


def test_wsd_schedule():
    m = [
        lr_multiplier(s, max_steps=100, warmup_steps=10, schedule="wsd", decay_frac=0.2)
        for s in range(101)
    ]
    assert m[0] == pytest.approx(0.1)
    assert all(v == 1.0 for v in m[10:80])
    assert m[100] == 0.0
    assert all(a >= b for a, b in itertools.pairwise(m[80:]))


def test_cosine_schedule():
    m = [lr_multiplier(s, 100, 0, "cosine", min_lr_ratio=0.1) for s in range(101)]
    assert m[0] == 1.0
    assert m[50] == pytest.approx(0.55)
    assert m[100] == pytest.approx(0.1)


def test_newton_schulz_is_roughly_orthogonal():
    G = torch.randn(64, 128)
    X = newton_schulz_orthogonalize(G).float()
    s = torch.linalg.svdvals(X)
    assert s.min() > 0.5 and s.max() < 1.3


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_train_end_to_end(tmp_path, optimizer):
    """A few optimizer steps on random tokens: loss goes down, checkpoint and logs exist."""
    data = tmp_path / "tokens"
    data.mkdir()
    rng = np.random.default_rng(0)
    pattern = rng.integers(0, 50, size=64)
    np.tile(pattern, 200).astype(np.uint16).tofile(data / "train.bin")  # learnable
    np.tile(pattern, 20).astype(np.uint16).tofile(data / "val.bin")
    (data / "meta.json").write_text(json.dumps({"vocab_size": 50, "val_bytes_per_token": 4.0}))

    cfg = TrainConfig(
        data_dir=data,
        out_dir=tmp_path / "run",
        model=GPTConfig(max_seq_len=32, n_layer=2, n_head=2, n_kv_head=2, d_model=32),
        batch_size=8,
        max_steps=60,
        warmup_steps=5,
        eval_interval=30,
        eval_batches=2,
        optimizer=optimizer,
        device="cpu",
    )
    result = train(cfg)
    lines = [
        json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    ]
    evals = [r for r in lines if "val_loss" in r]
    assert evals[-1]["val_loss"] < evals[0]["val_loss"] - 1.0
    assert "val_bpb" in evals[-1]
    assert (tmp_path / "run" / "ckpt.pt").exists()
    assert result["best_val_loss"] == min(r["val_loss"] for r in evals)
