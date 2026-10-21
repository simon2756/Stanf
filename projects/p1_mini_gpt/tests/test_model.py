import math

import pytest
import torch

from mini_gpt.model import GPT, GPTConfig, apply_rope, rope_cache


def tiny(**kw) -> GPTConfig:
    kw = {"n_kv_head": 4, **kw}
    return GPTConfig(vocab_size=97, max_seq_len=32, n_layer=2, n_head=4, d_model=32, **kw)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def test_shapes_and_initial_loss():
    model = GPT(tiny())
    x, y = torch.randint(0, 97, (2, 3, 16))
    logits, loss = model(x, y)
    assert logits.shape == (3, 16, 97)
    assert loss is not None
    # At init the prediction is close to uniform, so the loss on random targets is close to
    # ln(vocab). (With targets == inputs it is lower: tied embeddings favour the input token.)
    assert abs(loss.item() - math.log(97)) < 0.3


@pytest.mark.parametrize(
    "kw", [{}, {"pos_emb": "learned"}, {"qk_norm": False}, {"n_kv_head": 2}, {"n_kv_head": 1}]
)
def test_causality(kw):
    """Changing a future token must not change the logits at earlier positions."""
    model = GPT(tiny(**kw)).eval()
    x = torch.randint(0, 97, (1, 20))
    x2 = x.clone()
    x2[0, 12] = (x[0, 12] + 1) % 97
    a, _ = model(x)
    b, _ = model(x2)
    torch.testing.assert_close(a[:, :12], b[:, :12])
    assert not torch.allclose(a[:, 12:], b[:, 12:])


def test_rope_preserves_norm():
    cos, sin = rope_cache(head_dim=16, max_seq_len=64, theta=10_000.0)
    x = torch.randn(2, 3, 64, 16)
    torch.testing.assert_close(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1))


def test_rope_scores_depend_only_on_relative_position():
    cos, sin = rope_cache(head_dim=16, max_seq_len=64, theta=10_000.0)
    q, k = torch.randn(16), torch.randn(16)

    def score(m: int, n: int) -> torch.Tensor:
        qm = apply_rope(q.expand(64, 16), cos, sin)[m]
        kn = apply_rope(k.expand(64, 16), cos, sin)[n]
        return qm @ kn

    torch.testing.assert_close(score(10, 3), score(40, 33))
    torch.testing.assert_close(score(5, 5), score(50, 50))


def test_weight_tying():
    model = GPT(tiny())
    assert model.lm_head.weight is model.tok_emb.weight
    untied = GPT(tiny(tie_embeddings=False))
    assert untied.num_params(False) == model.num_params(False) + 97 * 32


def test_gqa_reduces_kv_params():
    full = GPT(tiny()).num_params()
    gqa = GPT(tiny(n_kv_head=1)).num_params()
    # k_proj and v_proj shrink from 32x32 to 32x8 in each of the 2 layers.
    assert full - gqa == 2 * 2 * (32 * 32 - 32 * 8)


def test_overfits_single_batch():
    model = GPT(tiny())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, 97, (4, 32))
    for _ in range(150):
        _, loss = model(x[:, :-1], x[:, 1:])
        assert loss is not None
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_generate_appends_tokens():
    model = GPT(tiny()).eval()
    out = model.generate(torch.zeros(2, 3, dtype=torch.long), max_new_tokens=40, temperature=0.0)
    # Generation continues past max_seq_len by cropping the context.
    assert out.shape == (2, 43)
