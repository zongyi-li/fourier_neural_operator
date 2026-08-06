import pytest
import torch

from ..embeddings import ConditioningEmbedding


@pytest.mark.parametrize("type_t", ["sinusoidal", "power"])
@pytest.mark.parametrize("n_params", [1, 3, 5])
def test_output_shape_and_width(type_t, n_params):
    embed_dim = 8
    emb = ConditioningEmbedding(embed_dim=embed_dim, n_params=n_params, type_t=type_t)
    assert emb.out_channels == n_params * embed_dim
    assert emb.out_dim == n_params * embed_dim

    B = 4
    # positive t so the power branch is valid too
    t = torch.rand(B, n_params) + 0.1
    e = emb(t)
    assert e.shape == (B, n_params * embed_dim)
    assert torch.isfinite(e).all()


def test_sinusoidal_accepts_nonpositive_t():
    emb = ConditioningEmbedding(embed_dim=8, n_params=2, type_t="sinusoidal")
    t = torch.tensor([[-1.0, 0.0], [-3.5, 2.0]])
    e = emb(t)
    assert e.shape == (2, 16)
    assert torch.isfinite(e).all()


def test_sinusoidal_at_zero_is_sin0_cos1():
    """At t=0 the sinusoidal features are [sin(0)=0..., cos(0)=1...]."""
    embed_dim = 8
    emb = ConditioningEmbedding(embed_dim=embed_dim, n_params=1, type_t="sinusoidal")
    e = emb(torch.zeros(1, 1))
    half = embed_dim // 2
    torch.testing.assert_close(e[0, :half], torch.zeros(half))
    torch.testing.assert_close(e[0, half:embed_dim], torch.ones(half))


def test_power_rejects_nonpositive_t():
    emb = ConditioningEmbedding(embed_dim=8, n_params=2, type_t="power")
    with pytest.raises(ValueError, match=r"requires t > 0"):
        emb(torch.tensor([[1.0, -0.5], [2.0, 3.0]]))


def test_unknown_type_raises():
    with pytest.raises(ValueError, match="type_t"):
        ConditioningEmbedding(embed_dim=8, type_t="bogus")


def test_differentiable_in_t():
    emb = ConditioningEmbedding(embed_dim=8, n_params=2, type_t="sinusoidal")
    t = torch.randn(3, 2, requires_grad=True)
    emb(t).sum().backward()
    assert t.grad is not None
    assert torch.isfinite(t.grad).all()


def test_vector_is_per_param_concat():
    """The P-param embedding is exactly the concatenation of P scalar embeds."""
    embed_dim = 8
    emb1 = ConditioningEmbedding(embed_dim=embed_dim, n_params=1)
    emb3 = ConditioningEmbedding(embed_dim=embed_dim, n_params=3)
    t = torch.tensor([[0.2, -1.0, 3.0]])
    joint = emb3(t)
    parts = [emb1(t[:, p:p + 1]) for p in range(3)]
    torch.testing.assert_close(joint, torch.cat(parts, dim=-1))
