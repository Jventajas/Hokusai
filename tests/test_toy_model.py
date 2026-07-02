import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from hokusai.models.toy_model import ToyVelocity, sinusoidal_time_embedding


def test_embedding_shape_and_range():
    t = jnp.linspace(0.0, 1.0, 8)
    emb = sinusoidal_time_embedding(t, dim=16)
    assert emb.shape == (8, 16)
    assert jnp.all(jnp.abs(emb) <= 1.0)


def test_model_rejects_odd_time_dim():
    with pytest.raises(ValueError, match="time_dim must be even"):
        ToyVelocity(time_dim=15, rngs=nnx.Rngs(0))


def test_model_forward_shape():
    model = ToyVelocity(data_dim=2, rngs=nnx.Rngs(0))
    x_t = jax.random.normal(jax.random.key(0), (4, 2))
    t = jnp.zeros((4,))
    assert model(x_t, t).shape == (4, 2)
