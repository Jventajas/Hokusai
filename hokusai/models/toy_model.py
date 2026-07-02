"""Toy velocity network: a small MLP mapping (x_t, t) -> velocity for 2D data.

`cond` is accepted and ignored, keeping the call signature stable for the
conditioned models that replace this one later.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

Array = jax.Array


def sinusoidal_time_embedding(
    t: Array,
    dim: int,
    *,
    max_period: float = 10000.0,
    time_scale: float = 1000.0,
) -> Array:
    """Sinusoidal embedding of scalar times, shape [B] -> [B, dim].

    `max_period` sets the slowest frequency; `time_scale` maps t in [0, 1] up to
    a wide angle range so the frequencies sweep many full cycles. `dim` must be
    even (validated by callers).
    """
    half = dim // 2
    freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(half) / half)
    angles = t[:, None] * freqs[None, :] * time_scale
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


class ToyVelocity(nnx.Module):
    """MLP mapping (x_t in R^d, t in [0,1]) -> velocity in R^d."""

    def __init__(
        self,
        *,
        data_dim: int = 2,
        hidden: int = 256,
        depth: int = 4,
        time_dim: int = 64,
        rngs: nnx.Rngs,
    ):
        if time_dim % 2:
            raise ValueError(f"time_dim must be even, got {time_dim}")
        self.time_dim = time_dim
        self.time_mlp = nnx.Linear(time_dim, hidden, rngs=rngs)

        layers = [nnx.Linear(data_dim + hidden, hidden, rngs=rngs)]
        for _ in range(depth - 1):
            layers.append(nnx.Linear(hidden, hidden, rngs=rngs))
        self.layers = nnx.List(layers)
        self.out = nnx.Linear(hidden, data_dim, rngs=rngs)

    def __call__(self, x_t: Array, t: Array, cond: Array | None = None) -> Array:
        del cond  # unused; kept for a stable signature.
        t_emb = nnx.gelu(self.time_mlp(sinusoidal_time_embedding(t, self.time_dim)))
        h = jnp.concatenate([x_t, t_emb], axis=-1)
        for layer in self.layers:
            h = nnx.gelu(layer(h))
        return self.out(h)
