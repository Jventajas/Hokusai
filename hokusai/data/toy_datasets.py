"""2D toy distributions for easy debugging

Simple data shapes to test generation algorithms -- Each function returns a batch of shape [n, 2].
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def two_moons(
    rng_key: Array,
    num_points: int,
    noise: float = 0.1,
    center: tuple[float, float] = (0.5, 0.25),
    scale: float = 1.5,
) -> Array:
    """Two interleaving half-circles, recentered by `center` and `scale`.

    The default `center`/`scale` put the cloud near the origin with ~unit
    variance, matching the N(0, I) noise the flow integrates from.
    """
    upper_key, lower_key, noise_key = jax.random.split(rng_key, 3)
    num_upper = num_points // 2
    num_lower = num_points - num_upper

    upper_angle = jnp.pi * jax.random.uniform(upper_key, (num_upper,))
    upper_moon = jnp.stack([jnp.cos(upper_angle), jnp.sin(upper_angle)], axis=-1)

    lower_angle = jnp.pi * jax.random.uniform(lower_key, (num_lower,))
    lower_moon = jnp.stack(
        [1.0 - jnp.cos(lower_angle), 0.5 - jnp.sin(lower_angle)], axis=-1
    )

    points = jnp.concatenate([upper_moon, lower_moon], axis=0)
    points = points + noise * jax.random.normal(noise_key, points.shape)
    return (points - jnp.array(center)) * scale


def checkerboard(rng_key: Array, num_points: int) -> Array:
    """Points on a 4x4 checkerboard of occupied squares."""
    x_key, y_key = jax.random.split(rng_key)
    x = jax.random.uniform(x_key, (num_points,), minval=-2.0, maxval=2.0)
    raw_y = jax.random.uniform(y_key, (num_points,), minval=-2.0, maxval=2.0)
    # Each column is "on" for alternating rows; this parity shifts y into the
    # occupied square so x and y are on the same colour of the checkerboard.
    column_parity = jnp.floor(x) % 2
    y = raw_y - (jnp.floor(raw_y) % 2) + column_parity
    return jnp.stack([x, y], axis=-1)


DATASETS = {"moons": two_moons, "checkerboard": checkerboard}
