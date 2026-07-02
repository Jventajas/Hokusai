import jax
import jax.numpy as jnp
import pytest

from hokusai.data.toy_datasets import DATASETS, checkerboard, two_moons


@pytest.mark.parametrize("dataset_fn", DATASETS.values(), ids=DATASETS.keys())
def test_same_key_is_deterministic(dataset_fn):
    # A data generator must be reproducible for a fixed seed.
    key = jax.random.key(0)
    assert jnp.array_equal(dataset_fn(key, 128), dataset_fn(key, 128))


@pytest.mark.parametrize("num_points", [1, 7, 128])
def test_two_moons_row_count(num_points):
    # The upper/lower split floors, so odd counts must still total num_points.
    points = two_moons(jax.random.key(0), num_points)
    assert points.shape == (num_points, 2)


def test_checkerboard_points_stay_on_one_colour():
    # The parity trick guarantees x and y land on same-colour squares, i.e.
    # floor(x) + floor(y) is always even. This is the invariant that breaks
    # if the shift logic is wrong.
    points = checkerboard(jax.random.key(0), 4096)
    cell_parity = (jnp.floor(points[:, 0]) + jnp.floor(points[:, 1])) % 2
    assert jnp.all(cell_parity == 0)
