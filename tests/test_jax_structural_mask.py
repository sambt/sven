"""Tests for structural (whole-leaf) masking in the JAX ``SvenWrapper``.

The ``mask_by_block=True`` path differentiates only the selected pytree
leaves, so its Jacobian must equal the corresponding *columns* of the full
Jacobian, concatenated in ravel-offset order (the order ``mask_indices``
gathers ``flat_params``). Also covers ``apply_update`` masking,
``actual_param_fraction``, ``resample_every``, the legacy elementwise path,
and the per-selection compile cache.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from sven.jax import SvenWrapper

B = 12
DIMS = [6, 16, 16, 3]


def build_mlp(dims=None, seed=0):
    dims = DIMS if dims is None else dims
    n_layers = len(dims) - 1
    rng = np.random.default_rng(seed)
    params = {}
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        params[f"layer{i}"] = {
            "w": jnp.asarray(rng.standard_normal((a, b)) / np.sqrt(a)),
            "b": jnp.asarray(rng.standard_normal((b,)) * 0.1),
        }

    def apply_fn(params, x):
        h = x
        for i in range(n_layers):
            p = params[f"layer{i}"]
            h = h @ p["w"] + p["b"]
            if i < n_layers - 1:
                h = jnp.tanh(h)
        return h

    return params, apply_fn


def loss_fn(preds, y):
    return jnp.mean((preds - y) ** 2, axis=-1)


def make_data(seed=1):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((B, DIMS[0])))
    y = jnp.asarray(rng.standard_normal((B, DIMS[-1])))
    return x, y


@pytest.fixture(scope="module")
def setup():
    params, apply_fn = build_mlp()
    x, y = make_data()
    full = SvenWrapper(apply_fn, params, loss_fn)
    full.loss_and_grad((x, y))
    return params, apply_fn, x, y, np.asarray(full.jac)


def find_key(wrapper, predicate, n_tries=200):
    """First PRNG key whose leaf selection satisfies ``predicate(order)``."""
    for s in range(n_tries):
        key = jax.random.PRNGKey(s)
        if predicate(wrapper._select_block_leaves(key)):
            return key
    raise AssertionError("no key found satisfying predicate")


# ----------------------------------------------------------------------
# (1) block-masked Jacobian == columns of the full Jacobian
# ----------------------------------------------------------------------

def test_structural_jac_matches_full_columns(setup):
    params, apply_fn, x, y, J_full = setup
    w = SvenWrapper(
        apply_fn, params, loss_fn, param_fraction=0.5, mask_by_block=True
    )
    # A selection whose greedy INCLUSION order is out of ravel order: catches
    # implementations that concatenate per-leaf Jacobians in inclusion order
    # instead of ravel-offset order.
    key_out = find_key(w, lambda order: len(order) > 1 and list(order) != sorted(order))

    for key in (key_out, jax.random.PRNGKey(1001)):
        w.loss_and_grad((x, y), key=key)
        idx = np.asarray(w.mask_indices)
        assert np.all(np.diff(idx) > 0)  # sorted flat indices
        assert w.jac.shape == (B, idx.size)
        assert np.max(np.abs(np.asarray(w.jac) - J_full[:, idx])) < 1e-12


# ----------------------------------------------------------------------
# (2) apply_update writes exactly the masked entries
# ----------------------------------------------------------------------

def test_apply_update_masked_entries(setup):
    params, apply_fn, x, y, _ = setup
    w = SvenWrapper(
        apply_fn, params, loss_fn, param_fraction=0.4, mask_by_block=True
    )
    w.loss_and_grad((x, y), key=jax.random.PRNGKey(0))
    before = np.asarray(w.flat_params)
    w.apply_update(jnp.ones(w.n_active, dtype=w.flat_params.dtype))
    after = np.asarray(w.flat_params)
    mask = np.zeros(w.n_params, dtype=bool)
    mask[np.asarray(w.mask_indices)] = True
    assert np.allclose(after[mask] - before[mask], 1.0)
    assert np.array_equal(after[~mask], before[~mask])


# ----------------------------------------------------------------------
# (3) actual_param_fraction is sane
# ----------------------------------------------------------------------

def test_actual_param_fraction(setup):
    params, apply_fn, x, y, _ = setup
    frac = 0.5
    w = SvenWrapper(
        apply_fn, params, loss_fn, param_fraction=frac, mask_by_block=True
    )
    assert w.actual_param_fraction is None  # unset until first selection
    w.loss_and_grad((x, y), key=jax.random.PRNGKey(0))
    apf = w.actual_param_fraction
    assert 0.0 < apf <= frac  # greedy include-if-fits never exceeds the budget
    assert w.n_active == w.mask_indices.size == round(apf * w.n_params)

    # Budget below every leaf: falls back to the single smallest leaf.
    w_tiny = SvenWrapper(
        apply_fn, params, loss_fn, param_fraction=1e-6, mask_by_block=True
    )
    w_tiny.loss_and_grad((x, y), key=jax.random.PRNGKey(0))
    assert w_tiny.n_active == int(np.min(w_tiny._leaf_sizes))
    assert w_tiny.actual_param_fraction > 0.0


# ----------------------------------------------------------------------
# (4) resample_every keeps the mask for that many calls
# ----------------------------------------------------------------------

def test_resample_every(setup):
    params, apply_fn, x, y, _ = setup
    # 0.25: small enough that different permutations pick different leaf sets
    # (at 0.5 every permutation on this model selects the same set).
    w = SvenWrapper(
        apply_fn,
        params,
        loss_fn,
        param_fraction=0.25,
        mask_by_block=True,
        resample_every=3,
    )
    key0 = jax.random.PRNGKey(0)
    sel0 = tuple(sorted(w._select_block_leaves(key0)))
    key_alt = find_key(
        w, lambda order: tuple(sorted(order)) != sel0
    )

    w.loss_and_grad((x, y), key=key0)
    idx0 = np.asarray(w.mask_indices)
    w.loss_and_grad((x, y))  # reuse; key not needed
    assert np.array_equal(np.asarray(w.mask_indices), idx0)
    w.loss_and_grad((x, y), key=key_alt)  # reuse; key ignored
    assert np.array_equal(np.asarray(w.mask_indices), idx0)
    w.loss_and_grad((x, y), key=key_alt)  # 4th call resamples
    assert not np.array_equal(np.asarray(w.mask_indices), idx0)


# ----------------------------------------------------------------------
# (5) elementwise legacy path still matches full-Jacobian columns
# ----------------------------------------------------------------------

def test_elementwise_matches_full_columns(setup):
    params, apply_fn, x, y, J_full = setup
    w = SvenWrapper(apply_fn, params, loss_fn, param_fraction=0.3)
    w.loss_and_grad((x, y), key=jax.random.PRNGKey(7))
    idx = np.asarray(w.mask_indices)
    assert w.jac.shape == (B, idx.size)
    assert np.max(np.abs(np.asarray(w.jac) - J_full[:, idx])) < 1e-12


# ----------------------------------------------------------------------
# (6) compile cache: one entry (and one trace) per leaf selection
# ----------------------------------------------------------------------

def test_structural_compile_cache(setup):
    params, apply_fn, x, y, _ = setup
    w = SvenWrapper(
        apply_fn, params, loss_fn, param_fraction=0.25, mask_by_block=True
    )
    key = jax.random.PRNGKey(0)
    sel = tuple(sorted(w._select_block_leaves(key)))

    w.loss_and_grad((x, y), key=key)
    assert len(w._structural_cache) == 1
    fn1, _ = w._structural_cache[sel]
    w.loss_and_grad((x, y), key=key)  # same selection -> same compiled fn
    assert len(w._structural_cache) == 1
    fn2, _ = w._structural_cache[sel]
    assert fn1 is fn2
    if hasattr(fn1, "_cache_size"):
        # same fn object + same shapes -> jit reused the single trace
        assert fn1._cache_size() == 1

    key_alt = find_key(w, lambda order: tuple(sorted(order)) != sel)
    w.loss_and_grad((x, y), key=key_alt)  # new selection -> new cache entry
    assert len(w._structural_cache) == 2
