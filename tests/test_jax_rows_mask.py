"""Tests for rows-mode (leading-axis slice) masking in the JAX ``SvenWrapper``.

``mask_mode="rows"`` differentiates a random per-leaf leading-axis slice
selection, so its Jacobian must equal the corresponding *columns* of the full
Jacobian, concatenated in ravel-offset order (the order ``mask_indices``
gathers ``flat_params``). Slice counts are deterministic per fraction, so all
shapes are fixed and the Jacobian function compiles exactly once — indices
are traced data. Also covers the end-to-end ``Sven.step``, kappa +
microbatching, resampling without retraces, mask coverage,
``actual_param_fraction``, and the legacy elementwise/tensor paths.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from sven.jax import Sven, SvenWrapper, pinv
from sven.jax.sven import _compute_delta

B = 10
D_IN, D_OUT = 5, 3


def build_model(seed=0):
    """MLP whose params dict traps leaf-order bugs.

    Insertion order ("z_first", "m_mid", "a_out") differs from the sorted key
    order JAX flattens dicts in ("a_out", "m_mid", "z_first"), so iterating
    insertion order instead of ravel order scrambles the columns. "m_mid"
    holds a 3-d leaf used through a reshape, and every leaf has a different
    leading dimension.
    """
    rng = np.random.default_rng(seed)
    params = {}
    params["z_first"] = {
        "w": jnp.asarray(rng.standard_normal((D_IN, 7)) / np.sqrt(D_IN)),
        "b": jnp.asarray(rng.standard_normal((7,)) * 0.1),
    }
    params["m_mid"] = {
        "w3": jnp.asarray(rng.standard_normal((7, 4, 2)) / np.sqrt(7)),
    }
    params["a_out"] = {
        "w": jnp.asarray(rng.standard_normal((8, D_OUT)) / np.sqrt(8)),
        "b": jnp.asarray(rng.standard_normal((D_OUT,)) * 0.1),
    }

    def apply_fn(params, x):
        h = jnp.tanh(x @ params["z_first"]["w"] + params["z_first"]["b"])
        w3 = params["m_mid"]["w3"]
        h = jnp.tanh(h @ w3.reshape(w3.shape[0], -1))
        return h @ params["a_out"]["w"] + params["a_out"]["b"]

    return params, apply_fn


def loss_fn(preds, y):
    return jnp.mean((preds - y) ** 2, axis=-1)


def make_data(seed=1):
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.standard_normal((B, D_IN)))
    y = jnp.asarray(rng.standard_normal((B, D_OUT)))
    return x, y


@pytest.fixture(scope="module")
def setup():
    params, apply_fn = build_model()
    x, y = make_data()
    full = SvenWrapper(apply_fn, params, loss_fn)
    full.loss_and_grad((x, y))
    return params, apply_fn, x, y, np.asarray(full.jac)


# ----------------------------------------------------------------------
# (1) rows-mode Jacobian == columns of the full Jacobian, ravel order
# ----------------------------------------------------------------------

def test_rows_jac_matches_full_columns(setup):
    params, apply_fn, x, y, J_full = setup
    w = SvenWrapper(
        apply_fn, params, loss_fn, param_fraction=0.4, mask_mode="rows"
    )
    for s in range(4):
        w.loss_and_grad((x, y), key=jax.random.PRNGKey(s))
        idx = np.asarray(w.mask_indices)
        assert np.all(np.diff(idx) > 0)  # sorted flat indices
        assert w.jac.shape == (B, idx.size)
        assert idx.size == w.n_active
        # every leaf contributes at least one slice — the ordering trap bites
        for st, sz in zip(w._leaf_starts, w._leaf_sizes):
            assert np.any((idx >= st) & (idx < st + sz))
        assert np.max(np.abs(np.asarray(w.jac) - J_full[:, idx])) < 1e-12


# ----------------------------------------------------------------------
# (2) end-to-end Sven.step == hand pinv update on the masked Jacobian
# ----------------------------------------------------------------------

def test_rows_step_matches_manual_pinv(setup):
    params, apply_fn, x, y, _ = setup
    lr, k, rtol = 0.1, 6, 1e-10
    w = SvenWrapper(
        apply_fn, params, loss_fn, param_fraction=0.4, mask_mode="rows"
    )
    opt = Sven(w, lr=lr, k=k, rtol=rtol, svd_mode="full")
    w.loss_and_grad((x, y), key=jax.random.PRNGKey(3))
    before = np.asarray(w.flat_params)
    idx = np.asarray(w.mask_indices)

    # Hand-computed update from the masked Jacobian (step() consumes .jac)
    VhT, S_inv, U_T = pinv(
        w.jac, k=k, key=jax.random.PRNGKey(0), rtol=rtol, mode="full"
    )
    delta = np.asarray(_compute_delta(U_T, S_inv, VhT, w.residuals))
    expected = before.copy()
    expected[idx] -= lr * delta

    opt.step()
    after = np.asarray(w.flat_params)
    assert np.linalg.norm(after - expected) / np.linalg.norm(expected) < 1e-12
    mask = np.zeros(before.size, dtype=bool)
    mask[idx] = True
    assert np.array_equal(after[~mask], before[~mask])  # only masked entries move
    assert not np.array_equal(after[mask], before[mask])


# ----------------------------------------------------------------------
# (3) kappa != 2 combined with microbatching
# ----------------------------------------------------------------------

def test_rows_kappa_microbatch(setup):
    params, apply_fn, x, y, _ = setup
    kw = dict(kappa=1.4, microbatch_size=2)
    full = SvenWrapper(apply_fn, params, loss_fn, **kw)
    full.loss_and_grad((x, y))
    J_full = np.asarray(full.jac)
    assert J_full.shape[0] == B // 2

    w = SvenWrapper(
        apply_fn, params, loss_fn, param_fraction=0.4, mask_mode="rows", **kw
    )
    w.loss_and_grad((x, y), key=jax.random.PRNGKey(5))
    idx = np.asarray(w.mask_indices)
    assert w.jac.shape == (B // 2, idx.size)
    assert np.max(np.abs(np.asarray(w.jac) - J_full[:, idx])) < 1e-12
    r_expect = np.asarray(w.losses) ** 0.7
    assert np.max(np.abs(np.asarray(w.residuals) - r_expect)) < 1e-14


# ----------------------------------------------------------------------
# (4) deterministic slice counts: fixed shapes, no retrace on resample
# ----------------------------------------------------------------------

def test_rows_deterministic_shapes_no_retrace(setup):
    params, apply_fn, x, y, _ = setup
    w = SvenWrapper(
        apply_fn, params, loss_fn, param_fraction=0.3, mask_mode="rows"
    )
    w.loss_and_grad((x, y), key=jax.random.PRNGKey(0))
    shape0 = w.jac.shape
    idx0 = np.asarray(w.mask_indices)
    for s in range(1, 50):
        w.loss_and_grad((x, y), key=jax.random.PRNGKey(s))
        if not np.array_equal(np.asarray(w.mask_indices), idx0):
            break
    else:
        raise AssertionError("no key found with a different selection")
    assert w.jac.shape == shape0  # counts deterministic per fraction
    assert w.mask_indices.size == idx0.size
    if hasattr(w._jac_fn, "_cache_size"):
        # indices are traced data: one trace covers every selection
        assert w._jac_fn._cache_size() == 1


# ----------------------------------------------------------------------
# (5) resampled masks cover (nearly) all parameters
# ----------------------------------------------------------------------

def test_rows_coverage_union(setup):
    params, apply_fn, x, y, _ = setup
    w = SvenWrapper(
        apply_fn, params, loss_fn, param_fraction=0.25, mask_mode="rows"
    )
    covered = np.zeros(w.n_params, dtype=bool)
    masks = []
    for s in range(20):
        w.loss_and_grad((x, y), key=jax.random.PRNGKey(s))
        idx = np.asarray(w.mask_indices)
        masks.append(idx)
        covered[idx] = True
    assert any(not np.array_equal(masks[0], m) for m in masks[1:])
    assert covered.sum() > 0.9 * w.n_params  # every parameter reachable


# ----------------------------------------------------------------------
# (6) achieved fraction (deterministic, set at construction)
# ----------------------------------------------------------------------

def test_rows_actual_param_fraction(setup):
    params, apply_fn, x, y, _ = setup
    for frac in (0.25, 0.5, 0.6):
        w = SvenWrapper(
            apply_fn, params, loss_fn, param_fraction=frac, mask_mode="rows"
        )
        assert 0.5 * frac <= w.actual_param_fraction <= 1.5 * frac
        w.loss_and_grad((x, y), key=jax.random.PRNGKey(1))
        assert w.n_active == w.mask_indices.size
        assert w.actual_param_fraction == w.mask_indices.size / w.n_params


# ----------------------------------------------------------------------
# (7) legacy paths unchanged; mask_mode wiring and validation
# ----------------------------------------------------------------------

def test_legacy_paths_and_mask_mode_wiring(setup):
    params, apply_fn, x, y, J_full = setup
    # elementwise via the default derivation
    w_el = SvenWrapper(apply_fn, params, loss_fn, param_fraction=0.3)
    assert w_el.mask_mode == "elementwise" and not w_el.mask_by_block
    w_el.loss_and_grad((x, y), key=jax.random.PRNGKey(7))
    idx = np.asarray(w_el.mask_indices)
    assert np.max(np.abs(np.asarray(w_el.jac) - J_full[:, idx])) < 1e-12

    # explicit tensor mode == legacy mask_by_block selection under the same key
    key = jax.random.PRNGKey(3)
    w_t = SvenWrapper(
        apply_fn, params, loss_fn, param_fraction=0.5, mask_mode="tensor"
    )
    assert w_t.mask_by_block
    w_legacy = SvenWrapper(
        apply_fn, params, loss_fn, param_fraction=0.5, mask_by_block=True
    )
    assert w_legacy.mask_mode == "tensor"
    w_t.loss_and_grad((x, y), key=key)
    w_legacy.loss_and_grad((x, y), key=key)
    assert np.array_equal(
        np.asarray(w_t.mask_indices), np.asarray(w_legacy.mask_indices)
    )
    idx = np.asarray(w_t.mask_indices)
    assert np.max(np.abs(np.asarray(w_t.jac) - J_full[:, idx])) < 1e-12

    with pytest.raises(ValueError, match="mask_mode"):
        SvenWrapper(apply_fn, params, loss_fn, mask_mode="banana")
