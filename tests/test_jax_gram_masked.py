"""Correctness tests for stochastic parameter masking on the JAX Gram pipeline.

Masked ``SvenGram`` steps (mask_mode "rows"/"tensor"/"elementwise") must
reproduce the reference masked-Jacobian pipeline — ``SvenWrapper`` with the
SAME mask + ``pinv(mode="full")`` + ``_compute_delta`` — from group-bounded
Gram accumulation. Both pipelines draw masks through the samplers shared in
``sven.jax.wrapper``, so the same key forces identical ``mask_indices``
(asserted before every comparison). Groups with no masked columns must skip
their ``jacrev`` entirely.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from sven.jax import GramSvenWrapper, SvenGram, SvenWrapper, pinv
from sven.jax.sven import _compute_delta

B = 16
DIMS = [8, 32, 32, 4]
MODES = ("rows", "tensor", "elementwise")


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


def make_data(b=B, d_in=DIMS[0], d_out=DIMS[-1], seed=1, dup_scales=()):
    """Random data; rows 1, 2, ... optionally near-duplicate row 0 at the
    given scales, to force small singular values."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((b, d_in))
    y = rng.standard_normal((b, d_out))
    for i, scale in enumerate(dup_scales, start=1):
        x[i] = x[0] + scale * rng.standard_normal(d_in)
        y[i] = y[0] + scale * rng.standard_normal(d_out)
    return jnp.asarray(x), jnp.asarray(y)


def rel_err(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def masked_reference_delta(
    apply_fn, params, x, y, k, rtol, frac, mode, key, **wrapper_kw
):
    """delta via the masked-Jacobian pipeline; returns (delta, idx, J_A, S_inv)."""
    w = SvenWrapper(
        apply_fn, params, loss_fn, param_fraction=frac, mask_mode=mode, **wrapper_kw
    )
    w.loss_and_grad((x, y), key=key)
    VhT, S_inv, U_T = pinv(
        w.jac, k=k, key=jax.random.PRNGKey(0), rtol=rtol, mode="full"
    )
    delta = _compute_delta(U_T, S_inv, VhT, w.residuals)
    return (
        np.asarray(delta),
        np.asarray(w.mask_indices),
        np.asarray(w.jac),
        np.asarray(S_inv),
    )


def masked_gram_step(
    apply_fn, params, x, y, k, rtol, frac, mode, key,
    lr=1.0, group_numel=2**22, **wrapper_kw,
):
    """One real masked SvenGram step; returns (before, after, idx, gw)."""
    gw = GramSvenWrapper(
        apply_fn, params, loss_fn, param_fraction=frac, mask_mode=mode,
        group_numel=group_numel, **wrapper_kw,
    )
    opt = SvenGram(gw, lr=lr, k=k, rtol=rtol)
    before = np.asarray(gw.flat_params)
    gw.loss_and_grad((x, y), key=key)
    idx = np.asarray(gw.mask_indices)
    opt.step()
    return before, np.asarray(gw.flat_params), idx, gw


# ----------------------------------------------------------------------
# (1) masked SvenGram step vs the reference masked pipeline, per mode,
#     with k < M and the rtol truncation firing
# ----------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_masked_step_matches_reference(mode):
    params, apply_fn = build_mlp()
    # 3 near-duplicate rows put tiny singular values inside the top-k window
    x, y = make_data(dup_scales=(1e-5, 1e-6, 1e-7))
    k, rtol, frac = 15, 1e-3, 0.3
    key = jax.random.PRNGKey(11)
    delta_ref, idx_ref, _, S_inv = masked_reference_delta(
        apply_fn, params, x, y, k, rtol, frac, mode, key
    )
    assert k < B and np.sum(S_inv == 0) >= 2  # k < M and the rtol mask fires

    before, after, idx, gw = masked_gram_step(
        apply_fn, params, x, y, k, rtol, frac, mode, key
    )
    assert np.array_equal(idx, idx_ref)  # SAME selection, both pipelines
    assert 0 < idx.size < gw.n_params  # nontrivial mask
    delta_g = (before - after)[idx_ref]  # params += -lr * delta with lr=1
    assert rel_err(delta_g, delta_ref) < 1e-9
    frozen = np.ones(gw.n_params, dtype=bool)
    frozen[idx_ref] = False
    assert np.array_equal(after[frozen], before[frozen])  # only masked entries
    assert gw.gram is None and gw.residuals is None  # step consumed the state


# ----------------------------------------------------------------------
# (2) masked G == J_A @ J_A^T from the reference masked Jacobian
# ----------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_masked_gram_identity(mode):
    params, apply_fn = build_mlp()
    x, y = make_data()
    full = SvenWrapper(apply_fn, params, loss_fn)
    full.loss_and_grad((x, y))
    J = np.asarray(full.jac, dtype=np.float64)

    gw = GramSvenWrapper(
        apply_fn, params, loss_fn, param_fraction=0.3, mask_mode=mode
    )
    gw.loss_and_grad((x, y), key=jax.random.PRNGKey(2))
    assert gw.jac is None
    assert isinstance(gw.gram, np.ndarray) and gw.gram.dtype == np.float64
    assert gw.gram.shape == (B, B)
    j_a = J[:, np.asarray(gw.mask_indices)]
    assert np.max(np.abs(gw.gram - j_a @ j_a.T)) < 1e-10


# ----------------------------------------------------------------------
# (3) group skipping: empty groups never run (or compile) their jacrev
# ----------------------------------------------------------------------

def test_group_skipping_still_exact():
    params, apply_fn = build_mlp()
    x, y = make_data()
    k, rtol, frac = B, 1e-12, 0.25
    key = jax.random.PRNGKey(4)
    delta_ref, idx_ref, _, _ = masked_reference_delta(
        apply_fn, params, x, y, k, rtol, frac, "tensor", key
    )
    before, after, idx, gw = masked_gram_step(
        apply_fn, params, x, y, k, rtol, frac, "tensor", key, group_numel=300
    )
    assert len(gw._groups) >= 3
    assert np.array_equal(idx, idx_ref)

    # 'layer1.w' (1024 numel) never fits the 0.25 * 1476 tensor budget, so its
    # single-leaf group is always empty and must be skipped entirely.
    empty = [
        lo == hi
        for lo, hi in (
            np.searchsorted(idx, [g_start, g_end])
            for g_start, g_end in gw._group_spans
        )
    ]
    assert any(empty)
    if hasattr(gw._group_fns[0][1], "_cache_size"):
        for is_empty, (_, masked_fn, _) in zip(empty, gw._group_fns):
            # skipped groups never even compile their masked contraction
            assert masked_fn._cache_size() == (0 if is_empty else 1)
    assert rel_err((before - after)[idx], delta_ref) < 1e-9


# ----------------------------------------------------------------------
# (4) kappa / microbatch masked case (cross-terms nonzero)
# ----------------------------------------------------------------------

def test_masked_kappa_microbatch():
    params, apply_fn = build_mlp()
    x, y = make_data()
    kw = dict(kappa=1.4, microbatch_size=2)
    m = B // 2
    key = jax.random.PRNGKey(6)
    delta_ref, idx_ref, j_a, _ = masked_reference_delta(
        apply_fn, params, x, y, m, 1e-12, 0.3, "rows", key, **kw
    )
    assert j_a.shape[0] == m
    before, after, idx, _ = masked_gram_step(
        apply_fn, params, x, y, m, 1e-12, 0.3, "rows", key, **kw
    )
    assert np.array_equal(idx, idx_ref)
    assert rel_err((before - after)[idx], delta_ref) < 1e-9


# ----------------------------------------------------------------------
# (5) delta_from_w is exactly J_A^T w
# ----------------------------------------------------------------------

def test_masked_delta_from_w_is_jat_w():
    params, apply_fn = build_mlp()
    x, y = make_data()
    full = SvenWrapper(apply_fn, params, loss_fn)
    full.loss_and_grad((x, y))
    J = np.asarray(full.jac, dtype=np.float64)

    gw = GramSvenWrapper(
        apply_fn, params, loss_fn, param_fraction=0.3, mask_mode="elementwise"
    )
    gw.loss_and_grad((x, y), key=jax.random.PRNGKey(8))
    idx = np.asarray(gw.mask_indices)
    w = np.random.default_rng(5).standard_normal(B)
    delta = np.asarray(gw.delta_from_w(w))
    assert delta.shape == (idx.size,)
    assert rel_err(delta, J[:, idx].T @ w) < 1e-10


# ----------------------------------------------------------------------
# (6) param_fraction=1.0 ignores mask_mode (bitwise); guards
# ----------------------------------------------------------------------

def test_full_fraction_ignores_mask_mode():
    params, apply_fn = build_mlp()
    x, y = make_data()
    plain = GramSvenWrapper(apply_fn, params, loss_fn)
    masked = GramSvenWrapper(
        apply_fn, params, loss_fn, param_fraction=1.0, mask_mode="rows"
    )
    assert masked.mask_mode is None
    for gw in (plain, masked):
        gw.loss_and_grad((x, y))
        SvenGram(gw, lr=0.5, k=8, rtol=1e-8).step()
    assert masked.mask_indices is None
    assert np.array_equal(
        np.asarray(masked.flat_params), np.asarray(plain.flat_params)
    )  # bitwise


def test_masked_guards_raise():
    params, apply_fn = build_mlp()
    x, y = make_data()

    # param_fraction < 1 without an explicit mask_mode
    with pytest.raises(ValueError, match="mask_mode"):
        GramSvenWrapper(apply_fn, params, loss_fn, param_fraction=0.5)

    # masked loss_and_grad requires a key
    gw = GramSvenWrapper(
        apply_fn, params, loss_fn, param_fraction=0.5, mask_mode="rows"
    )
    with pytest.raises(ValueError, match="key"):
        gw.loss_and_grad((x, y))


# ----------------------------------------------------------------------
# (7) training smoke: masked gram steps still fit the toy regression
# ----------------------------------------------------------------------

def test_masked_training_smoke():
    params, apply_fn = build_mlp(dims=[1, 16, 16, 1], seed=3)
    x = jnp.asarray(np.linspace(-1.0, 1.0, 32).reshape(-1, 1))
    y = jnp.sin(3.0 * x)

    gw = GramSvenWrapper(
        apply_fn, params, loss_fn, param_fraction=0.5, mask_mode="rows"
    )
    opt = SvenGram(gw, lr=0.5, k=32, rtol=1e-3)
    key = jax.random.PRNGKey(0)
    key, sub = jax.random.split(key)
    losses0, _ = gw.loss_and_grad((x, y), key=sub)
    l0 = float(jnp.mean(losses0))
    opt.step()
    for _ in range(14):
        key, sub = jax.random.split(key)
        gw.loss_and_grad((x, y), key=sub)
        opt.step()
    lf = float(jnp.mean(gw.evaluate_and_loss(x, y)))
    assert lf < 0.5 * l0
