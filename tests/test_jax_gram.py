"""Exactness tests for the JAX Gram/kernel-trick pipeline.

``SvenGram`` on a ``GramSvenWrapper`` must reproduce the reference update
from the full pipeline (``SvenWrapper`` + ``pinv(mode="full")`` +
``_compute_delta``) — including ``kappa``, microbatching, rtol truncation,
and multi-group Gram accumulation — without ever materialising the Jacobian.
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


def reference_delta(apply_fn, params, x, y, k, rtol, **wrapper_kw):
    """delta from the real full pipeline; returns (delta, J, S_inv)."""
    w = SvenWrapper(apply_fn, params, loss_fn, **wrapper_kw)
    w.loss_and_grad((x, y))
    VhT, S_inv, U_T = pinv(
        w.jac, k=k, key=jax.random.PRNGKey(0), rtol=rtol, mode="full"
    )
    delta = _compute_delta(U_T, S_inv, VhT, w.residuals)
    return np.asarray(delta), np.asarray(w.jac), np.asarray(S_inv)


def gram_delta(apply_fn, params, x, y, k, rtol, group_numel=2**22, **wrapper_kw):
    """delta recovered from one SvenGram step at lr=1; returns (delta, gw)."""
    gw = GramSvenWrapper(
        apply_fn, params, loss_fn, group_numel=group_numel, **wrapper_kw
    )
    opt = SvenGram(gw, lr=1.0, k=k, rtol=rtol)
    before = np.asarray(gw.flat_params)
    gw.loss_and_grad((x, y))
    opt.step()
    return before - np.asarray(gw.flat_params), gw


# ----------------------------------------------------------------------
# (1) SvenGram step vs the reference full pipeline
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "k,rtol,dup_scales",
    [
        (B, 1e-12, ()),           # full rank
        (6, 1e-12, ()),           # k < M
        (B, 1e-3, (1e-5, 1e-6)),  # rtol mask fires on near-duplicate rows
    ],
)
def test_step_matches_reference(k, rtol, dup_scales):
    params, apply_fn = build_mlp()
    x, y = make_data(dup_scales=dup_scales)
    delta_ref, _, S_inv = reference_delta(apply_fn, params, x, y, k, rtol)
    if dup_scales:
        # pinv masks at fixed rank k; the near-duplicates must trigger it.
        assert np.sum(S_inv == 0) >= len(dup_scales)
    delta_g, gw = gram_delta(apply_fn, params, x, y, k, rtol)
    assert rel_err(delta_g, delta_ref) < 1e-9
    # step consumed the state
    assert gw.gram is None and gw.residuals is None


# ----------------------------------------------------------------------
# (2) kappa / microbatch cases (cross-terms nonzero)
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "wrapper_kw",
    [
        dict(kappa=1.4),
        dict(microbatch_size=2),
        dict(kappa=1.4, microbatch_size=2),
    ],
)
def test_kappa_and_microbatch(wrapper_kw):
    params, apply_fn = build_mlp()
    x, y = make_data()
    m = B // wrapper_kw.get("microbatch_size", 1)
    delta_ref, J, _ = reference_delta(
        apply_fn, params, x, y, k=m, rtol=1e-12, **wrapper_kw
    )
    assert J.shape[0] == m
    delta_g, gw = gram_delta(apply_fn, params, x, y, k=m, rtol=1e-12, **wrapper_kw)
    assert rel_err(delta_g, delta_ref) < 1e-9


# ----------------------------------------------------------------------
# (3) G == J @ J.T
# ----------------------------------------------------------------------

def test_gram_equals_jjt():
    params, apply_fn = build_mlp()
    x, y = make_data()
    w = SvenWrapper(apply_fn, params, loss_fn)
    w.loss_and_grad((x, y))
    J = np.asarray(w.jac, dtype=np.float64)

    gw = GramSvenWrapper(apply_fn, params, loss_fn)
    losses, _ = gw.loss_and_grad((x, y))
    assert gw.jac is None
    assert isinstance(gw.gram, np.ndarray) and gw.gram.dtype == np.float64
    assert gw.gram.shape == (B, B)
    assert np.max(np.abs(gw.gram - J @ J.T)) < 1e-10
    assert np.max(np.abs(np.asarray(losses) - np.asarray(w.losses))) < 1e-14


# ----------------------------------------------------------------------
# (4) small group_numel forces multiple groups, still exact
# ----------------------------------------------------------------------

def test_multi_group_exact():
    params, apply_fn = build_mlp()
    x, y = make_data()
    delta_ref, J, _ = reference_delta(apply_fn, params, x, y, k=B, rtol=1e-12)

    delta_g, gw = gram_delta(
        apply_fn, params, x, y, k=B, rtol=1e-12, group_numel=300
    )
    assert len(gw._groups) >= 3
    # the partition covers every leaf exactly once
    covered = sorted(i for g in gw._groups for i in g)
    assert covered == list(range(gw._n_leaves))
    assert rel_err(delta_g, delta_ref) < 1e-9

    gw2 = GramSvenWrapper(apply_fn, params, loss_fn, group_numel=300)
    gw2.loss_and_grad((x, y))
    assert np.max(np.abs(gw2.gram - J @ J.T)) < 1e-10


# ----------------------------------------------------------------------
# (5) training smoke
# ----------------------------------------------------------------------

def test_training_smoke():
    params, apply_fn = build_mlp(dims=[1, 16, 16, 1], seed=3)
    x = jnp.asarray(np.linspace(-1.0, 1.0, 32).reshape(-1, 1))
    y = jnp.sin(3.0 * x)

    gw = GramSvenWrapper(apply_fn, params, loss_fn)
    opt = SvenGram(gw, lr=0.5, k=32, rtol=1e-3)
    losses0, _ = gw.loss_and_grad((x, y))
    l0 = float(jnp.mean(losses0))
    opt.step()
    for _ in range(19):
        gw.loss_and_grad((x, y))
        opt.step()
    lf = float(jnp.mean(gw.evaluate_and_loss(x, y)))
    assert lf < 0.5 * l0


# ----------------------------------------------------------------------
# (6) delta_from_w is exactly J^T w
# ----------------------------------------------------------------------

def test_delta_from_w_is_jt_w():
    params, apply_fn = build_mlp()
    x, y = make_data()
    w_full = SvenWrapper(apply_fn, params, loss_fn)
    w_full.loss_and_grad((x, y))
    J = np.asarray(w_full.jac, dtype=np.float64)

    gw = GramSvenWrapper(apply_fn, params, loss_fn)
    gw.loss_and_grad((x, y))
    w = np.random.default_rng(5).standard_normal(B)
    delta = np.asarray(gw.delta_from_w(w))
    assert rel_err(delta, J.T @ w) < 1e-10


# ----------------------------------------------------------------------
# Guards
# ----------------------------------------------------------------------

def test_rmsprop_pre_not_implemented():
    params, apply_fn = build_mlp()
    gw = GramSvenWrapper(apply_fn, params, loss_fn)
    with pytest.raises(NotImplementedError):
        SvenGram(gw, lr=0.1, k=8, use_rmsprop=True, rmsprop_post=False)


def test_step_requires_loss_and_grad():
    params, apply_fn = build_mlp()
    gw = GramSvenWrapper(apply_fn, params, loss_fn)
    opt = SvenGram(gw, lr=0.1, k=8)
    with pytest.raises(RuntimeError):
        opt.step()
