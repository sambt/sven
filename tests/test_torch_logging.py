"""C-L1 / C-T3: what the optimizers log, and what the flags cost.

Every case is float64 on CPU on a small MLP, so the spectrum and the update
can be compared with the explicit Jacobian exactly:

* ``svd_info["svs"]`` is the FULL spectrum (all M values, before the k / rtol
  cut) and equals ``torch.linalg.svdvals`` of the explicit Jacobian, on the
  Gram path and on the classic ``pinv`` path;
* the applied update equals ``J^T U diag(1/s^2) utr`` restricted to the kept
  directions, with the logged ``utr`` and ``svs``;
* ``log_this_step=False`` records nothing but ``num_nonzero_svs`` and leaves
  the parameter update bit-identical;
* ``empty_cache=False`` (now the default) skips every
  ``torch.cuda.empty_cache()`` call and leaves the parameter update
  bit-identical;
* the record is host-side and ``np.asarray``-able even off CPU, and a
  non-logged step costs no more device->host transfers than the step needs
  for correctness (the two things a CPU-only suite would otherwise miss);
* with float32 parameters the Gram route's usable spectral range stops at
  ``sv_noise_floor``, which the record carries.
"""

from __future__ import annotations

import copy
import os
import sys
from contextlib import contextmanager
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn as nn

from sven.nn import GramSvenWrapper, SvenWrapper
from sven.opt import Sven, SvenGram, SvenGramReg
from sven.opt import sven as sven_opt_module

DT = torch.float64
DEVICE = "cpu"
LR = 0.25


def per_sample_mse(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((pred - y) ** 2).mean(dim=1)


def make_mlp(
    dims: list[int] = [4, 16, 16, 3], seed: int = 0, dtype: torch.dtype = DT
) -> nn.Sequential:
    torch.manual_seed(seed)
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers).to(dtype)


def make_data(
    b: int = 8, d_in: int = 4, d_out: int = 3, seed: int = 1,
    dup_scales: tuple[float, ...] = (), dtype: torch.dtype = DT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random data; rows 1.. optionally near-duplicate row 0, which forces
    tiny singular values so the rtol truncation rule actually fires."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, d_in, generator=g, dtype=DT)
    y = torch.randn(b, d_out, generator=g, dtype=DT)
    for i, scale in enumerate(dup_scales, start=1):
        x[i] = x[0] + scale * torch.randn(d_in, generator=g, dtype=DT)
        y[i] = y[0] + scale * torch.randn(d_out, generator=g, dtype=DT)
    return x.to(dtype), y.to(dtype)


def explicit_jacobian(
    template: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The (M, P) Jacobian and the (M,) residual rows at the same parameters."""
    wrapper = SvenWrapper(copy.deepcopy(template), per_sample_mse, DEVICE)
    wrapper.loss_and_grad((x, y))
    return wrapper.grads.detach().clone(), wrapper.residuals.detach().clone()


def run_gram_step(
    template: nn.Module, x: torch.Tensor, y: torch.Tensor, k: int, rtol: float,
    capture: str = "hooks", **opt_kwargs,
) -> tuple[torch.Tensor, torch.Tensor, SvenGram]:
    """One SvenGram step.  Returns (applied update, the Gram it solved, opt)."""
    wrapper = GramSvenWrapper(copy.deepcopy(template), per_sample_mse, DEVICE, capture=capture)
    p0 = wrapper.params.detach().clone()
    wrapper.loss_and_grad((x, y))
    gram = wrapper.gram.detach().clone()
    opt = SvenGram(wrapper, lr=LR, k=k, rtol=rtol, track_svd_info=True, **opt_kwargs)
    opt.step()
    return (p0 - wrapper.params.detach()) / LR, gram, opt


def run_classic_step(
    template: nn.Module, x: torch.Tensor, y: torch.Tensor, k: int, rtol: float,
    **opt_kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Sven]:
    """One classic Sven step.  Returns (update, J, residuals, opt)."""
    wrapper = SvenWrapper(copy.deepcopy(template), per_sample_mse, DEVICE)
    p0 = wrapper.params.detach().clone()
    wrapper.loss_and_grad((x, y))
    jac = wrapper.grads.detach().clone()
    resid = wrapper.residuals.detach().clone()
    opt = Sven(wrapper, lr=LR, k=k, rtol=rtol, track_svd_info=True,
               svd_mode="torch", **opt_kwargs)
    opt.step((x, y))
    return (p0 - wrapper.params.detach()) / LR, jac, resid, opt


def run_steps(
    cls: type, template: nn.Module, x: torch.Tensor, y: torch.Tensor,
    n_steps: int = 3, k: int = 8, rtol: float = 1e-12, log: bool = True,
    **opt_kwargs,
):
    """``n_steps`` steps of ``cls``; returns (final params, optimizer)."""
    if cls is Sven:
        wrapper = SvenWrapper(copy.deepcopy(template), per_sample_mse, DEVICE)
    else:
        wrapper = GramSvenWrapper(copy.deepcopy(template), per_sample_mse, DEVICE)
    opt = cls(wrapper, lr=LR, k=k, rtol=rtol, track_svd_info=True, **opt_kwargs)
    opt.log_this_step = log
    for _ in range(n_steps):
        wrapper.loss_and_grad((x, y))
        opt.step((x, y))
    return wrapper.params.detach().clone(), opt


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


def assert_spectrum_matches(svs: torch.Tensor, reference: torch.Tensor) -> None:
    """``svs`` equals ``svdvals(J)`` to within the Gram route's own precision.

    Both paths take sigma from an ``eigh`` of ``J J^T``, which squares the
    condition number: the absolute error on sigma_i goes as
    ``eps * sigma_max^2 / (2 sigma_i)``, i.e. a relative error
    ``eps (sigma_max/sigma_i)^2 / 2``.  For a well-conditioned batch that is
    ~eps, so the flat 1e-9 dominates; a spectrum spanning five decades
    (near-duplicate rows) loses its tail to this even in float64.  See
    ``test_float32_spectrum_trustworthy_only_above_the_noise_floor`` for the
    float32 consequence, which is what the campaign actually runs.
    """
    eps = torch.finfo(svs.dtype).eps
    gram_err = 4.0 * eps * reference[0] ** 2 / reference.clamp_min(eps * reference[0])
    ok = (svs - reference).abs() <= 1e-9 * reference + 1e-12 + gram_err
    assert bool(ok.all()), (
        f"svs deviates beyond the Gram route's precision:\n"
        f"  svs      = {svs.tolist()}\n  svdvals  = {reference.tolist()}\n"
        f"  budget   = {(1e-9 * reference + 1e-12 + gram_err).tolist()}"
    )


# ----------------------------------------------------------------------
# (1) svs is the full, untruncated spectrum (F17/F20)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("capture", ["hooks", "chunked"])
def test_gram_svs_match_svdvals(capture):
    template, (x, y) = make_mlp(), make_data()
    jac, _ = explicit_jacobian(template, x, y)
    # k / rtol truncate hard, yet the logged spectrum must be the full one
    _, _, opt = run_gram_step(template, x, y, k=4, rtol=1e-1, capture=capture)
    info = opt.finalize_svd_info()
    svs = torch.as_tensor(info["svs"][0], dtype=DT)
    assert info["step"].tolist() == [0]
    assert svs.shape == (x.shape[0],)  # all M values, not the k kept ones
    assert_spectrum_matches(svs, torch.linalg.svdvals(jac))
    assert int(info["num_nonzero_svs"][0]) < x.shape[0]  # the cut did fire
    # sv_min_kept is the smallest INVERTED sigma, not sigma_M (F20)
    n_kept = int(info["num_nonzero_svs"][0])
    assert info["sv_min_kept"][0] == pytest.approx(float(svs[n_kept - 1]), rel=1e-12)
    assert info["sv_min_kept"][0] > float(svs[-1])


def test_classic_svs_match_svdvals():
    template, (x, y) = make_mlp(), make_data()
    _, jac, _, opt = run_classic_step(template, x, y, k=4, rtol=1e-1)
    info = opt.finalize_svd_info()
    svs = torch.as_tensor(info["svs"][0], dtype=DT)
    assert svs.shape == (x.shape[0],)
    assert_spectrum_matches(svs, torch.linalg.svdvals(jac))
    assert int(info["num_nonzero_svs"][0]) == 4
    assert info["sv_min_kept"][0] == pytest.approx(float(svs[3]), rel=1e-9)


# ----------------------------------------------------------------------
# (2) the applied update == J^T U diag(1/s^2) utr on the kept directions
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "k,rtol,dup_scales",
    [(8, 1e-12, ()), (5, 1e-12, ()), (8, 1e-3, (1e-4, 1e-5))],
    ids=["k_full", "k_small", "rtol_trunc"],
)
def test_gram_update_equals_utr_formula(k, rtol, dup_scales):
    template = make_mlp()
    x, y = make_data(dup_scales=dup_scales)
    jac, resid = explicit_jacobian(template, x, y)
    update, gram, opt = run_gram_step(template, x, y, k=k, rtol=rtol)
    info = opt.finalize_svd_info()

    # the optimizer's own eigenbasis: same input, same LAPACK call
    sigma, _, evecs = Sven._spectrum_from_gram(gram.to(DT))
    utr = torch.as_tensor(info["utr"][0], dtype=DT)
    torch.testing.assert_close(utr, evecs.T @ resid, rtol=1e-10, atol=1e-14)
    assert_spectrum_matches(
        torch.as_tensor(info["svs"][0], dtype=DT), torch.linalg.svdvals(jac)
    )

    n = int(info["num_nonzero_svs"][0])
    assert 0 < n <= min(k, x.shape[0])
    w = evecs[:, :n] @ (utr[:n] / sigma[:n] ** 2)
    assert rel_err(update, jac.T @ w) < 1e-9
    # update_norm is the APPLIED change ||lr * update||; `update` is pre-lr
    assert info["update_norm"][0] == pytest.approx(LR * float(update.norm()), rel=1e-9)
    assert info["resid_norm"][0] == pytest.approx(float(resid.norm()), rel=1e-12)
    if dup_scales:
        assert n < x.shape[0]  # the rtol rule actually truncated


@pytest.mark.parametrize("k", [8, 5])
def test_classic_update_equals_utr_formula(k):
    template, (x, y) = make_mlp(), make_data()
    update, jac, resid, opt = run_classic_step(template, x, y, k=k, rtol=1e-12)
    info = opt.finalize_svd_info()

    # the classic path logs the spectrum of a float64 eigh of J J^T
    sigma, _, evecs = Sven._spectrum_from_gram(Sven._gram_fp64(jac))
    utr = torch.as_tensor(info["utr"][0], dtype=DT)
    torch.testing.assert_close(utr, evecs.T @ resid, rtol=1e-10, atol=1e-14)

    n = int(info["num_nonzero_svs"][0])
    assert n == k
    w = evecs[:, :n] @ (utr[:n] / sigma[:n] ** 2)
    assert rel_err(update, jac.T @ w) < 1e-8
    assert info["update_norm"][0] == pytest.approx(LR * float(update.norm()), rel=1e-9)


def test_variable_k_logs_the_applied_update():
    """Under variable_k the logged norms describe the accepted components."""
    template, (x, y) = make_mlp(), make_data()
    wrapper = SvenWrapper(copy.deepcopy(template), per_sample_mse, DEVICE)
    p0 = wrapper.params.detach().clone()
    wrapper.loss_and_grad((x, y))
    opt = Sven(wrapper, lr=LR, k=8, rtol=1e-12, track_svd_info=True, variable_k=True)
    opt.step((x, y))
    update = (p0 - wrapper.params.detach()) / LR
    info = opt.finalize_svd_info()

    k_used = int(info["k_used"][0])
    assert len(info["variable_k_substep_losses"][0]) == k_used + 1
    # the applied parameter change, i.e. lr * (sum of the accepted rank-1 updates)
    assert info["update_norm"][0] == pytest.approx(LR * float(update.norm()), rel=1e-9)
    assert info["update_norm"][0] == pytest.approx(
        float((p0 - wrapper.params.detach()).norm()), rel=1e-12
    )
    svs = torch.as_tensor(info["svs"][0], dtype=DT)
    if k_used:
        assert info["sv_min_kept"][0] == pytest.approx(float(svs[k_used - 1]), rel=1e-8)
    else:
        assert info["sv_min_kept"][0] == float("inf")


def test_variable_k_rejecting_every_component_logs_a_zero_update():
    """The ``applied is None`` branch: lr so large the first component is
    rejected, so nothing is applied and there is no kept singular value."""
    template, (x, y) = make_mlp(), make_data()
    wrapper = SvenWrapper(copy.deepcopy(template), per_sample_mse, DEVICE)
    p0 = wrapper.params.detach().clone()
    wrapper.loss_and_grad((x, y))
    opt = Sven(wrapper, lr=1e6, k=8, rtol=1e-12, track_svd_info=True, variable_k=True)
    opt.step((x, y))
    info = opt.finalize_svd_info()
    assert int(info["k_used"][0]) == 0
    assert torch.equal(p0, wrapper.params.detach())  # the trial update was reverted
    assert info["update_norm"][0] == 0.0
    assert info["sv_min_kept"][0] == float("inf")


def test_gram_fp64_blocks_match_one_shot():
    """The blocked accumulation is the whole point; it must be the same Gram."""
    jac, _ = explicit_jacobian(make_mlp(), *make_data())
    monkey = Sven._GRAM_BLOCK_ELEMS
    try:
        Sven._GRAM_BLOCK_ELEMS = 64  # force many blocks
        blocked = Sven._gram_fp64(jac)
    finally:
        Sven._GRAM_BLOCK_ELEMS = monkey
    assert rel_err(blocked, jac @ jac.T) < 1e-14


def test_reg_logs_full_spectrum_and_utr():
    """SvenGramReg carries the same record (damping only: rhs = r, no jvp)."""
    template, (x, y) = make_mlp(), make_data()
    jac, resid = explicit_jacobian(template, x, y)
    wrapper = GramSvenWrapper(copy.deepcopy(template), per_sample_mse, DEVICE)
    p0 = wrapper.params.detach().clone()
    wrapper.loss_and_grad((x, y))
    gram = wrapper.gram.detach().clone()
    opt = SvenGramReg(wrapper, lr=LR, k=8, rtol=1e-12, damping=1e-3, track_svd_info=True)
    opt.step()
    update = (p0 - wrapper.params.detach()) / LR
    sigma, _, evecs = Sven._spectrum_from_gram(gram.to(DT))
    info = opt.finalize_svd_info()

    assert_spectrum_matches(
        torch.as_tensor(info["svs"][0], dtype=DT), torch.linalg.svdvals(jac)
    )
    torch.testing.assert_close(
        torch.as_tensor(info["utr"][0], dtype=DT), evecs.T @ resid, rtol=1e-10, atol=1e-14
    )
    assert info["resid_norm"][0] == pytest.approx(float(resid.norm()), rel=1e-12)
    assert info["update_norm"][0] == pytest.approx(LR * float(update.norm()), rel=1e-9)
    # the soft filter only drops the fp64 noise floor, so nothing is dropped
    assert info["sv_min_kept"][0] == pytest.approx(float(sigma[-1]), rel=1e-9)

    opt.log_this_step = False
    wrapper.loss_and_grad((x, y))
    opt.step()
    assert opt.svd_info["step"] == [0]
    assert len(opt.svd_info["num_nonzero_svs"]) == 2


# ----------------------------------------------------------------------
# (3) log_this_step=False: only num_nonzero_svs, identical update
# ----------------------------------------------------------------------


HEAVY_KEYS = ("step", "svs", "utr", "update_norm", "resid_norm", "sv_min_kept",
              "sv_noise_floor")


@pytest.mark.parametrize("cls", [Sven, SvenGram])
def test_log_this_step_false_records_only_rank(cls):
    template, (x, y) = make_mlp(), make_data()
    p_logged, opt_logged = run_steps(cls, template, x, y, log=True)
    p_quiet, opt_quiet = run_steps(cls, template, x, y, log=False)

    assert torch.equal(p_logged, p_quiet)  # bit-identical parameters
    assert opt_quiet.step_count == 3
    assert opt_quiet.svd_info["num_nonzero_svs"] == opt_logged.svd_info["num_nonzero_svs"]
    assert len(opt_quiet.svd_info["num_nonzero_svs"]) == 3
    for key in HEAVY_KEYS:
        assert opt_quiet.svd_info[key] == [], key
        assert len(opt_logged.svd_info[key]) == 3, key
    assert opt_logged.svd_info["step"] == [0, 1, 2]
    # finalize is safe on the empty record and repeatable
    assert opt_quiet.finalize_svd_info()["svs"] == []
    first = opt_logged.finalize_svd_info()
    again = opt_logged.finalize_svd_info()
    assert first["num_nonzero_svs"].tolist() == again["num_nonzero_svs"].tolist()
    assert again["step"].tolist() == [0, 1, 2]


@pytest.mark.parametrize("cls", [Sven, SvenGram])
def test_track_svd_info_false_records_nothing(cls):
    """Legacy default: no logging at all, same update."""
    template, (x, y) = make_mlp(), make_data()
    wrapper_cls = SvenWrapper if cls is Sven else GramSvenWrapper
    wrapper = wrapper_cls(copy.deepcopy(template), per_sample_mse, DEVICE)
    opt = cls(wrapper, lr=LR, k=8, rtol=1e-12)
    for _ in range(2):
        wrapper.loss_and_grad((x, y))
        opt.step((x, y))
    assert all(v == [] for v in opt.svd_info.values())
    assert opt.step_count == 2


# ----------------------------------------------------------------------
# (4) empty_cache=False: no empty_cache() calls, identical update
# ----------------------------------------------------------------------


@contextmanager
def count_empty_cache():
    """Pretend CUDA is present and count ``empty_cache()`` calls."""
    calls: list[int] = []
    with mock.patch.object(torch.cuda, "is_available", lambda: True), \
         mock.patch.object(torch.cuda, "empty_cache", lambda: calls.append(1)):
        yield calls


@pytest.mark.parametrize(
    "cls,per_step", [(Sven, 2), (SvenGram, 1), (SvenGramReg, 1)]
)
def test_empty_cache_flag(cls, per_step):
    """All 4 call sites are guarded, and the flag cannot change the update.

    ``per_step`` counts the guarded sites each class reaches per step: 2 in
    ``Sven.step`` (after the solve, end of step), 1 each in the two Gram
    classes -- 4 sites in total.
    """
    template, (x, y) = make_mlp(), make_data()
    params = {}
    for flag in (True, False):
        wrapper_cls = SvenWrapper if cls is Sven else GramSvenWrapper
        wrapper = wrapper_cls(copy.deepcopy(template), per_sample_mse, DEVICE)
        opt = cls(wrapper, lr=LR, k=8, rtol=1e-12, empty_cache=flag)
        with count_empty_cache() as calls:
            for _ in range(2):
                wrapper.loss_and_grad((x, y))
                opt.step((x, y))
        assert len(calls) == (2 * per_step if flag else 0)
        params[flag] = wrapper.params.detach().clone()
    assert torch.equal(params[True], params[False])


@pytest.mark.parametrize("cls", [Sven, SvenGram, SvenGramReg])
def test_empty_cache_defaults_to_false(cls):
    """CONTRACTS scope update: the per-step empty_cache() cost 4.5x on CIFAR."""
    template, (x, y) = make_mlp(), make_data()
    wrapper_cls = SvenWrapper if cls is Sven else GramSvenWrapper
    wrapper = wrapper_cls(copy.deepcopy(template), per_sample_mse, DEVICE)
    opt = cls(wrapper, lr=LR, k=8, rtol=1e-12)
    assert opt.empty_cache is False
    with count_empty_cache() as calls:
        wrapper.loss_and_grad((x, y))
        opt.step((x, y))
    assert calls == []


# ----------------------------------------------------------------------
# (5) the divergence guard survives the sync removal
# ----------------------------------------------------------------------


def test_zero_gram_still_raises():
    template, (x, y) = make_mlp(), make_data()
    wrapper = GramSvenWrapper(copy.deepcopy(template), per_sample_mse, DEVICE)
    wrapper.loss_and_grad((x, y))
    wrapper.gram = torch.zeros_like(wrapper.gram)
    opt = SvenGram(wrapper, lr=LR, k=8, rtol=1e-3, track_svd_info=True)
    with pytest.raises(RuntimeError, match="run diverged"):
        opt.step()


# ----------------------------------------------------------------------
# (6) a non-logged step costs no more host transfers than correctness needs
# ----------------------------------------------------------------------


HOST_TRANSFER_METHODS = ("item", "tolist", "cpu", "numpy", "__bool__", "__int__",
                         "__float__")


@contextmanager
def count_host_transfers():
    """Record every device->host transfer issued from inside ``sven/opt``.

    Only the IMMEDIATE caller frame is inspected, so the count is exactly the
    transfers written in this track's files; transfers inside the ``sven/nn``
    wrappers belong to another track and are identical on logged and
    non-logged steps.  On CPU these calls are cheap, but each one is a device
    synchronisation on CUDA, which is what the budget is about.
    """
    opt_dir = os.path.dirname(sven_opt_module.__file__)
    sites: list[str] = []
    originals = {name: getattr(torch.Tensor, name) for name in HOST_TRANSFER_METHODS}

    def make_probe(name, orig):
        def probe(self, *args, **kwargs):
            frame = sys._getframe(1)
            if os.path.dirname(frame.f_code.co_filename) == opt_dir:
                sites.append(
                    f"{os.path.basename(frame.f_code.co_filename)}"
                    f":{frame.f_lineno}:{name}"
                )
            return orig(self, *args, **kwargs)
        return probe

    for name, orig in originals.items():
        setattr(torch.Tensor, name, make_probe(name, orig))
    try:
        yield sites
    finally:
        for name, orig in originals.items():
            setattr(torch.Tensor, name, orig)


@pytest.mark.parametrize(
    "cls,budget",
    [(Sven, 2), (SvenGram, 1), (SvenGramReg, 1)],
    ids=["classic", "gram", "gram_reg"],
)
def test_non_logged_step_transfer_budget(cls, budget):
    """The point of ``log_this_step=False``: no diagnostic syncs on the step.

    The budget is what the step needs anyway: the Gram classes' divergence
    guard (which now carries the retained rank in the same transfer) and, for
    the classic path, ``pinv``'s rtol slice length plus the per-step rank.
    """
    template, (x, y) = make_mlp(), make_data()
    sites = {}
    for log in (True, False):
        wrapper_cls = SvenWrapper if cls is Sven else GramSvenWrapper
        wrapper = wrapper_cls(copy.deepcopy(template), per_sample_mse, DEVICE)
        opt = cls(wrapper, lr=LR, k=8, rtol=1e-12, track_svd_info=True)
        opt.log_this_step = log
        wrapper.loss_and_grad((x, y))
        with count_host_transfers() as recorded:
            opt.step((x, y))
        sites[log] = list(recorded)
    assert len(sites[False]) == budget, sites[False]
    assert len(sites[False]) < len(sites[True]), sites


# ----------------------------------------------------------------------
# (7) the record is host-side: what generic_scan._split_diagnostics needs
# ----------------------------------------------------------------------


class _FakeDeviceTensor(torch.Tensor):
    """A CPU tensor that reports a CUDA device.

    Lets a CPU-only suite exercise the off-CPU branch of any code that
    switches on ``tensor.device``; a 0-d tensor stored in ``svd_info`` on GPU
    makes the runner's ``np.asarray(..., dtype=np.int32)`` raise
    ``TypeError: can't convert cuda:0 device type tensor to numpy``.
    """

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", 0)


def assert_host_side(info: dict) -> None:
    """No torch tensors, and every ``np.asarray`` the runner makes succeeds."""
    for key, values in info.items():
        if key == "variable_k_substep_losses":
            continue  # ragged nested lists of tensors; the runner maps float() over it
        for v in values:
            assert not isinstance(v, torch.Tensor), (key, type(v))
    # generic_scan._split_diagnostics
    np.asarray(info["num_nonzero_svs"], dtype=np.int32)
    np.asarray(info["step"], dtype=np.int32)
    for key in ("update_norm", "resid_norm", "sv_min_kept", "sv_noise_floor"):
        np.asarray(info[key], dtype=np.float32)
    np.array([np.max(s) for s in info["svs"]], dtype=np.float32)


@pytest.mark.parametrize("cls", [Sven, SvenGram, SvenGramReg])
@pytest.mark.parametrize("log", [True, False])
def test_svd_info_is_host_side(cls, log):
    _, opt = run_steps(cls, make_mlp(), *make_data(), log=log)
    assert_host_side(opt.svd_info)
    assert all(type(v) is int for v in opt.svd_info["num_nonzero_svs"])
    assert_host_side(opt.finalize_svd_info())


def test_record_rank_is_host_side_off_cpu():
    """A 0-d CUDA tensor in ``num_nonzero_svs`` stops the whole svd campaign."""
    wrapper = SvenWrapper(make_mlp(), per_sample_mse, DEVICE)
    opt = Sven(wrapper, lr=LR, k=8, rtol=1e-12, track_svd_info=True)
    kept = torch.tensor([2.0, 1.0, 0.0]).as_subclass(_FakeDeviceTensor)
    assert kept.device.type == "cuda"  # the branch under test is reachable
    opt._record_rank(kept)
    assert opt.svd_info["num_nonzero_svs"] == [2]
    assert type(opt.svd_info["num_nonzero_svs"][0]) is int
    assert np.asarray(opt.svd_info["num_nonzero_svs"], dtype=np.int32).tolist() == [2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("cls", [Sven, SvenGram])
def test_svd_info_is_host_side_on_cuda(cls):
    template, (x, y) = make_mlp(), make_data()
    wrapper_cls = SvenWrapper if cls is Sven else GramSvenWrapper
    wrapper = wrapper_cls(copy.deepcopy(template).cuda(), per_sample_mse, "cuda")
    x, y = x.cuda(), y.cuda()
    opt = cls(wrapper, lr=LR, k=8, rtol=1e-12, track_svd_info=True)
    for _ in range(2):
        wrapper.loss_and_grad((x, y))
        opt.step((x, y))
    assert_host_side(opt.svd_info)
    assert np.asarray(opt.svd_info["num_nonzero_svs"], dtype=np.int32).shape == (2,)


# ----------------------------------------------------------------------
# (8) float32: the Gram route's usable range stops at sv_noise_floor (F17/F19)
# ----------------------------------------------------------------------


def test_float32_spectrum_trustworthy_only_above_the_noise_floor():
    """Both paths take sigma from an eigh of ``J J^T``, which squares the
    condition number.  In float64 (every other test here) that is invisible;
    with the float32 parameters the campaign actually uses, sigma below
    ``sqrt(eps32) * sigma_max`` (~3e-4 sigma_max) is noise -- three decades of
    usable range, not the ~1e-7 of an ``svdvals(J)``.  The recorded
    ``sv_noise_floor`` is where analysis must truncate.
    """
    template = make_mlp(dtype=torch.float32)
    x, y = make_data(dup_scales=(1e-3, 1e-5, 1e-7), dtype=torch.float32)
    jac, _ = explicit_jacobian(template, x, y)
    assert jac.dtype is torch.float32
    reference = torch.linalg.svdvals(jac.double())

    _, _, opt = run_gram_step(template, x, y, k=8, rtol=1e-12)
    info = opt.finalize_svd_info()
    svs = torch.as_tensor(info["svs"][0], dtype=DT)
    floor = float(info["sv_noise_floor"][0])
    assert floor == pytest.approx(
        float(np.sqrt(np.finfo(np.float32).eps)) * float(svs[0]), rel=1e-6
    )

    above = svs > floor
    assert bool(above.any()) and not bool(above.all())  # the floor cuts this spectrum
    rel = (svs - reference).abs() / reference
    assert float(rel[above].max()) < 1e-6  # usable above the floor
    assert float(rel[-1]) > 1e-3  # sigma_M is noise, NOT accurate to 1e-7
