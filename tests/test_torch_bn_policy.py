"""C-E2: the normalisation-statistics policy of the torch Sven wrappers.

``bn_mode="batch"`` must train with **batch** statistics and advance the
running statistics from the training batch **exactly once per optimizer
step**; ``bn_mode="frozen"`` must never write a norm buffer, in train or in
eval; :meth:`evaluate` must be eval-mode and side-effect-free under both; and
the Gram matrix / update must be bit-for-bit the same quantity as before the
change (batch-statistic normalisation, verified against an explicit
buffer-free reference).  CPU / float64 throughout, and the "exactly once"
references are produced by ``F.batch_norm`` itself, so those comparisons are
bitwise.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.func import functional_call
from torch.nn.modules.batchnorm import _NormBase

from sven.nn import GramSvenWrapper, SvenWrapper
from sven.opt import Sven, SvenGram, SvenGramReg

DT = torch.float64
DEVICE = "cpu"
B = 8
MOMENTUM = 0.1
EPS = 1e-5
CHUNK = 64          # several jacrev groups for the ~165-parameter model below
GRAM_ATOL = 1e-10
DELTA_RTOL = 1e-9


def per_sample_mse(pred: Tensor, y: Tensor) -> Tensor:
    return ((pred - y) ** 2).mean(dim=1)


def rel_err(a: Tensor, b: Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


# ----------------------------------------------------------------------
# Models: a torch.func-compatible BatchNorm, and a buffer-free reference
# ----------------------------------------------------------------------


class FuncBatchNorm2d(_NormBase):
    """``torch.func``-compatible BatchNorm2d.

    A copy of the class sv3 vendors in ``experiments/nn/batchnorm.py`` (kept
    here so the sven tests stay independent of the sv3 tree).  The only
    difference from ``nn.BatchNorm2d`` is that ``num_batches_tracked`` is
    incremented **out of place**, which ``jacrev`` tolerates.
    """

    def __init__(self, num_features: int) -> None:
        super().__init__(num_features, EPS, MOMENTUM, True, True)

    def forward(self, input: Tensor) -> Tensor:
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")
        eaf = 0.0 if self.momentum is None else self.momentum
        if self.training and self.track_running_stats:
            self.num_batches_tracked = self.num_batches_tracked + 1
            if self.momentum is None:
                eaf = 1.0 / float(self.num_batches_tracked)
        if self.training:
            bn_training = True
        else:
            bn_training = self.running_mean is None and self.running_var is None
        track = not self.training or self.track_running_stats
        return F.batch_norm(
            input,
            self.running_mean if track else None,
            self.running_var if track else None,
            self.weight,
            self.bias,
            bn_training,
            eaf,
            self.eps,
        )


class PlainBatchStatNorm2d(nn.Module):
    """Batch-statistic normalisation with **no buffers at all**.

    The reference normalisation for the capture: the same ``F.batch_norm``
    kernel with ``None`` running statistics, so it provably cannot read or
    write a buffer while being bitwise identical to a train-mode BatchNorm.
    Declares ``weight`` before ``bias``, so the flat-parameter layout matches
    :class:`FuncBatchNorm2d`'s.
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features, dtype=DT))
        self.bias = nn.Parameter(torch.zeros(num_features, dtype=DT))

    def forward(self, input: Tensor) -> Tensor:
        return F.batch_norm(input, None, None, self.weight, self.bias, True, 0.0, EPS)


class BNCNN(nn.Module):
    """Two norm layers, the first one directly on the input."""

    def __init__(self, bn_cls=FuncBatchNorm2d) -> None:
        super().__init__()
        self.bn1 = bn_cls(2)
        self.conv = nn.Conv2d(2, 3, 3, padding=1)
        self.bn2 = bn_cls(3)
        self.fc = nn.Linear(3 * 4 * 4, 2)

    def forward(self, x: Tensor) -> Tensor:
        h = self.bn1(x)
        h = torch.tanh(self.conv(h))
        h = torch.tanh(self.bn2(h))
        return self.fc(h.flatten(1))


def make_model_and_data(bn_cls=FuncBatchNorm2d, seed: int = 3):
    """Train-mode model with non-trivial running statistics, plus a batch."""
    torch.manual_seed(seed)
    model = BNCNN(bn_cls).to(DT)
    model.train()
    with torch.no_grad():
        for _ in range(3):  # buffers away from their (0, 1) init
            model(torch.randn(B, 2, 4, 4, dtype=DT))
    x = torch.randn(B, 2, 4, 4, dtype=DT)
    y = torch.randn(B, 2, dtype=DT)
    return model, x, y


# ----------------------------------------------------------------------
# Helpers: buffers after exactly n updates, and the reference Jacobian
# ----------------------------------------------------------------------


def _bn_inputs(model: nn.Module, x: Tensor) -> dict[str, Tensor]:
    """Each norm layer's input during ONE train-mode forward of ``x``.

    Run on a copy: train-mode normalisation uses batch statistics only, so
    these inputs depend on the parameters and ``x`` alone — the same for the
    first update and for a (wrong) second one.
    """
    probe = copy.deepcopy(model)
    probe.train()
    got: dict[str, Tensor] = {}
    handles = [
        mod.register_forward_pre_hook(
            lambda m, inp, name=name: got.__setitem__(name, inp[0].detach().clone())
        )
        for name, mod in probe.named_modules()
        if isinstance(mod, _NormBase)
    ]
    with torch.no_grad():
        probe(x)
    for handle in handles:
        handle.remove()
    return got


def expected_stats(
    model: nn.Module, x: Tensor, n_updates: int
) -> dict[str, tuple[Tensor, Tensor]]:
    """``(running_mean, running_var)`` per norm layer after exactly ``n_updates``.

    Applies ``F.batch_norm`` itself to the pre-step buffers, so the result is
    the bitwise value the module would hold — no hand-rolled EMA formula.
    """
    inputs = _bn_inputs(model, x)
    out: dict[str, tuple[Tensor, Tensor]] = {}
    with torch.no_grad():
        for name, mod in model.named_modules():
            if not isinstance(mod, _NormBase):
                continue
            mean, var = mod.running_mean.clone(), mod.running_var.clone()
            for _ in range(n_updates):
                F.batch_norm(
                    inputs[name], mean, var, mod.weight, mod.bias, True, mod.momentum, mod.eps
                )
            out[name] = (mean, var)
    return out


def assert_updated_exactly_once(
    model: nn.Module,
    once: dict[str, tuple[Tensor, Tensor]],
    twice: dict[str, tuple[Tensor, Tensor]],
) -> None:
    for name, mod in model.named_modules():
        if not isinstance(mod, _NormBase):
            continue
        assert torch.equal(mod.running_mean, once[name][0]), f"{name}.running_mean"
        assert torch.equal(mod.running_var, once[name][1]), f"{name}.running_var"
        # the test must be able to tell one update from two
        assert (once[name][0] - twice[name][0]).abs().max() > 1e-6, name


def count_norm_forwards(model: nn.Module) -> tuple[dict[str, int], list]:
    """Count norm-layer forwards, split by whether the writes are suppressed.

    ``tracking`` counts the forwards that a norm layer would write a buffer in
    (train mode, statistics tracked) — exactly one per layer per optimizer
    step under ``bn_mode="batch"``; ``suppressed`` counts the capture /
    ``jacrev`` group / ``jvp`` / ``delta_from_w`` passes, which lets a test see
    that a pass it means to cover really happened.
    """
    counts = {"tracking": 0, "suppressed": 0}

    def hook(mod: nn.Module, inputs: tuple[Tensor, ...]) -> None:
        tracked = mod.training and mod.track_running_stats
        counts["tracking" if tracked else "suppressed"] += 1

    handles = [
        mod.register_forward_pre_hook(hook)
        for mod in model.modules()
        if isinstance(mod, _NormBase)
    ]
    return counts, handles


def reference_jacobian(model: nn.Module, x: Tensor, y: Tensor) -> Tensor:
    """``(B, P)`` Jacobian of the rows (kappa=2 => rows = per-sample loss).

    Computed on the buffer-free :class:`PlainBatchStatNorm2d` model, i.e. with
    the batch-statistic normalisation written out explicitly.
    """
    entries = [(name, p.shape, p.numel()) for name, p in model.named_parameters()]
    flat = torch.cat([p.detach().reshape(-1) for _, p in model.named_parameters()])

    def f(flat_: Tensor) -> Tensor:
        params: dict[str, Tensor] = {}
        start = 0
        for name, shape, n in entries:
            params[name] = flat_[start : start + n].view(shape)
            start += n
        return per_sample_mse(functional_call(model, params, x), y)

    return torch.func.jacrev(f)(flat)


def batch_stat_reference(model: nn.Module) -> nn.Module:
    """A parameter-identical copy of ``model`` with buffer-free norm layers."""
    ref = BNCNN(PlainBatchStatNorm2d).to(DT)
    src = dict(model.named_parameters())
    with torch.no_grad():
        for name, p in ref.named_parameters():
            p.copy_(src[name])
    return ref


def buffer_snapshot(model: nn.Module) -> dict[str, Tensor]:
    return {name: buf.clone() for name, buf in model.named_buffers()}


def assert_buffers_unchanged(model: nn.Module, before: dict[str, Tensor]) -> None:
    for name, buf in model.named_buffers():
        assert torch.equal(buf, before[name]), name


# ----------------------------------------------------------------------
# (1) exactly one running-stat update per optimizer step
# ----------------------------------------------------------------------


@pytest.mark.parametrize("opt_cls", [SvenGram, SvenGramReg])
@pytest.mark.parametrize("capture", ["full", "chunked"])
def test_batch_mode_updates_stats_exactly_once_gram(capture, opt_cls):
    """``SvenGramReg(fisher_decay > 0)`` adds the ``jvp`` pass (the only
    ``torch.func`` transform outside the capture), so it must be suppressed
    too — otherwise the buffers land on the two-update value."""
    model, x, y = make_model_and_data()
    wrapper = GramSvenWrapper(
        model, per_sample_mse, DEVICE, capture=capture, bn_mode="batch", chunk_numel=CHUNK
    )
    n_norm = sum(isinstance(mod, _NormBase) for mod in wrapper.model.modules())
    if capture == "chunked":
        assert len(wrapper._param_groups()) >= 3  # several groups, each a jacrev pass
    once = expected_stats(wrapper.model, x, 1)
    twice = expected_stats(wrapper.model, x, 2)
    # fisher_decay > 0 makes the Woodbury rhs coefficient non-zero, which is
    # what puts _rows_jvp() on the step's path
    kwargs = {"fisher_decay": 1e-3} if opt_cls is SvenGramReg else {}
    counts, handles = count_norm_forwards(wrapper.model)

    wrapper.loss_and_grad((x, y))
    opt_cls(wrapper, lr=0.1, k=B, rtol=1e-10, **kwargs).step()  # includes delta_from_w
    for handle in handles:
        handle.remove()

    assert_updated_exactly_once(wrapper.model, once, twice)
    # one writing forward per layer, and the passes the step is meant to make:
    # capture (>= 1 group) + delta_from_w, plus jvp for the regularized step
    assert counts["tracking"] == n_norm
    assert counts["suppressed"] >= (3 if opt_cls is SvenGramReg else 2) * n_norm


def test_batch_mode_updates_stats_exactly_once_classic():
    """The classic SvenWrapper + Sven (Jacobian) path."""
    model, x, y = make_model_and_data()
    wrapper = SvenWrapper(model, per_sample_mse, DEVICE)
    assert wrapper.bn_mode == "batch"  # the Jacobian path's default
    once = expected_stats(wrapper.model, x, 1)
    twice = expected_stats(wrapper.model, x, 2)

    wrapper.loss_and_grad((x, y))
    Sven(wrapper, lr=0.1, k=B, rtol=1e-10).step()

    assert_updated_exactly_once(wrapper.model, once, twice)


def test_batch_mode_skipped_without_running_stats():
    """No norm layer with running statistics => no extra forward at all."""
    torch.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Linear(6, 2)).to(DT)
    x, y = torch.randn(B, 4, dtype=DT), torch.randn(B, 2, dtype=DT)
    wrapper = SvenWrapper(mlp, per_sample_mse, DEVICE, bn_mode="batch")
    assert wrapper.update_norm_running_stats((x, y)) is False
    wrapper.loss_and_grad((x, y))  # must not raise
    assert wrapper._norm_stat_modules() == []


# ----------------------------------------------------------------------
# (2) evaluate() is eval-mode and side-effect-free
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bn_mode", ["batch", "frozen"])
@pytest.mark.parametrize("kind", ["classic", "gram"])
def test_evaluate_is_eval_mode_and_side_effect_free(kind, bn_mode):
    model, x, y = make_model_and_data()
    if kind == "classic":
        wrapper = SvenWrapper(model, per_sample_mse, DEVICE, bn_mode=bn_mode)
    else:
        wrapper = GramSvenWrapper(
            model, per_sample_mse, DEVICE, capture="full", bn_mode=bn_mode
        )
    before = buffer_snapshot(wrapper.model)
    modes = {name: mod.training for name, mod in wrapper.model.named_modules()}

    pred = wrapper.evaluate(x)

    assert_buffers_unchanged(wrapper.model, before)
    assert {name: mod.training for name, mod in wrapper.model.named_modules()} == modes
    assert all(modes.values())  # the model really was left in train mode

    # eval-mode normalisation: identical to the module's own eval forward
    ref = copy.deepcopy(wrapper.model).eval()
    with torch.no_grad():
        assert torch.equal(pred, ref(x))

    # a fixed example's prediction does not depend on its batch companions
    x_other = x.clone()
    x_other[1:] = torch.randn_like(x_other[1:])
    assert torch.equal(wrapper.evaluate(x_other)[0], pred[0])
    assert torch.allclose(wrapper.evaluate(x[:1])[0], pred[0], atol=1e-12)


@pytest.mark.parametrize("bn_mode", ["batch", "frozen"])
def test_evaluate_and_loss_follows_the_capture_and_writes_nothing(bn_mode):
    """The variable_k line search must not switch normalisation mid-step."""
    model, x, y = make_model_and_data()
    wrapper = SvenWrapper(model, per_sample_mse, DEVICE, bn_mode=bn_mode)
    before = buffer_snapshot(wrapper.model)

    loss = wrapper.evaluate_and_loss(x, y)
    assert_buffers_unchanged(wrapper.model, before)

    ref = copy.deepcopy(wrapper.model)
    ref.train() if bn_mode == "batch" else ref.eval()
    with torch.no_grad():
        assert torch.equal(loss, per_sample_mse(ref(x), y))

    eval_loss = per_sample_mse(wrapper.evaluate(x), y)
    if bn_mode == "batch":  # batch vs running statistics really do differ
        assert not torch.allclose(loss, eval_loss)
    else:
        assert torch.equal(loss, eval_loss)


# ----------------------------------------------------------------------
# (3) frozen mode never writes a buffer, and restores modes per module
# ----------------------------------------------------------------------


@pytest.mark.parametrize("capture", ["hooks", "chunked", "full"])
def test_frozen_mode_writes_no_buffer(capture):
    model, x, y = make_model_and_data()
    wrapper = GramSvenWrapper(
        model, per_sample_mse, DEVICE, capture=capture, bn_mode="frozen", chunk_numel=CHUNK
    )
    before = buffer_snapshot(wrapper.model)

    wrapper.loss_and_grad((x, y))
    SvenGram(wrapper, lr=0.1, k=B, rtol=1e-10).step()
    wrapper.evaluate(x)
    wrapper.evaluate_and_loss(x, y)

    assert_buffers_unchanged(wrapper.model, before)
    assert all(mod.training for mod in wrapper.model.modules())  # restored


def test_frozen_restores_each_modules_own_mode():
    """Per-module restore, not a blanket .train(): an eval-mode norm layer
    the caller froze deliberately must still be in eval afterwards."""
    model, x, y = make_model_and_data()
    model.bn2.eval()
    wrapper = GramSvenWrapper(
        model, per_sample_mse, DEVICE, capture="chunked", bn_mode="frozen", chunk_numel=CHUNK
    )
    wrapper.loss_and_grad((x, y))
    SvenGram(wrapper, lr=0.1, k=B, rtol=1e-10).step()
    assert wrapper.model.bn1.training
    assert not wrapper.model.bn2.training


def test_no_norm_stat_updates_restores_flags_and_modes():
    model, x, y = make_model_and_data()
    wrapper = SvenWrapper(model, per_sample_mse, DEVICE, bn_mode="batch")
    model.bn2.eval()
    model.bn2.track_running_stats = False
    before = buffer_snapshot(wrapper.model)

    with wrapper.no_norm_stat_updates():
        assert not model.bn1.track_running_stats
        assert not model.bn2.track_running_stats
        with torch.no_grad():
            model(x)

    assert_buffers_unchanged(wrapper.model, before)
    assert model.bn1.track_running_stats and model.bn1.training
    assert not model.bn2.track_running_stats and not model.bn2.training


# ----------------------------------------------------------------------
# (4) the capture normalisation (hence the Gram and the update) is unchanged
# ----------------------------------------------------------------------


@pytest.mark.parametrize("capture", ["full", "chunked"])
def test_batch_mode_gram_and_update_match_explicit_batch_stats(capture):
    model, x, y = make_model_and_data()
    wrapper = GramSvenWrapper(
        model, per_sample_mse, DEVICE, capture=capture, bn_mode="batch", chunk_numel=CHUNK
    )
    j_ref = reference_jacobian(batch_stat_reference(wrapper.model), x, y)

    wrapper.loss_and_grad((x, y))

    assert (wrapper.gram - j_ref @ j_ref.T).abs().max() < GRAM_ATOL
    w = torch.linspace(-1.0, 1.0, B, dtype=DT)
    assert rel_err(wrapper.delta_from_w(w), j_ref.T @ w) < DELTA_RTOL


def test_batch_mode_classic_jacobian_matches_explicit_batch_stats():
    model, x, y = make_model_and_data()
    wrapper = SvenWrapper(model, per_sample_mse, DEVICE, bn_mode="batch")
    j_ref = reference_jacobian(batch_stat_reference(wrapper.model), x, y)

    wrapper.loss_and_grad((x, y))

    assert rel_err(wrapper.grads, j_ref) < DELTA_RTOL


# ----------------------------------------------------------------------
# (5) guards: stock BatchNorm on the batch-stat path, bn_mode resolution
# ----------------------------------------------------------------------


def test_stock_batchnorm_rejected_on_batch_stat_path():
    """A train-mode stock nn.BatchNorm2d would mutate num_batches_tracked in
    place inside a torch.func transform; say so instead of crashing there."""
    model, x, y = make_model_and_data(bn_cls=nn.BatchNorm2d)
    wrapper = GramSvenWrapper(
        model, per_sample_mse, DEVICE, capture="full", bn_mode="batch"
    )
    with pytest.raises(NotImplementedError, match="torch.func-compatible BatchNorm"):
        wrapper.loss_and_grad((x, y))

    classic = SvenWrapper(copy.deepcopy(model), per_sample_mse, DEVICE, bn_mode="batch")
    with pytest.raises(NotImplementedError, match="torch.func-compatible BatchNorm"):
        classic.loss_and_grad((x, y))

    # frozen mode keeps working: eval-mode BatchNorm never hits the in-place op
    frozen = GramSvenWrapper(
        copy.deepcopy(model), per_sample_mse, DEVICE, capture="full", bn_mode="frozen"
    )
    frozen.loss_and_grad((x, y))
    assert frozen.gram is not None

    # ... and so does an already-frozen stock module on the batch path, which
    # is how the reference pipelines in the other test files use it
    eval_model = copy.deepcopy(model)
    eval_model.bn1.eval()
    eval_model.bn2.eval()
    eval_wrapper = SvenWrapper(eval_model, per_sample_mse, DEVICE, bn_mode="batch")
    eval_wrapper.loss_and_grad((x, y))
    assert eval_wrapper.grads.shape[0] == B


def test_hooks_capture_rejects_batch_mode_norm_layers():
    model, x, y = make_model_and_data()
    wrapper = GramSvenWrapper(
        model, per_sample_mse, DEVICE, capture="hooks", bn_mode="batch"
    )
    with pytest.raises(ValueError, match="bn_mode='frozen' only"):
        wrapper.loss_and_grad((x, y))

    # ... and also when the norm layers are already in eval mode: such a run
    # would train with FROZEN statistics while labelling itself "batch"
    eval_model = copy.deepcopy(model)
    eval_model.bn1.eval()
    eval_model.bn2.eval()
    eval_wrapper = GramSvenWrapper(
        eval_model, per_sample_mse, DEVICE, capture="hooks", bn_mode="batch"
    )
    with pytest.raises(ValueError, match="bn_mode='frozen' only"):
        eval_wrapper.loss_and_grad((x, y))


def test_instancenorm_rejected_on_batch_stat_path():
    """``track_running_stats=False`` does not suppress an InstanceNorm write
    (and changes its normalisation), so the batch-stat path must refuse it."""
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.InstanceNorm1d(3, affine=True, track_running_stats=True, dtype=DT),
        nn.Flatten(),
        nn.Linear(3 * 5, 2, dtype=DT),
    )
    net.train()
    x, y = torch.randn(B, 3, 5, dtype=DT), torch.randn(B, 2, dtype=DT)

    wrapper = SvenWrapper(net, per_sample_mse, DEVICE, bn_mode="batch")
    with pytest.raises(NotImplementedError, match="regardless of track_running_stats"):
        wrapper.loss_and_grad((x, y))

    # frozen mode is unaffected: the layer runs in eval mode and writes nothing
    frozen = SvenWrapper(copy.deepcopy(net), per_sample_mse, DEVICE, bn_mode="frozen")
    before = buffer_snapshot(frozen.model)
    frozen.loss_and_grad((x, y))
    assert_buffers_unchanged(frozen.model, before)


def test_bn_mode_resolution_and_alias():
    model, _, _ = make_model_and_data()
    assert GramSvenWrapper(copy.deepcopy(model), per_sample_mse, DEVICE).bn_mode == "frozen"
    assert SvenWrapper(copy.deepcopy(model), per_sample_mse, DEVICE).bn_mode == "batch"

    legacy = GramSvenWrapper(
        copy.deepcopy(model), per_sample_mse, DEVICE, capture="full", freeze_norm_stats=False
    )
    assert legacy.bn_mode == "batch" and legacy.freeze_norm_stats is False
    frozen = GramSvenWrapper(
        copy.deepcopy(model), per_sample_mse, DEVICE, freeze_norm_stats=True
    )
    assert frozen.bn_mode == "frozen" and frozen.freeze_norm_stats is True

    with pytest.raises(ValueError, match="bn_mode must be"):
        SvenWrapper(copy.deepcopy(model), per_sample_mse, DEVICE, bn_mode="eval")
    with pytest.raises(ValueError, match="conflicts"):
        SvenWrapper(
            copy.deepcopy(model),
            per_sample_mse,
            DEVICE,
            bn_mode="batch",
            freeze_norm_stats=True,
        )
