"""C-R4: the achieved parameter fraction is recorded for every mask mode.

The mask is redrawn every step, so ``actual_param_fraction`` alone cannot
answer "which fraction did this run actually use"; both wrappers therefore
expose ``mean_actual_param_fraction``, the running mean over the steps taken
(plain Python floats, no device sync).  Also pins F31: the classic wrapper's
**elementwise** path never set ``actual_param_fraction`` at all and reported
1.0 forever.  CPU / float64.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from sven.nn import GramSvenWrapper, SvenWrapper

DT = torch.float64
DEVICE = "cpu"
B = 8
FRACTION = 0.3
STEPS = 4
MODES = ["elementwise", "tensor", "rows"]


def per_sample_mse(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((pred - y) ** 2).mean(dim=1)


def make_mlp(seed: int = 0) -> nn.Sequential:
    """Eight parameter tensors, so whole-tensor packing can approach a target."""
    torch.manual_seed(seed)
    dims = [8, 16, 16, 16, 4]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers).to(DT)


def make_data(seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(B, 8, generator=g, dtype=DT),
        torch.randn(B, 4, generator=g, dtype=DT),
    )


def make_wrapper(kind: str, template: nn.Module, mode: str):
    if kind == "classic":
        return SvenWrapper(
            copy.deepcopy(template),
            per_sample_mse,
            DEVICE,
            param_fraction=FRACTION,
            mask_mode=mode,
        )
    return GramSvenWrapper(
        copy.deepcopy(template),
        per_sample_mse,
        DEVICE,
        capture="chunked",
        chunk_numel=128,
        param_fraction=FRACTION,
        mask_mode=mode,
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ["classic", "gram"])
def test_mean_actual_param_fraction_tracks_the_request(kind, mode):
    template, (x, y) = make_mlp(), make_data()
    wrapper = make_wrapper(kind, template, mode)
    # no masked step yet: report the REQUEST, never 1.0 (which would read as
    # "unmasked"), and never NaN (this value goes into the jsonl records)
    assert wrapper.mean_actual_param_fraction == FRACTION

    per_step: list[float] = []
    for step in range(STEPS):
        torch.manual_seed(100 + step)
        wrapper.loss_and_grad((x, y))
        achieved = wrapper.actual_param_fraction
        # whatever the mode, the reported fraction is the mask it actually used
        assert achieved == wrapper.param_mask.sum().item() / wrapper.n_params
        per_step.append(achieved)

    mean = wrapper.mean_actual_param_fraction
    assert type(mean) is float
    assert mean == pytest.approx(sum(per_step) / len(per_step), rel=0, abs=1e-15)
    # close to the requested fraction (whole tensors / whole rows can only
    # approximate it, hence the same band the other mask tests use)
    assert 0.5 * FRACTION <= mean <= 1.5 * FRACTION, (mode, per_step)


@pytest.mark.parametrize("kind", ["classic", "gram"])
def test_elementwise_fraction_is_recorded(kind):
    """F31: the classic elementwise sampler never set the attribute."""
    template, (x, y) = make_mlp(), make_data()
    wrapper = make_wrapper(kind, template, "elementwise")
    exact = int(FRACTION * wrapper.n_params) / wrapper.n_params

    assert wrapper.actual_param_fraction == 1.0  # unset before the first step
    wrapper.loss_and_grad((x, y))

    assert wrapper.actual_param_fraction == exact
    assert wrapper.actual_param_fraction != 1.0
    assert wrapper.mean_actual_param_fraction == exact


@pytest.mark.parametrize("kind", ["classic", "gram"])
def test_unmasked_run_reports_one(kind):
    template, (x, y) = make_mlp(), make_data()
    if kind == "classic":
        wrapper = SvenWrapper(copy.deepcopy(template), per_sample_mse, DEVICE)
    else:
        wrapper = GramSvenWrapper(
            copy.deepcopy(template), per_sample_mse, DEVICE, capture="chunked"
        )
    wrapper.loss_and_grad((x, y))
    assert wrapper.mean_actual_param_fraction == 1.0
    assert wrapper._apf_steps == 0  # nothing accumulated when nothing is masked


def test_mean_differs_from_the_last_step_when_the_mask_varies():
    """The mean is a genuine average, not the latest value."""
    template, (x, y) = make_mlp(), make_data()
    wrapper = make_wrapper("classic", template, "tensor")
    seen: list[float] = []
    for step in range(6):
        torch.manual_seed(7 * step + 1)
        wrapper.loss_and_grad((x, y))
        seen.append(wrapper.actual_param_fraction)
    assert len(set(seen)) > 1  # the redrawn tensor selection really varies
    assert wrapper.mean_actual_param_fraction == pytest.approx(sum(seen) / len(seen))
