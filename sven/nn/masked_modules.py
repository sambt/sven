"""Mask-aware twins of ``nn.Linear`` / ``nn.Conv2d`` for row-block masking.

Each twin shares the ORIGINAL module's ``Parameter`` objects, so swapping a
module for its twin changes neither parameter names nor storage — the
``SvenWrapper`` flat-vector tie applies unchanged.  With no selection set, a
twin's ``forward`` is identical to its parent class.

During a rows-mode Jacobian pass the wrapper injects traced row blocks
(``_active_weight`` / ``_active_bias``) and the forward splits: active output
rows come from the injected (differentiated) blocks, frozen rows from the
module's own detached parameters.  The frozen matmul/conv never forms a weight
gradient (``needs_input_grad`` pruning at the op boundary), so the
reverse-mode cotangent stays at the active-row count instead of the full
tensor size.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RowMaskedLinear(nn.Linear):
    """``nn.Linear`` twin computing selected output rows from injected blocks.

    Selection state (``_active_rows`` / ``_frozen_rows``: sorted index
    tensors) is set per step by the wrapper; ``_active_weight`` /
    ``_active_bias`` are set inside the ``jacrev``-traced function so they
    carry the traced tensors.  All default to ``None``: a twin with no
    selection forwards exactly like ``nn.Linear``.
    """

    _active_rows: torch.Tensor | None = None
    _frozen_rows: torch.Tensor | None = None
    _active_weight: torch.Tensor | None = None
    _active_bias: torch.Tensor | None = None

    @classmethod
    def from_module(cls, orig: nn.Linear) -> "RowMaskedLinear":
        """Build a twin sharing ``orig``'s ``Parameter`` objects (no copies).

        Args:
            orig: The ``nn.Linear`` to twin.

        Returns:
            A :class:`RowMaskedLinear` with the same parameters and config.
        """
        twin = cls(
            orig.in_features,
            orig.out_features,
            bias=orig.bias is not None,
            device=orig.weight.device,
            dtype=orig.weight.dtype,
        )
        twin.weight = orig.weight
        twin.bias = orig.bias
        return twin

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._active_weight is None:
            return F.linear(input, self.weight, self.bias)
        y_act = F.linear(input, self._active_weight, self._active_bias)
        out = y_act.new_zeros(*input.shape[:-1], self.out_features)
        out = out.index_copy(-1, self._active_rows, y_act)  # out-of-place: functorch-safe
        if self._frozen_rows.numel():
            w_frz = self.weight.detach().index_select(0, self._frozen_rows)
            b_frz = (
                self.bias.detach().index_select(0, self._frozen_rows)
                if self.bias is not None
                else None
            )
            out = out.index_copy(-1, self._frozen_rows, F.linear(input, w_frz, b_frz))
        return out


class RowMaskedConv2d(nn.Conv2d):
    """``nn.Conv2d`` twin computing selected output channels from injected blocks.

    Same selection/injection contract as :class:`RowMaskedLinear`; "rows" are
    output channels (whole filters).  ``groups != 1`` is unsupported: channel
    selection cannot split grouped filters.
    """

    _active_rows: torch.Tensor | None = None
    _frozen_rows: torch.Tensor | None = None
    _active_weight: torch.Tensor | None = None
    _active_bias: torch.Tensor | None = None

    @classmethod
    def from_module(cls, orig: nn.Conv2d) -> "RowMaskedConv2d":
        """Build a twin sharing ``orig``'s ``Parameter`` objects (no copies).

        Args:
            orig: The ``nn.Conv2d`` to twin; must have ``groups == 1``.

        Returns:
            A :class:`RowMaskedConv2d` with the same parameters and config.

        Raises:
            NotImplementedError: If ``orig.groups != 1``.
        """
        if orig.groups != 1:
            raise NotImplementedError(
                "RowMaskedConv2d requires groups=1: output-channel selection "
                f"cannot split grouped filters (got groups={orig.groups})"
            )
        twin = cls(
            orig.in_channels,
            orig.out_channels,
            orig.kernel_size,
            stride=orig.stride,
            padding=orig.padding,
            dilation=orig.dilation,
            groups=1,
            bias=orig.bias is not None,
            padding_mode=orig.padding_mode,
            device=orig.weight.device,
            dtype=orig.weight.dtype,
        )
        twin.weight = orig.weight
        twin.bias = orig.bias
        return twin

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._active_weight is None:
            return self._conv_forward(input, self.weight, self.bias)
        # _conv_forward applies stride/padding/dilation/padding_mode to both halves
        y_act = self._conv_forward(input, self._active_weight, self._active_bias)
        ch_dim = y_act.dim() - 3  # 1 for batched input, 0 for unbatched
        shape = list(y_act.shape)
        shape[ch_dim] = self.out_channels
        out = y_act.new_zeros(shape)
        out = out.index_copy(ch_dim, self._active_rows, y_act)
        if self._frozen_rows.numel():
            w_frz = self.weight.detach().index_select(0, self._frozen_rows)
            b_frz = (
                self.bias.detach().index_select(0, self._frozen_rows)
                if self.bias is not None
                else None
            )
            out = out.index_copy(
                ch_dim, self._frozen_rows, self._conv_forward(input, w_frz, b_frz)
            )
        return out


def replace_with_row_masked(model: nn.Module) -> nn.Module:
    """Swap every ``nn.Linear`` / ``nn.Conv2d`` (exact types) for its twin, in place.

    Twins share the original ``Parameter`` objects, so parameter names, values
    and storage are unchanged.  Subclasses are left alone — callers must
    reject them beforehand if their parameters need masking.

    Args:
        model: Module tree to convert.

    Returns:
        The same ``model`` with mask-aware twins in place.
    """
    for name, child in model.named_children():
        if type(child) is nn.Linear:
            setattr(model, name, RowMaskedLinear.from_module(child))
        elif type(child) is nn.Conv2d:
            setattr(model, name, RowMaskedConv2d.from_module(child))
        else:
            replace_with_row_masked(child)
    return model
