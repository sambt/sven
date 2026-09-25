"""JAX implementation of Sven.

A near drop-in SVD-based optimizer for models built with JAX / Flax.

Usage::

    from sven.jax import SvenWrapper, Sven

    wrapped = SvenWrapper(apply_fn, params, loss_fn)
    opt = Sven(wrapped, lr=0.1, k=64)

    losses, preds = wrapped.loss_and_grad((x, y))
    opt.step()

The Gram/kernel-trick variant applies the identical update without ever
materialising the ``(M, P)`` Jacobian::

    from sven.jax import GramSvenWrapper, SvenGram

    wrapped = GramSvenWrapper(apply_fn, params, loss_fn)
    opt = SvenGram(wrapped, lr=0.1, k=64)

    losses, preds = wrapped.loss_and_grad((x, y))
    opt.step()
"""

from .gram import GramSvenWrapper
from .pinv import pinv
from .sven import Sven, SvenGram
from .wrapper import SvenWrapper

__all__ = ["SvenWrapper", "Sven", "GramSvenWrapper", "SvenGram", "pinv"]
