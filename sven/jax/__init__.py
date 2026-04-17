"""JAX implementation of Sven.

A near drop-in SVD-based optimizer for models built with JAX / Flax.

Usage::

    from sven.jax import SvenWrapper, Sven

    wrapped = SvenWrapper(apply_fn, params, loss_fn)
    opt = Sven(wrapped, lr=0.1, k=64)

    losses, preds = wrapped.loss_and_grad((x, y))
    opt.step()
"""

from .wrapper import SvenWrapper
from .sven import Sven
from .pinv import pinv

__all__ = ["SvenWrapper", "Sven", "pinv"]
