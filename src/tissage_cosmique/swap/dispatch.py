"""Dispatch layer: swappable wrapper and original context manager."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from collections.abc import Iterator
from typing import Any

import numpy as np

from ..emulators.training import params_to_feature_matrix
from .registry import get_entry


def swappable(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a computation function to route through a registered emulator.

    When an emulator is registered and enabled for ``fn``, calls are
    intercepted: the input dict and scale-factor array are converted to a
    feature matrix and passed to ``emulator.predict``. When no emulator is
    registered or the swap is disabled, the original function runs normally.

    Usage::

        compute = swappable(comoving_angular_distance)
        result = compute(cosmo_params, a)  # emulator if registered, pyccl otherwise
    """

    @wraps(fn)
    def wrapper(cosmo_params: dict[str, Any], a: np.ndarray) -> np.ndarray:
        entry = get_entry(fn)
        if entry is not None and entry.enabled and entry.emulator.is_fitted:
            X = params_to_feature_matrix(cosmo_params, a, param_names=entry.param_names)
            return entry.emulator.predict(X)
        return fn(cosmo_params, a)

    wrapper._original = fn  # type: ignore[attr-defined]
    return wrapper


@contextmanager
def original(fn: Callable[..., Any]) -> Iterator[None]:
    """Context manager to temporarily bypass the emulator for a function.

    Usage::

        with original(comoving_angular_distance):
            truth = compute(params, a)  # always uses pyccl
    """
    entry = get_entry(fn)
    if entry is None:
        yield
        return

    was_enabled = entry.enabled
    entry.enabled = False
    try:
        yield
    finally:
        entry.enabled = was_enabled
