"""Utilities for generating emulator training data from tracked computations."""

from collections.abc import Callable
from typing import Any

import numpy as np


def build_training_data(
    computation_fn: Callable[..., np.ndarray],
    param_samples: list[dict[str, Any]],
    a_grid: np.ndarray,
    param_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Run a computation over a parameter grid and collect (X, y) training pairs.

    Each row of X is ``[param_values..., a_i]`` and each element of y is the
    corresponding scalar output.

    Parameters
    ----------
    computation_fn
        A function with signature ``(params_dict, a_array) -> result_array``.
    param_samples
        List of parameter dictionaries, one per cosmology.
    a_grid
        Shared array of scale factor values.
    param_names
        Ordered list of parameter keys to extract from each dict.

    Returns
    -------
    X
        Feature matrix of shape ``(n_samples * n_a, len(param_names) + 1)``.
    y
        Target array of shape ``(n_samples * n_a,)``.
    """
    X_rows: list[list[float]] = []
    y_values: list[float] = []

    for params in param_samples:
        result = computation_fn(params, a_grid)
        pvals = [params[k] for k in param_names]
        for i, a_val in enumerate(a_grid):
            X_rows.append(pvals + [float(a_val)])
            y_values.append(float(result[i]))

    return np.array(X_rows), np.array(y_values)


def params_to_feature_matrix(
    cosmo_params: dict[str, Any],
    a: np.ndarray,
    param_names: list[str],
) -> np.ndarray:
    """Build a feature matrix for one parameter set over an array of scale factors.

    Parameters
    ----------
    cosmo_params
        Cosmological parameter dictionary.
    a
        Scale factor array of shape ``(n_a,)``.
    param_names
        Ordered list of parameter keys (must match training order).

    Returns
    -------
    np.ndarray
        Feature matrix of shape ``(n_a, len(param_names) + 1)``.
    """
    pvals = [cosmo_params[k] for k in param_names]
    n_a = len(a)
    X = np.empty((n_a, len(param_names) + 1))
    X[:, :-1] = pvals
    X[:, -1] = a
    return X
