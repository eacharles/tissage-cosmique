"""Emulator inversion: find input parameters from target outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from macon.common import unexpected
from scipy.optimize import minimize

from .base import Emulator


@dataclass
class InversionResult:
    """Result of an emulator inversion."""

    x_solution: dict[str, float]
    y_predicted: np.ndarray
    y_target: np.ndarray
    residual: float
    success: bool
    message: str


def _resolve_indices(
    feature_names: list[str],
    free_params: list[str],
    fixed_params: dict[str, float | np.ndarray],
) -> tuple[list[int], list[int]]:
    free_indices = [feature_names.index(p) for p in free_params]
    fixed_indices = [feature_names.index(p) for p in fixed_params]
    return free_indices, fixed_indices


def _build_feature_matrix(
    free_values: np.ndarray,
    free_indices: list[int],
    fixed_values: dict[int, float | np.ndarray],
    n_features: int,
    n_targets: int,
) -> np.ndarray:
    X = np.empty((n_targets, n_features))
    for idx in free_indices:
        X[:, idx] = free_values[free_indices.index(idx)]
    for idx, val in fixed_values.items():
        if isinstance(val, np.ndarray):
            X[:, idx] = val
        else:
            X[:, idx] = val
    return X


def invert_minimize(
    emulator: Emulator,
    y_target: np.ndarray,
    free_params: list[str],
    fixed_params: dict[str, float | np.ndarray],
    *,
    x0: dict[str, float] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    method: str = "L-BFGS-B",
) -> InversionResult:
    """Find free parameter values that reproduce target outputs.

    Uses ``scipy.optimize.minimize`` to minimize
    ``||emulator.predict(X) - y_target||^2``.

    Parameters
    ----------
    emulator
        A fitted emulator.
    y_target
        Target output array of shape ``(n_targets,)``.
    free_params
        Names of parameters to solve for.
    fixed_params
        ``{name: value}`` for fixed features. Values can be scalar (broadcast
        to all targets) or arrays of length ``n_targets``.
    x0
        Initial guess ``{name: value}`` for free parameters. Defaults to
        midpoint of bounds or 0.
    bounds
        ``{name: (low, high)}`` for each free parameter.
    method
        Scipy minimize method (default: L-BFGS-B).
    """
    feature_names = emulator.feature_names
    if unexpected(feature_names is None):
        raise ValueError("Emulator must have feature_names set for inversion")
    assert feature_names is not None

    y_target = np.atleast_1d(y_target)
    n_targets = len(y_target)
    n_features = len(feature_names)

    free_indices, fixed_indices = _resolve_indices(feature_names, free_params, fixed_params)
    fixed_idx_val = {feature_names.index(k): v for k, v in fixed_params.items()}

    if x0 is not None:
        x0_arr = np.array([x0[p] for p in free_params])
    elif bounds is not None:
        x0_arr = np.array([(bounds[p][0] + bounds[p][1]) / 2 for p in free_params])
    else:  # pragma: no cover
        x0_arr = np.zeros(len(free_params))

    scipy_bounds = None
    if bounds is not None:
        scipy_bounds = [bounds[p] for p in free_params]

    y_scale = np.maximum(np.abs(y_target).mean(), 1e-10)

    def objective(free_values: np.ndarray) -> float:
        X = _build_feature_matrix(free_values, free_indices, fixed_idx_val, n_features, n_targets)
        y_pred = emulator.predict(X)
        return float(np.sum(((y_pred - y_target) / y_scale) ** 2))

    result = minimize(objective, x0_arr, method=method, bounds=scipy_bounds)

    X_final = _build_feature_matrix(result.x, free_indices, fixed_idx_val, n_features, n_targets)
    y_pred = emulator.predict(X_final)
    rel_residual = float(np.sqrt(np.mean(((y_pred - y_target) / y_scale) ** 2)))

    return InversionResult(
        x_solution={p: float(v) for p, v in zip(free_params, result.x)},
        y_predicted=y_pred,
        y_target=y_target,
        residual=rel_residual,
        success=bool(result.success),
        message=str(result.message),
    )
