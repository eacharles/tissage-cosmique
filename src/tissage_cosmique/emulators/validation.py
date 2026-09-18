"""Validation tools for assessing emulator accuracy and uncertainty calibration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from macon.common import unexpected

from .base import Emulator
from .training import build_training_data


@dataclass
class ValidationResult:
    """Container for emulator validation metrics."""

    n_test: int
    mae: float
    rmse: float
    max_abs_error: float
    mean_relative_error: float
    max_relative_error: float
    r2_score: float
    percentile_95_error: float
    residuals: np.ndarray


@dataclass
class CalibrationResult:
    """Container for uncertainty calibration metrics (GP-specific)."""

    expected_coverage: np.ndarray
    observed_coverage: np.ndarray
    mean_std: float
    std_vs_error_correlation: float


def validate_emulator(
    emulator: Emulator,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> ValidationResult:
    """Compute accuracy metrics for an emulator on a test set.

    Parameters
    ----------
    emulator
        A fitted emulator.
    X_test
        Test feature matrix of shape (n_test, n_features).
    y_test
        Test target array of shape (n_test,).
    """
    y_pred = emulator.predict(X_test)
    residuals = y_pred - y_test
    abs_errors = np.abs(residuals)

    nonzero = np.abs(y_test) > 1e-10
    if not unexpected(not np.any(nonzero)):
        rel_errors = abs_errors[nonzero] / np.abs(y_test[nonzero])
        mean_rel = float(np.mean(rel_errors))
        max_rel = float(np.max(rel_errors))
    else:  # pragma: no cover
        mean_rel = 0.0
        max_rel = 0.0

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return ValidationResult(
        n_test=len(y_test),
        mae=float(np.mean(abs_errors)),
        rmse=float(np.sqrt(np.mean(residuals**2))),
        max_abs_error=float(np.max(abs_errors)),
        mean_relative_error=mean_rel,
        max_relative_error=max_rel,
        r2_score=r2,
        percentile_95_error=float(np.percentile(abs_errors, 95)),
        residuals=residuals,
    )


def validate_against_computation(
    emulator: Emulator,
    computation_fn: Callable[..., np.ndarray],
    test_params: list[dict[str, Any]],
    a_grid: np.ndarray,
    param_names: list[str],
) -> ValidationResult:
    """Validate an emulator by comparing against a reference computation.

    Parameters
    ----------
    emulator
        A fitted emulator.
    computation_fn
        The original computation function with signature (params_dict, a_array) -> array.
    test_params
        List of parameter dictionaries for the test set.
    a_grid
        Scale factor grid to evaluate at.
    param_names
        Ordered parameter names (must match training).
    """
    X_test, y_test = build_training_data(computation_fn, test_params, a_grid, param_names=param_names)
    return validate_emulator(emulator, X_test, y_test)


def cross_validate(
    emulator_factory: Callable[[], Emulator],
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_folds: int = 5,
) -> list[ValidationResult]:
    """Run k-fold cross-validation on an emulator.

    Parameters
    ----------
    emulator_factory
        A callable that returns a fresh (unfitted) emulator instance.
    X
        Full feature matrix.
    y
        Full target array.
    n_folds
        Number of cross-validation folds.

    Returns
    -------
    list[ValidationResult]
        One result per fold.
    """
    n = len(y)
    indices = np.arange(n)
    rng = np.random.default_rng(0)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    results: list[ValidationResult] = []
    for i in range(n_folds):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])

        emu = emulator_factory()
        emu.fit(X[train_idx], y[train_idx])
        result = validate_emulator(emu, X[test_idx], y[test_idx])
        results.append(result)

    return results


def check_calibration(
    emulator: Emulator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    levels: list[float] | None = None,
) -> CalibrationResult:
    """Check whether predicted uncertainties are well-calibrated.

    Parameters
    ----------
    emulator
        A fitted emulator with a ``predict_with_std`` method.
    X_test
        Test feature matrix.
    y_test
        Test target array.
    levels
        Coverage levels to check (default: [0.68, 0.95, 0.99]).

    Raises
    ------
    TypeError
        If the emulator does not support ``predict_with_std``.
    """
    if not hasattr(emulator, "predict_with_std"):
        raise TypeError(f"{type(emulator).__name__} does not support predict_with_std")

    if levels is None:
        levels = [0.68, 0.95, 0.99]

    mean, std = emulator.predict_with_std(X_test)
    abs_errors = np.abs(mean - y_test)

    z_scores_for_levels = {
        0.68: 1.0,
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576,
    }

    observed = []
    for level in levels:
        z = z_scores_for_levels.get(level, float(np.abs(__import__("scipy").stats.norm.ppf((1 - level) / 2))))
        within = abs_errors <= z * std
        observed.append(float(np.mean(within)))

    nonzero_std = std[std > 0]
    nonzero_err = abs_errors[std > 0]
    if not unexpected(len(nonzero_std) <= 1):
        corr = float(np.corrcoef(nonzero_std, nonzero_err)[0, 1])
    else:  # pragma: no cover
        corr = 0.0

    return CalibrationResult(
        expected_coverage=np.array(levels),
        observed_coverage=np.array(observed),
        mean_std=float(np.mean(std)),
        std_vs_error_correlation=corr,
    )
