"""Gaussian likelihood for cosmological data vectors."""

from __future__ import annotations

import numpy as np


def chi_squared(
    data: np.ndarray,
    model: np.ndarray,
    cov_inv: np.ndarray,
) -> float:
    """Compute χ² = (d - m)ᵀ C⁻¹ (d - m).

    Parameters
    ----------
    data
        Observed data vector.
    model
        Model prediction at the same points.
    cov_inv
        Inverse covariance matrix.
    """
    residual = data - model
    return float(residual @ cov_inv @ residual)


def gaussian_log_likelihood(
    data: np.ndarray,
    model: np.ndarray,
    cov_inv: np.ndarray,
) -> float:
    """Compute the Gaussian log-likelihood = -χ²/2.

    Parameters
    ----------
    data
        Observed data vector.
    model
        Model prediction.
    cov_inv
        Inverse covariance matrix.
    """
    return -0.5 * chi_squared(data, model, cov_inv)
