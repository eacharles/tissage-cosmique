"""Fisher matrix forecast for cosmological parameter constraints."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FisherResult:
    """Result of a Fisher matrix computation."""

    fisher_matrix: np.ndarray
    param_names: list[str]
    fiducial: dict[str, float]
    derivatives: dict[str, np.ndarray] = field(repr=False)

    @property
    def covariance(self) -> np.ndarray:
        """Parameter covariance matrix (F⁻¹)."""
        return np.linalg.inv(self.fisher_matrix)

    def sigma(self, param: str) -> float:
        """Marginalized 1σ constraint on a parameter."""
        idx = self.param_names.index(param)
        return float(np.sqrt(self.covariance[idx, idx]))

    def sigma_conditional(self, param: str) -> float:
        """Conditional (unmarginalized) 1σ constraint."""
        idx = self.param_names.index(param)
        return float(1.0 / np.sqrt(self.fisher_matrix[idx, idx]))

    def correlation_matrix(self) -> np.ndarray:
        """Parameter correlation matrix from F⁻¹."""
        cov = self.covariance
        diag = np.sqrt(np.diag(cov))
        return cov / np.outer(diag, diag)

    def ellipse(
        self, p1: str, p2: str, *, n_sigma: float = 1.0, n_points: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute a 2D confidence ellipse for a pair of parameters.

        Parameters
        ----------
        p1, p2
            Parameter names.
        n_sigma
            Number of sigma for the ellipse contour.
        n_points
            Number of points on the ellipse.

        Returns
        -------
        x, y
            Arrays of (p1, p2) values tracing the ellipse.
        """
        i = self.param_names.index(p1)
        j = self.param_names.index(p2)
        cov = self.covariance
        sub_cov = np.array([[cov[i, i], cov[i, j]], [cov[j, i], cov[j, j]]])

        eigenvalues, eigenvectors = np.linalg.eigh(sub_cov)
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        a = n_sigma * np.sqrt(eigenvalues[0])
        b = n_sigma * np.sqrt(eigenvalues[1])

        t = np.linspace(0, 2 * np.pi, n_points)
        x_raw = a * np.cos(t)
        y_raw = b * np.sin(t)

        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x = self.fiducial[p1] + cos_a * x_raw - sin_a * y_raw
        y = self.fiducial[p2] + sin_a * x_raw + cos_a * y_raw

        return x, y


def compute_fisher_matrix(
    model_fn: Callable[[dict[str, Any]], np.ndarray],
    fiducial_params: dict[str, float],
    cov_inv: np.ndarray,
    param_names: list[str],
    *,
    step_sizes: dict[str, float] | None = None,
) -> FisherResult:
    """Compute the Fisher information matrix via numerical derivatives.

    Parameters
    ----------
    model_fn
        Callable that takes a parameter dict and returns a data vector.
    fiducial_params
        Fiducial parameter values (the expansion point).
    cov_inv
        Inverse covariance matrix.
    param_names
        Parameters to include in the Fisher matrix.
    step_sizes
        Step size for each parameter. Defaults to 1% of |fiducial| or 0.01.

    Returns
    -------
    FisherResult
        Fisher matrix, parameter names, fiducial values, and derivatives.
    """
    if step_sizes is None:
        step_sizes = {}

    n_params = len(param_names)
    derivatives: dict[str, np.ndarray] = {}

    for pname in param_names:
        fid_val = fiducial_params[pname]
        step = step_sizes.get(pname, max(abs(fid_val) * 0.01, 0.01))

        params_plus = {**fiducial_params, pname: fid_val + step}
        params_minus = {**fiducial_params, pname: fid_val - step}

        dv_plus = model_fn(params_plus)
        dv_minus = model_fn(params_minus)

        derivatives[pname] = (dv_plus - dv_minus) / (2.0 * step)

    fisher = np.zeros((n_params, n_params))
    for i, pi in enumerate(param_names):
        for j, pj in enumerate(param_names):
            fisher[i, j] = float(derivatives[pi] @ cov_inv @ derivatives[pj])

    return FisherResult(
        fisher_matrix=fisher,
        param_names=param_names,
        fiducial={p: fiducial_params[p] for p in param_names},
        derivatives=derivatives,
    )
