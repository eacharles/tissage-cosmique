"""Nuisance parameter evaluation: parametric models and functional interpolation."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def evaluate_nuisance(
    z: np.ndarray,
    *,
    model: str | None = None,
    params: dict[str, float] | None = None,
    nodes_z: list[float] | None = None,
    nodes_amp: list[float] | None = None,
    static_z: list[float] | None = None,
    static_values: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Evaluate a nuisance function on a redshift grid.

    Resolution priority: parametric > functional (nodes) > static arrays > None.

    Parameters
    ----------
    z
        Redshift array to evaluate on.
    model
        Parametric model name: "constant", "linear", "quadratic", "power_law".
    params
        Parameters for the parametric model (e.g., {"a0": 1.5, "a1": 0.2}).
    nodes_z
        Z-node positions for functional interpolation.
    nodes_amp
        Amplitude values at each z-node.
    static_z
        Static redshift array (pre-computed).
    static_values
        Static function values (pre-computed).

    Returns
    -------
    tuple[np.ndarray, np.ndarray] | None
        ``(z, f(z))`` tuple, or None if no specification is provided.
    """
    if model is not None and params is not None:
        return z, _evaluate_parametric(z, model, params)

    if nodes_z is not None and nodes_amp is not None:
        return z, _evaluate_functional(z, nodes_z, nodes_amp)

    if static_z is not None and static_values is not None:
        return np.array(static_z), np.array(static_values)

    return None


def _evaluate_parametric(
    z: np.ndarray,
    model: str,
    params: dict[str, float],
) -> np.ndarray:
    """Evaluate a parametric nuisance model."""
    if model == "constant":
        return np.full_like(z, params.get("a0", 1.0))

    if model == "linear":
        a0 = params.get("a0", 0.0)
        a1 = params.get("a1", 0.0)
        return a0 + a1 * z

    if model == "quadratic":
        a0 = params.get("a0", 0.0)
        a1 = params.get("a1", 0.0)
        a2 = params.get("a2", 0.0)
        return a0 + a1 * z + a2 * z**2

    if model == "power_law":
        a0 = params.get("a0", 1.0)
        a1 = params.get("a1", 0.0)
        z_pivot = params.get("z_pivot", 0.5)
        return a0 * ((1 + z) / (1 + z_pivot)) ** a1

    raise ValueError(f"Unknown parametric model: {model}")


def _evaluate_functional(
    z: np.ndarray,
    nodes_z: list[float],
    nodes_amp: list[float],
) -> np.ndarray:
    """Interpolate node amplitudes onto a redshift grid."""
    f = interp1d(
        nodes_z, nodes_amp,
        kind="linear", bounds_error=False, fill_value="extrapolate",
    )
    return f(z)
