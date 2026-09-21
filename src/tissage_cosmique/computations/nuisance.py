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


def perturb_dndz(
    z: np.ndarray,
    nz: np.ndarray,
    *,
    model: str | None = None,
    params: dict[str, float] | None = None,
    basis_z: list[float] | None = None,
    basis_amp: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Perturb a redshift distribution n(z) to model photo-z uncertainty.

    Parameters
    ----------
    z
        Redshift array.
    nz
        n(z) array (will be perturbed and re-normalized).
    model
        Perturbation model: ``"shift"``, ``"shift_stretch"``, or ``"comb"``.
    params
        Model parameters. For ``"shift"``: ``{"dz": float}``.
        For ``"shift_stretch"``: ``{"dz": float, "sigma_z": float}``.
    basis_z
        Z-nodes for ``"comb"`` perturbation.
    basis_amp
        Amplitudes at z-nodes for ``"comb"`` perturbation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Perturbed ``(z, nz)`` with preserved normalization.
    """
    if model is None:
        return z, nz

    if params is None:
        params = {}

    if model == "shift":
        nz_new = _perturb_shift(z, nz, params.get("dz", 0.0))
    elif model == "shift_stretch":
        nz_new = _perturb_shift_stretch(
            z, nz, params.get("dz", 0.0), params.get("sigma_z", 1.0),
        )
    elif model == "comb":
        if basis_z is None or basis_amp is None:
            return z, nz
        nz_new = _perturb_comb(z, nz, basis_z, basis_amp)
    else:
        raise ValueError(f"Unknown dndz perturbation model: {model}")

    nz_new = np.maximum(nz_new, 0.0)
    norm = np.trapezoid(nz_new, z)
    if norm > 0:
        orig_norm = np.trapezoid(nz, z)
        nz_new = nz_new * (orig_norm / norm)

    return z, nz_new


def _perturb_shift(
    z: np.ndarray, nz: np.ndarray, dz: float,
) -> np.ndarray:
    """Shift n(z) by Δz: n(z) → n(z - Δz)."""
    f = interp1d(z, nz, kind="linear", bounds_error=False, fill_value=0.0)
    return f(z - dz)


def _perturb_shift_stretch(
    z: np.ndarray, nz: np.ndarray, dz: float, sigma_z: float,
) -> np.ndarray:
    """Shift and stretch n(z): n(z) → n((z - Δz) / σ_z)."""
    z_mean = np.average(z, weights=np.maximum(nz, 0.0) + 1e-30)
    z_transformed = z_mean + (z - z_mean - dz) / sigma_z
    f = interp1d(z, nz, kind="linear", bounds_error=False, fill_value=0.0)
    return f(z_transformed)


def _perturb_comb(
    z: np.ndarray, nz: np.ndarray,
    basis_z: list[float], basis_amp: list[float],
) -> np.ndarray:
    """Multiplicative comb perturbation: n(z) → n(z) * (1 + Σ a_k φ_k(z))."""
    f = interp1d(
        basis_z, basis_amp,
        kind="linear", bounds_error=False, fill_value=0.0,
    )
    perturbation = 1.0 + f(z)
    return nz * perturbation
