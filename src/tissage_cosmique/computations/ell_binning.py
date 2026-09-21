"""Multipole binning utilities for angular power spectra."""

from __future__ import annotations

import numpy as np


def bin_cl(
    ell: np.ndarray,
    cl: np.ndarray,
    bin_edges: np.ndarray | list[float],
    *,
    weights: str = "2ell+1",
) -> tuple[np.ndarray, np.ndarray]:
    """Bin C(ℓ) into bandpowers.

    Parameters
    ----------
    ell
        Multipole values.
    cl
        C(ℓ) values (same length as ell).
    bin_edges
        Bin edge values. N edges → N-1 bins.
    weights
        Weighting scheme: ``"2ell+1"`` (proportional to number of modes,
        default) or ``"uniform"``.

    Returns
    -------
    ell_eff
        Effective multipole per bin (weighted mean of ℓ within each bin).
    cl_binned
        Binned C(ℓ) values.
    """
    bin_edges = np.asarray(bin_edges)
    n_bins = len(bin_edges) - 1

    if weights == "2ell+1":
        w = 2.0 * ell + 1.0
    elif weights == "uniform":
        w = np.ones_like(ell, dtype=float)
    else:
        raise ValueError(f"Unknown weights: {weights}")

    ell_eff = np.zeros(n_bins)
    cl_binned = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (ell >= bin_edges[i]) & (ell < bin_edges[i + 1])
        if not np.any(mask):
            continue
        w_bin = w[mask]
        w_sum = w_bin.sum()
        ell_eff[i] = np.sum(ell[mask] * w_bin) / w_sum
        cl_binned[i] = np.sum(cl[mask] * w_bin) / w_sum

    return ell_eff, cl_binned


def make_log_bins(ell_min: float, ell_max: float, n_bins: int) -> np.ndarray:
    """Generate logarithmically-spaced bin edges.

    Returns
    -------
    np.ndarray
        Array of ``n_bins + 1`` bin edges.
    """
    return np.logspace(np.log10(ell_min), np.log10(ell_max), n_bins + 1)


def make_linear_bins(ell_min: float, ell_max: float, n_bins: int) -> np.ndarray:
    """Generate linearly-spaced bin edges.

    Returns
    -------
    np.ndarray
        Array of ``n_bins + 1`` bin edges.
    """
    return np.linspace(ell_min, ell_max, n_bins + 1)


def pair_key(t1: str, t2: str) -> str:
    """Create a canonical pair key from two tracer names."""
    return f"{t1}-{t2}"
