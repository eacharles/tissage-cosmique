"""Analytic Gaussian covariance matrix for angular power spectrum data vectors."""

from __future__ import annotations

from typing import Any

import numpy as np

from .data_vector import (
    DataVectorResult,
    _resolve_bin_edges_for_pair,
    _resolve_ell_for_pair,
    compute_data_vector,
    make_data_vector_result,
)
from .ell_binning import bin_cl, pair_key

ARCMIN2_PER_SR = (180.0 * 60.0 / np.pi) ** 2


def noise_power(tracer_config: dict[str, Any], ell: np.ndarray) -> np.ndarray:
    """Compute noise power spectrum N(ℓ) for a single tracer.

    Parameters
    ----------
    tracer_config
        Tracer configuration dict with ``tracer_type``, ``n_gal``,
        ``sigma_epsilon``.
    ell
        Multipole array.

    Returns
    -------
    np.ndarray
        Noise power spectrum, same shape as ell.
    """
    n_gal = tracer_config.get("n_gal")
    if n_gal is None:
        return np.zeros_like(ell, dtype=float)

    n_gal_sr = n_gal * ARCMIN2_PER_SR

    tracer_type = tracer_config.get("tracer_type", "")
    if tracer_type == "weak_lensing":
        sigma_e = tracer_config.get("sigma_epsilon", 0.26)
        return np.full_like(ell, sigma_e**2 / n_gal_sr, dtype=float)
    if tracer_type == "number_counts":
        return np.full_like(ell, 1.0 / n_gal_sr, dtype=float)

    return np.zeros_like(ell, dtype=float)


def _get_cl_with_noise(
    t1: str, t2: str,
    all_cl: dict[str, np.ndarray],
    noise_dict: dict[str, np.ndarray],
    ell: np.ndarray,
) -> np.ndarray:
    """Get C̃_ij(ℓ) = C_ij(ℓ) + δ_ij * N_i(ℓ), evaluated at given ell."""
    key = pair_key(t1, t2)
    key_rev = pair_key(t2, t1)
    cl: np.ndarray = all_cl.get(key, all_cl.get(key_rev, np.zeros_like(ell)))
    if t1 == t2:
        n = noise_dict.get(t1)
        if n is not None:
            cl = cl + n
    return cl


def compute_gaussian_covariance(
    all_cl: dict[str, np.ndarray],
    noise_dict: dict[str, np.ndarray],
    dv_result: DataVectorResult,
    f_sky: float,
    dv_config: dict[str, Any],
    ell_fine: np.ndarray,
) -> np.ndarray:
    """Build the full Gaussian covariance matrix for a data vector.

    For binned data vectors, the covariance is computed at each fine ℓ
    and then averaged into bandpower bins with (2ℓ+1) weighting.

    Parameters
    ----------
    all_cl
        All signal C(ℓ) spectra keyed by ``"t1-t2"``, at ``ell_fine``.
    noise_dict
        Noise N(ℓ) per tracer name, at ``ell_fine``.
    dv_result
        The computed data vector result.
    f_sky
        Sky fraction.
    dv_config
        Data vector configuration (for resolving ℓ ranges and bin edges).
    ell_fine
        Fine multipole grid at which all_cl is evaluated.
    """
    pairs = dv_result.pairs
    offsets = dv_result.offsets
    lengths = dv_result.lengths
    n_dv = len(dv_result.data_vector)
    cov = np.zeros((n_dv, n_dv))

    for a_idx, (ti, tj) in enumerate(pairs):
        ell_a = _resolve_ell_for_pair(ti, tj, dv_config, ell_fine)
        edges_a = _resolve_bin_edges_for_pair(ti, tj, dv_config)
        n_a = lengths[a_idx]
        off_a = offsets[a_idx]

        for b_idx, (tm, tn) in enumerate(pairs):
            ell_b = _resolve_ell_for_pair(tm, tn, dv_config, ell_fine)
            edges_b = _resolve_bin_edges_for_pair(tm, tn, dv_config)
            off_b = offsets[b_idx]

            if not np.array_equal(ell_a, ell_b):
                continue

            ell = ell_a
            c_im = _get_cl_with_noise(ti, tm, all_cl, noise_dict, ell_fine)
            c_jn = _get_cl_with_noise(tj, tn, all_cl, noise_dict, ell_fine)
            c_in = _get_cl_with_noise(ti, tn, all_cl, noise_dict, ell_fine)
            c_jm = _get_cl_with_noise(tj, tm, all_cl, noise_dict, ell_fine)

            ell_mask = np.isin(ell_fine, ell)
            c_im_pair = c_im[ell_mask] if len(c_im) == len(ell_fine) else c_im
            c_jn_pair = c_jn[ell_mask] if len(c_jn) == len(ell_fine) else c_jn
            c_in_pair = c_in[ell_mask] if len(c_in) == len(ell_fine) else c_in
            c_jm_pair = c_jm[ell_mask] if len(c_jm) == len(ell_fine) else c_jm

            prefactor = 1.0 / ((2.0 * ell + 1.0) * f_sky)
            cov_per_ell = prefactor * (c_im_pair * c_jn_pair + c_in_pair * c_jm_pair)

            if edges_a is not None and edges_b is not None and np.array_equal(edges_a, edges_b):
                _, cov_binned = bin_cl(ell, cov_per_ell, edges_a, weights="2ell+1")
                for k in range(n_a):
                    cov[off_a + k, off_b + k] = cov_binned[k]
            elif edges_a is None and edges_b is None:
                for k in range(min(n_a, len(cov_per_ell))):
                    cov[off_a + k, off_b + k] = cov_per_ell[k]

    return cov


def compute_covariance_from_data_vector(
    cosmo_params: dict[str, Any],
    tracer_configs: dict[str, dict[str, Any]],
    dv_config: dict[str, Any],
    ell: np.ndarray,
    f_sky: float,
) -> tuple[np.ndarray, DataVectorResult, np.ndarray]:
    """Compute data vector and its Gaussian covariance in one call.

    Returns
    -------
    data_vector
        Flat data vector.
    dv_result
        DataVectorResult with metadata.
    covariance
        Covariance matrix.
    """
    dv = compute_data_vector(cosmo_params, tracer_configs, dv_config, ell)
    dv_result = make_data_vector_result(dv, dv_config, ell)

    import pyccl

    from .cosmology import make_pyccl_cosmology
    from .tracers import make_pyccl_tracer

    cosmo = make_pyccl_cosmology(cosmo_params)
    tracer_names = dv_config["tracer_names"]
    tracers = {n: make_pyccl_tracer(cosmo, tracer_configs[n]) for n in tracer_names}

    all_cl = {}
    noise_dict = {}
    for t1 in tracer_names:
        noise_dict[t1] = noise_power(tracer_configs[t1], ell)
        for t2 in tracer_names:
            key = pair_key(t1, t2)
            if key not in all_cl:
                all_cl[key] = pyccl.angular_cl(cosmo, tracers[t1], tracers[t2], ell)

    covariance = compute_gaussian_covariance(
        all_cl, noise_dict, dv_result, f_sky, dv_config, ell,
    )
    return dv, dv_result, covariance
