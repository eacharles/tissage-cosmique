"""Analytic Gaussian covariance matrix for angular power spectrum data vectors."""

from __future__ import annotations

from typing import Any

import numpy as np

from .data_vector import (
    DataVectorResult,
    compute_data_vector,
    make_data_vector_result,
)
from .ell_binning import pair_key

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
) -> np.ndarray:
    """Get C̃_ij = C_ij + δ_ij * N_i."""
    key = pair_key(t1, t2)
    key_rev = pair_key(t2, t1)
    cl: np.ndarray = all_cl.get(key, all_cl.get(key_rev, np.zeros(1)))
    if t1 == t2:
        cl = cl + noise_dict.get(t1, np.zeros_like(cl))
    return cl


def compute_gaussian_covariance(
    all_cl: dict[str, np.ndarray],
    noise_dict: dict[str, np.ndarray],
    dv_result: DataVectorResult,
    f_sky: float,
) -> np.ndarray:
    """Build the full Gaussian covariance matrix for a data vector.

    Parameters
    ----------
    all_cl
        All signal C(ℓ) spectra keyed by ``"t1-t2"``, evaluated at
        the per-pair ℓ arrays (unbinned).
    noise_dict
        Noise N(ℓ) per tracer name, evaluated at the same ℓ arrays.
    dv_result
        The computed data vector result with pairs, offsets, lengths.
    f_sky
        Sky fraction.

    Returns
    -------
    np.ndarray
        Covariance matrix of shape ``(n_dv, n_dv)``.
    """
    n_dv = len(dv_result.data_vector)
    cov = np.zeros((n_dv, n_dv))

    pairs = dv_result.pairs
    offsets = dv_result.offsets
    lengths = dv_result.lengths

    for a_idx, (ti, tj) in enumerate(pairs):
        ell_a = dv_result.ell_per_pair[pair_key(ti, tj)]
        n_a = lengths[a_idx]
        off_a = offsets[a_idx]

        for b_idx, (tm, tn) in enumerate(pairs):
            ell_b = dv_result.ell_per_pair[pair_key(tm, tn)]
            n_b = lengths[b_idx]
            off_b = offsets[b_idx]

            if n_a != n_b:
                continue
            if not np.allclose(ell_a, ell_b):
                continue

            ell = ell_a
            c_im = _get_cl_with_noise(ti, tm, all_cl, noise_dict)
            c_jn = _get_cl_with_noise(tj, tn, all_cl, noise_dict)
            c_in = _get_cl_with_noise(ti, tn, all_cl, noise_dict)
            c_jm = _get_cl_with_noise(tj, tm, all_cl, noise_dict)

            prefactor = 1.0 / ((2.0 * ell + 1.0) * f_sky)
            cov_diag = prefactor * (c_im * c_jn + c_in * c_jm)

            for k in range(n_a):
                cov[off_a + k, off_b + k] = cov_diag[k]

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

    all_cl = {}
    noise_dict = {}
    tracer_names = dv_config["tracer_names"]
    for t1 in tracer_names:
        noise_dict[t1] = noise_power(tracer_configs[t1], ell)
        for t2 in tracer_names:
            key = pair_key(t1, t2)
            if key not in all_cl:
                import pyccl

                from .cosmology import make_pyccl_cosmology
                from .tracers import make_pyccl_tracer

                cosmo = make_pyccl_cosmology(cosmo_params)
                tr1 = make_pyccl_tracer(cosmo, tracer_configs[t1])
                tr2 = make_pyccl_tracer(cosmo, tracer_configs[t2])
                all_cl[key] = pyccl.angular_cl(cosmo, tr1, tr2, ell)

    covariance = compute_gaussian_covariance(all_cl, noise_dict, dv_result, f_sky)
    return dv, dv_result, covariance
