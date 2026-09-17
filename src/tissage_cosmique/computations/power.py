"""Tracked linear matter power spectrum computation wrapping pyccl."""

from typing import Any

import numpy as np
from tisserande.tracking.annotations import ArrayArg, ConfigDict
from tisserande.tracking.decorator import track

from .cosmology import make_pyccl_cosmology


@track
def linear_matter_power(
    cosmo_params: ConfigDict[dict[str, Any]],
    k: ArrayArg[np.ndarray],
    a: ArrayArg[np.ndarray],
) -> np.ndarray:
    """Compute the linear matter power spectrum.

    Parameters
    ----------
    cosmo_params
        Cosmological parameters as a dict (e.g., from CosmologyParams.model_dump()).
    k
        Wavenumber(s) in 1/Mpc.
    a
        Scale factor(s). Scalar or 1-D array. Must be in (0, 1].
        When ``a`` is a 1-D array, returns a 2-D array of shape ``(n_a, n_k)``.

    Returns
    -------
    np.ndarray
        Linear matter power spectrum P(k, a) in Mpc^3.
        Shape is ``(n_k,)`` for scalar ``a``, or ``(n_a, n_k)`` for array ``a``.
    """
    import pyccl

    cosmo = make_pyccl_cosmology(cosmo_params)
    return pyccl.linear_matter_power(cosmo, k, a)
