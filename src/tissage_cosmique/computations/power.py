"""Tracked linear matter power spectrum computation wrapping pyccl."""

from typing import Any

import numpy as np
from tisserande.tracking.annotations import ArrayArg, ConfigDict, Param
from tisserande.tracking.decorator import track

from .cosmology import make_pyccl_cosmology


@track
def linear_matter_power(
    cosmo_params: ConfigDict[dict[str, Any]],
    k: ArrayArg[np.ndarray],
    a: Param[float],
) -> np.ndarray:
    """Compute the linear matter power spectrum.

    Parameters
    ----------
    cosmo_params
        Cosmological parameters as a dict (e.g., from CosmologyParams.model_dump()).
    k
        Wavenumber(s) in 1/Mpc. Can be a single value or array.
    a
        Scale factor (scalar). Must be in (0, 1].

    Returns
    -------
    np.ndarray
        Linear matter power spectrum P(k) in Mpc^3.
    """
    import pyccl

    cosmo = make_pyccl_cosmology(cosmo_params)
    return pyccl.linear_matter_power(cosmo, k, a)
