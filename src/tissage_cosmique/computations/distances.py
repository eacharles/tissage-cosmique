"""Tracked cosmological distance computations wrapping pyccl."""

from typing import Any

import numpy as np
from tisserande.tracking.annotations import ArrayArg, ConfigDict
from tisserande.tracking.decorator import track

from .cosmology import make_pyccl_cosmology


@track
def comoving_angular_distance(
    cosmo_params: ConfigDict[dict[str, Any]],
    a: ArrayArg[np.ndarray],
) -> np.ndarray:
    """Compute comoving angular diameter distance.

    Parameters
    ----------
    cosmo_params
        Cosmological parameters as a dict (e.g., from CosmologyParams.model_dump()).
    a
        Scale factor(s). Must be in (0, 1].

    Returns
    -------
    np.ndarray
        Comoving angular diameter distance(s) in Mpc.
    """
    cosmo = make_pyccl_cosmology(cosmo_params)
    return cosmo.comoving_angular_distance(a)
