"""Tracked angular power spectrum computation wrapping pyccl."""

from typing import Any

import numpy as np
from tisserande.tracking.annotations import ArrayArg, ConfigDict
from tisserande.tracking.decorator import track

from .cosmology import make_pyccl_cosmology
from .tracers import make_pyccl_tracer


@track
def angular_cl(
    cosmo_params: ConfigDict[dict[str, Any]],
    tracer1_config: ConfigDict[dict[str, Any]],
    tracer2_config: ConfigDict[dict[str, Any]],
    ell: ArrayArg[np.ndarray],
) -> np.ndarray:
    """Compute the angular (cross-)power spectrum C(ℓ) for a pair of tracers.

    Parameters
    ----------
    cosmo_params
        Cosmological parameters as a dict.
    tracer1_config
        First tracer configuration dict (from TracerConfig.model_dump()).
    tracer2_config
        Second tracer configuration dict (from TracerConfig.model_dump()).
    ell
        Angular multipole(s) at which to evaluate C(ℓ).

    Returns
    -------
    np.ndarray
        Angular power spectrum C(ℓ) array.
    """
    import pyccl

    cosmo = make_pyccl_cosmology(cosmo_params)
    tracer1 = make_pyccl_tracer(cosmo, tracer1_config)
    tracer2 = make_pyccl_tracer(cosmo, tracer2_config)
    return pyccl.angular_cl(cosmo, tracer1, tracer2, ell)
