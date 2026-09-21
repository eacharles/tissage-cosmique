"""Export data vectors, covariance matrices, and tracer configs to SACC files."""

from __future__ import annotations

from typing import Any

import numpy as np

from .data_vector import DataVectorResult
from .ell_binning import pair_key

_SACC_TYPE_MAP: dict[tuple[str, str], str] = {
    ("weak_lensing", "weak_lensing"): "galaxy_shear_cl_ee",
    ("number_counts", "number_counts"): "galaxy_density_cl",
    ("weak_lensing", "number_counts"): "galaxy_shearDensity_cl_e",
    ("number_counts", "weak_lensing"): "galaxy_shearDensity_cl_e",
    ("cmb_lensing", "cmb_lensing"): "cmb_convergence_cl",
    ("weak_lensing", "cmb_lensing"): "cmbGalaxy_convergenceShear_cl_e",
    ("cmb_lensing", "weak_lensing"): "cmbGalaxy_convergenceShear_cl_e",
    ("number_counts", "cmb_lensing"): "cmbGalaxy_convergenceDensity_cl",
    ("cmb_lensing", "number_counts"): "cmbGalaxy_convergenceDensity_cl",
}


def _get_sacc_data_type(
    t1_type: str, t2_type: str,
) -> str:
    """Map tracer types to a sacc standard data type string."""
    key = (t1_type, t2_type)
    dtype = _SACC_TYPE_MAP.get(key)
    if dtype is None:
        return "cl_00"
    return dtype


def _get_sacc_tracer_pair(
    t1_name: str, t2_name: str,
    t1_type: str, t2_type: str,
) -> tuple[str, str]:
    """Order the tracer names for sacc (shear before density, CMB first)."""
    key = (t1_type, t2_type)
    if key in _SACC_TYPE_MAP:
        return t1_name, t2_name
    rev = (t2_type, t1_type)
    if rev in _SACC_TYPE_MAP:
        return t2_name, t1_name
    return t1_name, t2_name


def build_sacc(
    dv_result: DataVectorResult,
    tracer_configs: dict[str, dict[str, Any]],
    covariance: np.ndarray | None = None,
) -> Any:
    """Build a sacc.Sacc object from a data vector result and tracer configs.

    Parameters
    ----------
    dv_result
        Computed data vector with pairs, ell_per_pair, offsets, lengths.
    tracer_configs
        ``{tracer_name: config_dict}`` for all tracers in the data vector.
    covariance
        Covariance matrix matching the data vector. If None, no covariance
        is attached.

    Returns
    -------
    sacc.Sacc
        A sacc object with tracers, data points, and optionally covariance.
    """
    import sacc

    s = sacc.Sacc()

    for name, config in tracer_configs.items():
        ttype = config.get("tracer_type", "")
        dndz_z = config.get("dndz_z")
        dndz_nz = config.get("dndz_nz")

        if dndz_z is not None and dndz_nz is not None:
            s.add_tracer("NZ", name, np.array(dndz_z), np.array(dndz_nz))
        elif ttype == "cmb_lensing":
            s.add_tracer("Map", name, 0, np.array([0.0]), np.array([0.0]))
        else:
            s.add_tracer("misc", name)

    for i, (t1, t2) in enumerate(dv_result.pairs):
        key = pair_key(t1, t2)
        ell_eff = dv_result.ell_per_pair[key]
        cl = dv_result.get_cl(t1, t2)

        t1_type = tracer_configs[t1].get("tracer_type", "")
        t2_type = tracer_configs[t2].get("tracer_type", "")
        dtype = _get_sacc_data_type(t1_type, t2_type)
        sacc_t1, sacc_t2 = _get_sacc_tracer_pair(t1, t2, t1_type, t2_type)

        for ell_val, cl_val in zip(ell_eff, cl):
            s.add_data_point(dtype, (sacc_t1, sacc_t2), float(cl_val), ell=float(ell_val))

    if covariance is not None:
        s.add_covariance(covariance)

    return s


def save_sacc(
    path: str,
    dv_result: DataVectorResult,
    tracer_configs: dict[str, dict[str, Any]],
    covariance: np.ndarray | None = None,
) -> None:
    """Build and save a sacc file.

    Parameters
    ----------
    path
        Output file path (typically .fits or .sacc).
    dv_result
        Computed data vector result.
    tracer_configs
        Tracer configuration dicts.
    covariance
        Optional covariance matrix.
    """
    s = build_sacc(dv_result, tracer_configs, covariance)
    s.save_fits(path)
