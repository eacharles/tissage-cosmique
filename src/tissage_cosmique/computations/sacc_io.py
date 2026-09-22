"""SACC file I/O: export and import data vectors, covariance, and tracers."""

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


_REVERSE_SACC_TYPE_MAP: dict[str, tuple[str, str]] = {
    "galaxy_shear_cl_ee": ("weak_lensing", "weak_lensing"),
    "galaxy_shear_cl_bb": ("weak_lensing", "weak_lensing"),
    "galaxy_density_cl": ("number_counts", "number_counts"),
    "galaxy_shearDensity_cl_e": ("weak_lensing", "number_counts"),
    "galaxy_shearDensity_cl_b": ("weak_lensing", "number_counts"),
    "galaxy_convergenceDensity_cl": ("weak_lensing", "number_counts"),
    "cmb_convergence_cl": ("cmb_lensing", "cmb_lensing"),
    "cmbGalaxy_convergenceShear_cl_e": ("cmb_lensing", "weak_lensing"),
    "cmbGalaxy_convergenceDensity_cl": ("cmb_lensing", "number_counts"),
    "cl_00": ("unknown", "unknown"),
    "cl_ee": ("weak_lensing", "weak_lensing"),
}


def _infer_tracer_type(sacc_obj: Any, tracer_name: str) -> str:
    """Infer our tracer_type from sacc tracer metadata and data types."""
    tracer = sacc_obj.tracers[tracer_name]
    tclass = type(tracer).__name__

    if tclass == "MapTracer":
        return "cmb_lensing"

    dtype_hints: dict[str, int] = {}
    for d in sacc_obj.data:
        if tracer_name in d.tracers:
            pair_types = _REVERSE_SACC_TYPE_MAP.get(d.data_type)
            if pair_types:
                for tt in pair_types:
                    dtype_hints[tt] = dtype_hints.get(tt, 0) + 1

    if "weak_lensing" in dtype_hints and "number_counts" not in dtype_hints:
        return "weak_lensing"
    if "number_counts" in dtype_hints and "weak_lensing" not in dtype_hints:
        return "number_counts"
    if "weak_lensing" in dtype_hints and "number_counts" in dtype_hints:
        if dtype_hints["weak_lensing"] >= dtype_hints["number_counts"]:
            return "weak_lensing"
        return "number_counts"

    return "weak_lensing"


def load_sacc(
    path: str,
) -> tuple[
    dict[str, dict[str, Any]], list[tuple[str, str]],
    dict[str, np.ndarray], dict[str, np.ndarray],
    np.ndarray | None,
]:
    """Load a SACC file and extract tracers, C(ℓ), and covariance.

    Parameters
    ----------
    path
        Path to a SACC FITS file.

    Returns
    -------
    tracer_configs
        ``{name: TracerConfig-like dict}`` for each tracer.
    pairs
        Ordered list of ``(tracer1, tracer2)`` pairs found.
    ell_dict
        ``{pair_key: ell_array}`` for each pair.
    cl_dict
        ``{pair_key: cl_array}`` for each pair.
    covariance
        Covariance matrix, or None if not present.
    """
    import sacc

    s = sacc.Sacc.load_fits(path)

    tracer_configs: dict[str, dict[str, Any]] = {}
    for name in s.tracers:
        tracer = s.tracers[name]
        ttype = _infer_tracer_type(s, name)
        config: dict[str, Any] = {
            "name": name,
            "tracer_type": ttype,
        }
        if hasattr(tracer, "z") and hasattr(tracer, "nz"):
            config["dndz_z"] = tracer.z.tolist()
            config["dndz_nz"] = tracer.nz.tolist()
        if ttype == "cmb_lensing":
            config["z_source"] = 1100.0
        tracer_configs[name] = config

    seen_pairs: list[tuple[str, str]] = []
    seen_keys: set[str] = set()
    for d in s.data:
        t1, t2 = d.tracers
        key = pair_key(t1, t2)
        if key not in seen_keys:
            seen_pairs.append((t1, t2))
            seen_keys.add(key)

    ell_dict: dict[str, np.ndarray] = {}
    cl_dict: dict[str, np.ndarray] = {}

    for dtype_name in set(d.data_type for d in s.data):
        for t1, t2 in seen_pairs:
            try:
                ells, cls = s.get_ell_cl(dtype_name, t1, t2)
                if len(ells) > 0:
                    key = pair_key(t1, t2)
                    ell_dict[key] = ells
                    cl_dict[key] = cls
            except Exception:
                continue

    covariance = None
    if s.covariance is not None:
        covariance = s.covariance.dense

    return tracer_configs, seen_pairs, ell_dict, cl_dict, covariance


def load_sacc_to_db(
    path: str,
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """Load a SACC file and store tracers in the database.

    Parameters
    ----------
    path
        Path to a SACC FITS file.
    prefix
        Optional prefix for tracer names in the DB (e.g., "y3_").

    Returns
    -------
    dict
        Summary with keys: "tracer_names", "pairs", "n_data_points",
        "has_covariance".
    """
    from ..local_sync import tracer_config as tc_ops

    tracer_configs, pairs, ell_dict, cl_dict, covariance = load_sacc(path)

    for name, config in tracer_configs.items():
        db_name = f"{prefix}{name}" if prefix else name
        try:
            tc_ops.create_row(name=db_name, **{k: v for k, v in config.items() if k != "name"})
        except Exception:
            pass

    n_data = sum(len(v) for v in cl_dict.values())
    return {
        "tracer_names": list(tracer_configs.keys()),
        "pairs": pairs,
        "n_data_points": n_data,
        "has_covariance": covariance is not None,
    }
