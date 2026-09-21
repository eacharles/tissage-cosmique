"""Helper to create pyccl Tracer objects from tracer configuration dictionaries."""

from typing import Any

import numpy as np


_TRACER_TYPE_MAP = {
    "weak_lensing": "_make_weak_lensing",
    "number_counts": "_make_number_counts",
    "cmb_lensing": "_make_cmb_lensing",
    "isw": "_make_isw",
    "tsz": "_make_tsz",
    "cib": "_make_cib",
}


def make_pyccl_tracer(cosmo: Any, config: dict[str, Any]) -> Any:
    """Create a pyccl Tracer from a tracer configuration dictionary.

    Parameters
    ----------
    cosmo
        A ``pyccl.Cosmology`` object.
    config
        Tracer configuration dict (e.g., from ``TracerConfig.model_dump()``).
    """
    tracer_type = config["tracer_type"]
    builder_name = _TRACER_TYPE_MAP.get(tracer_type)
    if builder_name is None:
        raise ValueError(f"Unknown tracer type: {tracer_type}")
    builder = globals()[builder_name]
    return builder(cosmo, config)


def _get_dndz(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return np.array(config["dndz_z"]), np.array(config["dndz_nz"])


def _get_tuple_or_none(
    config: dict[str, Any], z_key: str, v_key: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    z = config.get(z_key)
    v = config.get(v_key)
    if z is not None and v is not None:
        return np.array(z), np.array(v)
    return None


def _make_weak_lensing(cosmo: Any, config: dict[str, Any]) -> Any:
    import pyccl

    kwargs: dict[str, Any] = {
        "dndz": _get_dndz(config),
        "has_shear": config.get("has_shear", True),
        "use_A_ia": config.get("use_A_ia", True),
        "n_samples": config.get("n_samples", 256),
    }
    ia_bias = _get_tuple_or_none(config, "ia_bias_z", "ia_bias_az")
    if ia_bias is not None:
        kwargs["ia_bias"] = ia_bias
    return pyccl.WeakLensingTracer(cosmo, **kwargs)


def _make_number_counts(cosmo: Any, config: dict[str, Any]) -> Any:
    import pyccl

    kwargs: dict[str, Any] = {
        "dndz": _get_dndz(config),
        "has_rsd": config.get("has_rsd", False),
        "n_samples": config.get("n_samples", 256),
    }
    bias = _get_tuple_or_none(config, "bias_z", "bias_bz")
    if bias is not None:
        kwargs["bias"] = bias
    mag_bias = _get_tuple_or_none(config, "mag_bias_z", "mag_bias_sz")
    if mag_bias is not None:
        kwargs["mag_bias"] = mag_bias
    return pyccl.NumberCountsTracer(cosmo, **kwargs)


def _make_cmb_lensing(cosmo: Any, config: dict[str, Any]) -> Any:
    import pyccl

    return pyccl.CMBLensingTracer(
        cosmo,
        z_source=config.get("z_source", 1100.0),
        n_samples=config.get("n_samples", 100),
    )


def _make_isw(cosmo: Any, config: dict[str, Any]) -> Any:
    import pyccl

    return pyccl.ISWTracer(cosmo, n_samples=config.get("n_samples", 256))


def _make_tsz(cosmo: Any, config: dict[str, Any]) -> Any:
    import pyccl

    return pyccl.tSZTracer(cosmo, n_samples=config.get("n_samples", 256))


def _make_cib(cosmo: Any, config: dict[str, Any]) -> Any:
    import pyccl

    return pyccl.CIBTracer(cosmo, n_samples=config.get("n_samples", 256))
