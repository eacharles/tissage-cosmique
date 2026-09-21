"""Helper to create pyccl Tracer objects from tracer configuration dictionaries."""

from typing import Any

import numpy as np

from .nuisance import evaluate_nuisance, perturb_dndz


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

    Nuisance parameters are resolved in priority order:
    parametric model > functional (z-nodes) > static arrays > None.

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
    z = np.array(config["dndz_z"])
    nz = np.array(config["dndz_nz"])
    dndz_model = config.get("dndz_model")
    if dndz_model is not None:
        z, nz = perturb_dndz(
            z, nz,
            model=dndz_model,
            params=config.get("dndz_params"),
            basis_z=config.get("dndz_basis_z"),
            basis_amp=config.get("dndz_basis_amp"),
        )
    return z, nz


def _resolve_nuisance(
    config: dict[str, Any],
    z: np.ndarray,
    prefix: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Resolve a nuisance function using the priority chain."""
    static_v_keys = {
        "bias": "bias_bz",
        "ia_bias": "ia_bias_az",
        "mag_bias": "mag_bias_sz",
    }
    static_v_key = static_v_keys.get(prefix, f"{prefix}_v")

    return evaluate_nuisance(
        z,
        model=config.get(f"{prefix}_model"),
        params=config.get(f"{prefix}_params"),
        nodes_z=config.get(f"{prefix}_nodes_z"),
        nodes_amp=config.get(f"{prefix}_nodes_amp"),
        static_z=config.get(f"{prefix}_z"),
        static_values=config.get(static_v_key),
    )


def _make_weak_lensing(cosmo: Any, config: dict[str, Any]) -> Any:
    import pyccl

    dndz = _get_dndz(config)
    z = dndz[0]

    kwargs: dict[str, Any] = {
        "dndz": dndz,
        "has_shear": config.get("has_shear", True),
        "use_A_ia": config.get("use_A_ia", True),
        "n_samples": config.get("n_samples", 256),
    }
    ia_bias = _resolve_nuisance(config, z, "ia_bias")
    if ia_bias is not None:
        kwargs["ia_bias"] = ia_bias
    return pyccl.WeakLensingTracer(cosmo, **kwargs)


def _make_number_counts(cosmo: Any, config: dict[str, Any]) -> Any:
    import pyccl

    dndz = _get_dndz(config)
    z = dndz[0]

    kwargs: dict[str, Any] = {
        "dndz": dndz,
        "has_rsd": config.get("has_rsd", False),
        "n_samples": config.get("n_samples", 256),
    }
    bias = _resolve_nuisance(config, z, "bias")
    if bias is not None:
        kwargs["bias"] = bias
    mag_bias = _resolve_nuisance(config, z, "mag_bias")
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
