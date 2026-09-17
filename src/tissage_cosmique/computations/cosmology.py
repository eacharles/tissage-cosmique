"""Helper to create pyccl Cosmology objects from parameter dictionaries."""

from typing import Any

_PYCCL_KEYS = {"Omega_c", "Omega_b", "h", "n_s", "sigma8", "A_s", "Omega_k", "w0", "wa"}


def make_pyccl_cosmology(params: dict[str, Any]) -> Any:
    """Create a pyccl.Cosmology from a parameter dictionary.

    Strips non-pyccl keys (e.g., ``name``, ``id_``) and drops None values
    so that sigma8/A_s mutual exclusivity is respected.
    """
    import pyccl

    ccl_kwargs = {k: v for k, v in params.items() if k in _PYCCL_KEYS and v is not None}
    return pyccl.Cosmology(**ccl_kwargs)
