"""Data vector computation: concatenated C(ℓ) spectra for all tracer pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from tisserande.tracking.annotations import ArrayArg, ConfigDict
from tisserande.tracking.decorator import track

from .cosmology import make_pyccl_cosmology
from .tracers import make_pyccl_tracer


def get_data_vector_pairs(
    tracer_names: list[str],
    include_pairs: list[list[str]] | None = None,
) -> list[tuple[str, str]]:
    """Compute the ordered list of tracer pairs for a data vector.

    Parameters
    ----------
    tracer_names
        Ordered list of tracer names.
    include_pairs
        Explicit pairs ``[[t1, t2], ...]``. If None, returns all unique
        upper-triangle pairs (including auto-spectra).

    Returns
    -------
    list[tuple[str, str]]
        Ordered list of ``(tracer_i, tracer_j)`` pairs.
    """
    if include_pairs is not None:
        return [(p[0], p[1]) for p in include_pairs]

    pairs = []
    for i, t1 in enumerate(tracer_names):
        for t2 in tracer_names[i:]:
            pairs.append((t1, t2))
    return pairs


@dataclass
class DataVectorResult:
    """Container for a computed data vector with metadata."""

    data_vector: np.ndarray
    pairs: list[tuple[str, str]]
    ell: np.ndarray
    n_ell: int

    def get_cl(self, tracer1: str, tracer2: str) -> np.ndarray:
        """Extract one C(ℓ) spectrum from the data vector."""
        for i, (t1, t2) in enumerate(self.pairs):
            if (t1 == tracer1 and t2 == tracer2) or (t1 == tracer2 and t2 == tracer1):
                start = i * self.n_ell
                return self.data_vector[start : start + self.n_ell]
        raise KeyError(f"Pair ({tracer1}, {tracer2}) not in data vector")

    def reshape_matrix(self) -> np.ndarray:
        """Reshape to (n_pairs, n_ell) matrix."""
        return self.data_vector.reshape(len(self.pairs), self.n_ell)


@track
def compute_data_vector(
    cosmo_params: ConfigDict[dict[str, Any]],
    tracer_configs: ConfigDict[dict[str, Any]],
    dv_config: ConfigDict[dict[str, Any]],
    ell: ArrayArg[np.ndarray],
) -> np.ndarray:
    """Compute a concatenated data vector of angular C(ℓ) spectra.

    Parameters
    ----------
    cosmo_params
        Cosmological parameters dict.
    tracer_configs
        ``{tracer_name: tracer_config_dict}`` for all tracers.
    dv_config
        Data vector configuration with ``tracer_names`` and
        optional ``include_pairs``.
    ell
        Shared multipole array.

    Returns
    -------
    np.ndarray
        Flat data vector of shape ``(n_pairs * n_ell,)``.
    """
    import pyccl

    cosmo = make_pyccl_cosmology(cosmo_params)

    tracers = {}
    for name in dv_config["tracer_names"]:
        tracers[name] = make_pyccl_tracer(cosmo, tracer_configs[name])

    pairs = get_data_vector_pairs(
        dv_config["tracer_names"],
        dv_config.get("include_pairs"),
    )

    spectra = []
    for t1_name, t2_name in pairs:
        cl = pyccl.angular_cl(cosmo, tracers[t1_name], tracers[t2_name], ell)
        spectra.append(cl)

    return np.concatenate(spectra)


def make_data_vector_result(
    data_vector: np.ndarray,
    dv_config: dict[str, Any],
    ell: np.ndarray,
) -> DataVectorResult:
    """Wrap a computed data vector with metadata for easy access."""
    pairs = get_data_vector_pairs(
        dv_config["tracer_names"],
        dv_config.get("include_pairs"),
    )
    return DataVectorResult(
        data_vector=data_vector,
        pairs=pairs,
        ell=ell,
        n_ell=len(ell),
    )
