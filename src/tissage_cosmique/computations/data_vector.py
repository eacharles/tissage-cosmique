"""Data vector computation: concatenated C(ℓ) spectra for all tracer pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from tisserande.tracking.annotations import ArrayArg, ConfigDict
from tisserande.tracking.decorator import track

from .cosmology import make_pyccl_cosmology
from .ell_binning import bin_cl, pair_key
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
    """
    if include_pairs is not None:
        return [(p[0], p[1]) for p in include_pairs]

    pairs = []
    for i, t1 in enumerate(tracer_names):
        for t2 in tracer_names[i:]:
            pairs.append((t1, t2))
    return pairs


def _resolve_ell_for_pair(
    t1: str, t2: str,
    dv_config: dict[str, Any],
    shared_ell: np.ndarray,
) -> np.ndarray:
    """Get the ℓ array for a specific pair, falling back to shared."""
    ell_per_pair = dv_config.get("ell_per_pair")
    if ell_per_pair is not None:
        key = pair_key(t1, t2)
        if key in ell_per_pair:
            return np.asarray(ell_per_pair[key])
        key_rev = pair_key(t2, t1)
        if key_rev in ell_per_pair:
            return np.asarray(ell_per_pair[key_rev])
    return shared_ell


def _resolve_bin_edges_for_pair(
    t1: str, t2: str,
    dv_config: dict[str, Any],
) -> np.ndarray | None:
    """Get bin edges for a specific pair, falling back to shared."""
    per_pair = dv_config.get("bin_edges_per_pair")
    if per_pair is not None:
        key = pair_key(t1, t2)
        if key in per_pair:
            return np.asarray(per_pair[key])
        key_rev = pair_key(t2, t1)
        if key_rev in per_pair:
            return np.asarray(per_pair[key_rev])
    shared = dv_config.get("bin_edges")
    if shared is not None:
        return np.asarray(shared)
    return None


@dataclass
class DataVectorResult:
    """Container for a computed data vector with per-pair metadata."""

    data_vector: np.ndarray
    pairs: list[tuple[str, str]]
    ell_per_pair: dict[str, np.ndarray]
    offsets: list[int]
    lengths: list[int]

    def get_cl(self, tracer1: str, tracer2: str) -> np.ndarray:
        """Extract one C(ℓ) spectrum from the data vector."""
        for i, (t1, t2) in enumerate(self.pairs):
            if (t1 == tracer1 and t2 == tracer2) or (t1 == tracer2 and t2 == tracer1):
                start = self.offsets[i]
                return self.data_vector[start : start + self.lengths[i]]
        raise KeyError(f"Pair ({tracer1}, {tracer2}) not in data vector")

    def get_ell(self, tracer1: str, tracer2: str) -> np.ndarray:
        """Get the effective ℓ array for a pair."""
        key = pair_key(tracer1, tracer2)
        if key in self.ell_per_pair:
            return self.ell_per_pair[key]
        key_rev = pair_key(tracer2, tracer1)
        if key_rev in self.ell_per_pair:
            return self.ell_per_pair[key_rev]
        raise KeyError(f"Pair ({tracer1}, {tracer2}) not in data vector")

    def get_all_cl(self) -> dict[str, np.ndarray]:
        """Return all spectra as a dict keyed by pair key."""
        result = {}
        for i, (t1, t2) in enumerate(self.pairs):
            key = pair_key(t1, t2)
            start = self.offsets[i]
            result[key] = self.data_vector[start : start + self.lengths[i]]
        return result


@track
def compute_data_vector(
    cosmo_params: ConfigDict[dict[str, Any]],
    tracer_configs: ConfigDict[dict[str, Any]],
    dv_config: ConfigDict[dict[str, Any]],
    ell: ArrayArg[np.ndarray],
) -> np.ndarray:
    """Compute a concatenated data vector of angular C(ℓ) spectra.

    Supports per-pair ℓ ranges and ℓ binning. The shared ``ell``
    argument is the default; ``dv_config["ell_per_pair"]`` overrides
    per pair. If ``bin_edges`` or ``bin_edges_per_pair`` is set,
    spectra are binned into bandpowers.
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

    bin_weights = dv_config.get("bin_weights", "2ell+1")
    segments = []

    for t1_name, t2_name in pairs:
        pair_ell = _resolve_ell_for_pair(t1_name, t2_name, dv_config, ell)
        cl = pyccl.angular_cl(cosmo, tracers[t1_name], tracers[t2_name], pair_ell)

        edges = _resolve_bin_edges_for_pair(t1_name, t2_name, dv_config)
        if edges is not None:
            _, cl = bin_cl(pair_ell, cl, edges, weights=bin_weights)

        segments.append(cl)

    return np.concatenate(segments)


def make_data_vector_result(
    data_vector: np.ndarray,
    dv_config: dict[str, Any],
    ell: np.ndarray,
) -> DataVectorResult:
    """Wrap a computed data vector with per-pair metadata."""
    pairs = get_data_vector_pairs(
        dv_config["tracer_names"],
        dv_config.get("include_pairs"),
    )

    bin_weights = dv_config.get("bin_weights", "2ell+1")
    ell_dict: dict[str, np.ndarray] = {}
    offsets: list[int] = []
    lengths: list[int] = []
    offset = 0

    for t1, t2 in pairs:
        pair_ell = _resolve_ell_for_pair(t1, t2, dv_config, ell)
        edges = _resolve_bin_edges_for_pair(t1, t2, dv_config)

        if edges is not None:
            n = len(edges) - 1
            ell_eff, _ = bin_cl(
                pair_ell, np.zeros_like(pair_ell), edges, weights=bin_weights,
            )
            ell_dict[pair_key(t1, t2)] = ell_eff
        else:
            n = len(pair_ell)
            ell_dict[pair_key(t1, t2)] = pair_ell

        offsets.append(offset)
        lengths.append(n)
        offset += n

    return DataVectorResult(
        data_vector=data_vector,
        pairs=pairs,
        ell_per_pair=ell_dict,
        offsets=offsets,
        lengths=lengths,
    )
