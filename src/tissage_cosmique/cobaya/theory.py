"""Cobaya Theory provider for tissage-cosmique galaxy C(ℓ) computation."""

from __future__ import annotations

from typing import Any

import numpy as np
from cobaya.theory import Theory


class TissageCCLTheory(Theory):
    """Cobaya Theory that computes galaxy angular power spectrum data vectors.

    Provides ``tcl_data_vector`` to downstream likelihoods. Computes C(ℓ)
    via pyccl for each tracer pair, concatenated into the standard data
    vector format.

    Configuration (in cobaya YAML or dict)::

        theory:
          tissage_cosmique.cobaya.TissageCCLTheory:
            tracer_configs:
              wl: {tracer_type: weak_lensing, dndz_z: [...], ...}
              nc: {tracer_type: number_counts, ...}
            dv_config:
              tracer_names: [wl, nc]
            ell: [10, 11, ..., 999]
            cosmo_param_names: [Omega_c, Omega_b, h, n_s, sigma8]
            cosmo_fixed: {Omega_k: 0.0, w0: -1.0, wa: 0.0}
    """

    tracer_configs: dict[str, dict[str, Any]]
    dv_config: dict[str, Any]
    ell: list[float]
    cosmo_param_names: list[str] = ["Omega_c", "Omega_b", "h", "n_s", "sigma8"]
    cosmo_fixed: dict[str, float] = {}

    _ell_array: np.ndarray

    def initialize(self) -> None:
        self._ell_array = np.array(self.ell)

    def get_can_provide(self) -> list[str]:
        return ["tcl_data_vector"]

    def calculate(
        self, state: dict[str, Any],
        want_derived: bool = True, **params_values: Any,  # noqa: FBT001, FBT002
    ) -> None:
        from ..computations.data_vector import compute_data_vector

        cosmo_params = {**self.cosmo_fixed}
        for p in self.cosmo_param_names:
            if p in params_values:
                cosmo_params[p] = params_values[p]

        dv = compute_data_vector(cosmo_params, self.tracer_configs, self.dv_config, self._ell_array)
        state["tcl_data_vector"] = dv

    def get_tcl_data_vector(self) -> np.ndarray:
        return self._current_state["tcl_data_vector"]
