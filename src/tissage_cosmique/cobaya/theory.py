"""Cobaya Theory provider for tissage-cosmique galaxy C(ℓ) computation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from cobaya.theory import Theory

log = logging.getLogger(__name__)


class TissageCCLTheory(Theory):
    """Cobaya Theory that computes galaxy angular power spectrum data vectors.

    Provides ``tcl_data_vector`` to downstream likelihoods. Computes C(ℓ)
    via pyccl for each tracer pair, with optional emulator hot-swap,
    provenance tracking, and validation.

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

            # Hot-swap: load emulators at init, route calls through them
            swap_config:
              angular_cl: /path/to/emulator.joblib
            swap_param_names: [Omega_c, h, sigma8]

            # Tracking: record provenance for selected evaluations
            tracking_mode: every_n   # "none", "all", "every_n"
            tracking_every_n: 100
            tracking_db_url: sqlite+aiosqlite:///mcmc_provenance.db

            # Validation: compare emulator vs pyccl periodically
            validate_every_n: 500
            validate_rtol: 0.05
    """

    tracer_configs: dict[str, dict[str, Any]]
    dv_config: dict[str, Any]
    ell: list[float]
    cosmo_param_names: list[str] = ["Omega_c", "Omega_b", "h", "n_s", "sigma8"]
    cosmo_fixed: dict[str, float] = {}

    swap_config: dict[str, str] | None = None
    swap_param_names: list[str] | None = None

    tracking_mode: str = "none"
    tracking_every_n: int = 100
    tracking_db_url: str | None = None

    validate_every_n: int = 0
    validate_rtol: float = 0.05

    _ell_array: np.ndarray
    _call_count: int
    _swap_registered: bool

    def initialize(self) -> None:
        self._ell_array = np.array(self.ell)
        self._call_count = 0
        self._swap_registered = False
        self._setup_swap()
        self._setup_tracking()

    def _setup_swap(self) -> None:
        if not self.swap_config:
            return

        from ..computations.angular_cl import angular_cl
        from ..emulators import GPEmulator
        from ..swap import register

        _fn_map = {"angular_cl": angular_cl}

        for fn_name, artifact_path in self.swap_config.items():
            fn = _fn_map.get(fn_name)
            if fn is None:
                log.warning("Unknown function for swap: %s", fn_name)
                continue

            emu = GPEmulator.load(artifact_path)
            param_names = self.swap_param_names or self.cosmo_param_names
            register(fn, emu, param_names)
            log.info("Registered emulator for %s from %s", fn_name, artifact_path)

        self._swap_registered = True

    def _setup_tracking(self) -> None:
        if self.tracking_mode == "none":
            return

        from tisserande.tracking.decorator import configure, get_backend

        if get_backend() is None and self.tracking_db_url:
            configure(db_url=self.tracking_db_url)
            log.info("Tracking configured: mode=%s, db=%s", self.tracking_mode, self.tracking_db_url)

    def _should_track(self) -> bool:
        if self.tracking_mode == "all":
            return True
        if self.tracking_mode == "every_n":
            return self._call_count % self.tracking_every_n == 0
        return False

    def _should_validate(self) -> bool:
        if self.validate_every_n <= 0:
            return False
        return self._call_count % self.validate_every_n == 0

    def get_can_provide(self) -> list[str]:
        return ["tcl_data_vector"]

    def calculate(
        self, state: dict[str, Any],
        want_derived: bool = True, **params_values: Any,  # noqa: FBT001, FBT002
    ) -> None:
        from ..computations.data_vector import compute_data_vector

        self._call_count += 1

        cosmo_params = {**self.cosmo_fixed}
        for p in self.cosmo_param_names:
            if p in params_values:
                cosmo_params[p] = params_values[p]

        if not self._should_track():
            from tisserande.tracking.decorator import get_backend, reset

            if get_backend() is not None:
                reset()

        dv = compute_data_vector(
            cosmo_params, self.tracer_configs, self.dv_config, self._ell_array,
        )
        state["tcl_data_vector"] = dv

        if self._should_validate() and self._swap_registered:
            self._validate(cosmo_params, dv)

    def _validate(self, cosmo_params: dict[str, Any], dv_emulator: np.ndarray) -> None:
        from ..computations.angular_cl import angular_cl
        from ..computations.data_vector import compute_data_vector
        from ..swap import original

        with original(angular_cl):
            dv_truth = compute_data_vector(
                cosmo_params, self.tracer_configs, self.dv_config, self._ell_array,
            )

        nonzero = np.abs(dv_truth) > 1e-30
        if np.any(nonzero):
            rel_err = np.abs(dv_emulator[nonzero] - dv_truth[nonzero]) / np.abs(dv_truth[nonzero])
            max_err = float(np.max(rel_err))
            mean_err = float(np.mean(rel_err))

            if max_err > self.validate_rtol:
                log.warning(
                    "Emulator validation at call %d: max_rel_err=%.4f (threshold=%.4f), mean=%.4f",
                    self._call_count, max_err, self.validate_rtol, mean_err,
                )
            else:
                log.debug(
                    "Emulator validation OK at call %d: max_rel_err=%.4f",
                    self._call_count, max_err,
                )

    def close(self) -> None:
        if self._swap_registered:
            from ..swap import reset as swap_reset

            swap_reset()
            self._swap_registered = False

        from tisserande.tracking.decorator import get_backend, reset as tracking_reset

        if get_backend() is not None:
            tracking_reset()

    def get_tcl_data_vector(self) -> np.ndarray:
        return self._current_state["tcl_data_vector"]
