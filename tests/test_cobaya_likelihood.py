"""Tests for cobaya Theory and Likelihood classes."""

import pytest

pyccl = pytest.importorskip("pyccl")
cobaya_mod = pytest.importorskip("cobaya")

import numpy as np  # noqa: E402

from tissage_cosmique.computations.covariance import compute_covariance_from_data_vector  # noqa: E402
from tissage_cosmique.computations.data_vector import compute_data_vector, make_data_vector_result  # noqa: E402
from tissage_cosmique.computations.sacc_io import save_sacc  # noqa: E402

Z = np.linspace(0.0, 2.0, 100).tolist()
NZ = np.exp(-0.5 * ((np.array(Z) - 0.5) / 0.15) ** 2).tolist()
ELL = np.arange(10, 100).astype(float).tolist()

TRACER_CONFIGS = {
    "wl": dict(
        tracer_type="weak_lensing",
        dndz_z=Z, dndz_nz=NZ, has_shear=True,
        n_gal=10.0, sigma_epsilon=0.26,
    ),
}

DV_CONFIG = {"tracer_names": ["wl"], "include_pairs": [["wl", "wl"]]}

FIDUCIAL = dict(
    Omega_c=0.2589, Omega_b=0.0486, h=0.6774,
    n_s=0.9667, sigma8=0.8159, Omega_k=0.0, w0=-1.0, wa=0.0,
)


class TestTissageCCLTheory:

    def test_standalone_calculate(self):
        from tissage_cosmique.cobaya.theory import TissageCCLTheory

        theory = TissageCCLTheory(
            {
                "tracer_configs": TRACER_CONFIGS,
                "dv_config": DV_CONFIG,
                "ell": ELL,
                "cosmo_fixed": {
                    k: v for k, v in FIDUCIAL.items()
                    if k not in ["Omega_c", "sigma8"]
                },
                "cosmo_param_names": ["Omega_c", "sigma8"],
            },
            standalone=True,
        )
        state: dict = {}
        theory.calculate(state, Omega_c=0.2589, sigma8=0.8159)
        dv = state["tcl_data_vector"]
        assert isinstance(dv, np.ndarray)
        assert len(dv) == len(ELL)
        assert np.all(dv >= 0)

    def test_get_can_provide(self):
        from tissage_cosmique.cobaya.theory import TissageCCLTheory

        theory = TissageCCLTheory(
            {
                "tracer_configs": TRACER_CONFIGS,
                "dv_config": DV_CONFIG,
                "ell": ELL,
            },
            standalone=True,
        )
        assert "tcl_data_vector" in theory.get_can_provide()


class TestTissageGaussianLikelihood:

    def test_inline_data(self):
        from tissage_cosmique.cobaya.likelihood import TissageGaussianLikelihood
        from tissage_cosmique.computations.data_vector import compute_data_vector

        ell_arr = np.array(ELL)
        dv = compute_data_vector(FIDUCIAL, TRACER_CONFIGS, DV_CONFIG, ell_arr)
        _, _, cov = compute_covariance_from_data_vector(
            FIDUCIAL, TRACER_CONFIGS, DV_CONFIG, ell_arr, f_sky=0.3,
        )

        like = TissageGaussianLikelihood(
            {
                "data_vector": dv.tolist(),
                "covariance": cov.tolist(),
            },
            standalone=True,
        )
        assert like._data.shape == dv.shape
        assert like._cov_inv.shape == cov.shape

    def test_sacc_loading(self, tmp_path):
        from tissage_cosmique.cobaya.likelihood import TissageSaccLikelihood

        ell_arr = np.array(ELL)
        dv, dv_result, cov = compute_covariance_from_data_vector(
            FIDUCIAL, TRACER_CONFIGS, DV_CONFIG, ell_arr, f_sky=0.3,
        )
        dv_result = make_data_vector_result(dv, DV_CONFIG, ell_arr)
        path = str(tmp_path / "test.sacc")
        save_sacc(path, dv_result, TRACER_CONFIGS, cov)

        like = TissageSaccLikelihood({"sacc_file": path}, standalone=True)
        assert len(like._data) == len(dv)

    def test_missing_data_raises(self):
        from tissage_cosmique.cobaya.likelihood import TissageGaussianLikelihood

        with pytest.raises(ValueError, match="Provide either"):
            TissageGaussianLikelihood({}, standalone=True)

    def test_gaussian_with_sacc_file(self, tmp_path):
        from tissage_cosmique.cobaya.likelihood import TissageGaussianLikelihood
        from tissage_cosmique.computations.data_vector import make_data_vector_result

        ell_arr = np.array(ELL)
        dv, _, cov = compute_covariance_from_data_vector(
            FIDUCIAL, TRACER_CONFIGS, DV_CONFIG, ell_arr, f_sky=0.3,
        )
        dv_result = make_data_vector_result(dv, DV_CONFIG, ell_arr)
        path = str(tmp_path / "test.sacc")
        save_sacc(path, dv_result, TRACER_CONFIGS, cov)

        like = TissageGaussianLikelihood({"sacc_file": path}, standalone=True)
        assert len(like._data) == len(ELL)

    def test_sacc_no_covariance_raises(self, tmp_path):
        from unittest.mock import patch

        from tissage_cosmique.cobaya.likelihood import TissageGaussianLikelihood

        with patch(
            "tissage_cosmique.computations.sacc_io.load_sacc",
            return_value=(None, None, None, {"wl-wl": np.ones(len(ELL))}, None),
        ):
            with pytest.raises(ValueError, match="has no covariance"):
                TissageGaussianLikelihood({"sacc_file": "fake.sacc"}, standalone=True)

    def test_get_requirements(self):
        from tissage_cosmique.cobaya.likelihood import TissageGaussianLikelihood

        ell_arr = np.array(ELL)
        dv = compute_data_vector(FIDUCIAL, TRACER_CONFIGS, DV_CONFIG, ell_arr)
        _, _, cov = compute_covariance_from_data_vector(
            FIDUCIAL, TRACER_CONFIGS, DV_CONFIG, ell_arr, f_sky=0.3,
        )
        like = TissageGaussianLikelihood(
            {"data_vector": dv.tolist(), "covariance": cov.tolist()}, standalone=True
        )
        assert like.get_requirements() == {"tcl_data_vector": None}

    def test_logp(self):
        from unittest.mock import MagicMock

        from tissage_cosmique.cobaya.likelihood import TissageGaussianLikelihood

        ell_arr = np.array(ELL)
        dv = compute_data_vector(FIDUCIAL, TRACER_CONFIGS, DV_CONFIG, ell_arr)
        _, _, cov = compute_covariance_from_data_vector(
            FIDUCIAL, TRACER_CONFIGS, DV_CONFIG, ell_arr, f_sky=0.3,
        )
        like = TissageGaussianLikelihood(
            {"data_vector": dv.tolist(), "covariance": cov.tolist()}, standalone=True
        )
        like.provider = MagicMock()
        like.provider.get_result.return_value = dv.copy()
        result = like.logp()
        assert np.isfinite(result)


class TestTissageCCLTheorySwapTracking:

    def _make_theory(self, extra_config: dict | None = None):
        from tissage_cosmique.cobaya.theory import TissageCCLTheory

        config: dict = {
            "tracer_configs": TRACER_CONFIGS,
            "dv_config": DV_CONFIG,
            "ell": ELL,
            "cosmo_fixed": {k: v for k, v in FIDUCIAL.items() if k not in ["Omega_c", "sigma8"]},
            "cosmo_param_names": ["Omega_c", "sigma8"],
        }
        if extra_config:
            config.update(extra_config)
        return TissageCCLTheory(config, standalone=True)

    def test_should_track_none(self):
        theory = self._make_theory()
        assert not theory._should_track()

    def test_should_track_all(self):
        theory = self._make_theory({"tracking_mode": "all"})
        assert theory._should_track()

    def test_should_track_every_n(self):
        theory = self._make_theory({"tracking_mode": "every_n", "tracking_every_n": 3})
        theory._call_count = 3
        assert theory._should_track()
        theory._call_count = 4
        assert not theory._should_track()

    def test_should_validate_enabled(self):
        theory = self._make_theory({"validate_every_n": 5})
        theory._call_count = 5
        assert theory._should_validate()
        theory._call_count = 4
        assert not theory._should_validate()

    def test_should_validate_disabled(self):
        theory = self._make_theory()
        theory._call_count = 100
        assert not theory._should_validate()

    def test_setup_tracking_configures_backend(self):
        from unittest.mock import patch

        with (
            patch("tisserande.tracking.decorator.get_backend", return_value=None),
            patch("tisserande.tracking.decorator.configure") as mock_configure,
        ):
            self._make_theory({
                "tracking_mode": "all",
                "tracking_db_url": "sqlite+aiosqlite:///mcmc.db",
            })
            mock_configure.assert_called_once_with(db_url="sqlite+aiosqlite:///mcmc.db")

    def test_setup_tracking_skips_if_backend_exists(self):
        from unittest.mock import MagicMock, patch

        with (
            patch("tisserande.tracking.decorator.get_backend", return_value=MagicMock()),
            patch("tisserande.tracking.decorator.configure") as mock_configure,
        ):
            self._make_theory({
                "tracking_mode": "all",
                "tracking_db_url": "sqlite+aiosqlite:///mcmc.db",
            })
            mock_configure.assert_not_called()

    def test_setup_swap_registers_emulator(self):
        from unittest.mock import MagicMock, patch

        mock_emu = MagicMock()
        with (
            patch("tissage_cosmique.emulators.GPEmulator") as mock_gp_cls,
            patch("tissage_cosmique.swap.register") as mock_register,
        ):
            mock_gp_cls.load.return_value = mock_emu
            theory = self._make_theory({
                "swap_config": {"angular_cl": "/fake/emu.joblib"},
                "swap_param_names": ["Omega_c", "sigma8"],
            })
            assert theory._swap_registered
            mock_register.assert_called_once()

    def test_setup_swap_unknown_fn_warns(self, caplog):
        import logging
        from unittest.mock import MagicMock, patch

        with (
            patch("tissage_cosmique.emulators.GPEmulator") as mock_gp_cls,
            patch("tissage_cosmique.swap.register") as mock_register,
            caplog.at_level(logging.WARNING, logger="tissage_cosmique.cobaya.theory"),
        ):
            mock_gp_cls.load.return_value = MagicMock()
            theory = self._make_theory({"swap_config": {"bad_fn": "/fake/emu.joblib"}})

        assert "Unknown function for swap" in caplog.text
        mock_register.assert_not_called()
        assert theory._swap_registered

    def test_calculate_resets_tracking_when_not_tracking(self):
        from unittest.mock import MagicMock, patch

        theory = self._make_theory()
        with (
            patch("tisserande.tracking.decorator.get_backend", return_value=MagicMock()),
            patch("tisserande.tracking.decorator.reset") as mock_reset,
        ):
            state: dict = {}
            theory.calculate(state, Omega_c=0.2589, sigma8=0.8159)
            mock_reset.assert_called_once()

    def test_validate_logs_warning_on_high_error(self, caplog):
        import logging
        from unittest.mock import patch

        theory = self._make_theory({"validate_every_n": 1, "validate_rtol": 0.01})
        theory._swap_registered = True

        dv_emulator = np.ones(len(ELL))
        dv_truth = np.ones(len(ELL)) * 1.5

        with (
            patch(
                "tissage_cosmique.computations.data_vector.compute_data_vector",
                return_value=dv_truth,
            ),
            caplog.at_level(logging.WARNING, logger="tissage_cosmique.cobaya.theory"),
        ):
            theory._validate(cosmo_params=FIDUCIAL, dv_emulator=dv_emulator)

        assert "Emulator validation at call" in caplog.text

    def test_validate_logs_debug_on_low_error(self, caplog):
        import logging
        from unittest.mock import patch

        theory = self._make_theory({"validate_every_n": 1, "validate_rtol": 1.0})
        theory._swap_registered = True

        dv = np.ones(len(ELL))

        with (
            patch(
                "tissage_cosmique.computations.data_vector.compute_data_vector",
                return_value=dv,
            ),
            caplog.at_level(logging.DEBUG, logger="tissage_cosmique.cobaya.theory"),
        ):
            theory._validate(cosmo_params=FIDUCIAL, dv_emulator=dv.copy())

        assert "validation OK" in caplog.text

    def test_close_clears_swap(self):
        from unittest.mock import MagicMock, patch

        with (
            patch("tissage_cosmique.emulators.GPEmulator") as mock_gp_cls,
            patch("tissage_cosmique.swap.register"),
        ):
            mock_gp_cls.load.return_value = MagicMock()
            theory = self._make_theory({"swap_config": {"angular_cl": "/fake/emu.joblib"}})

        assert theory._swap_registered
        theory.close()
        assert not theory._swap_registered

    def test_close_clears_tracking(self):
        from unittest.mock import MagicMock, patch

        theory = self._make_theory()
        with (
            patch("tisserande.tracking.decorator.get_backend", return_value=MagicMock()),
            patch("tisserande.tracking.decorator.reset") as mock_reset,
        ):
            theory.close()
            mock_reset.assert_called_once()

    def test_calculate_triggers_validate(self):
        theory = self._make_theory({"validate_every_n": 1})
        theory._swap_registered = True
        state: dict = {}
        theory.calculate(state, Omega_c=0.2589, sigma8=0.8159)
        assert "tcl_data_vector" in state

    def test_get_tcl_data_vector(self):
        theory = self._make_theory()
        dv = np.ones(len(ELL))
        theory._current_state = {"tcl_data_vector": dv}
        result = theory.get_tcl_data_vector()
        assert np.array_equal(result, dv)
