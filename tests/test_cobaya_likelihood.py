"""Tests for cobaya Theory and Likelihood classes."""

import pytest

pyccl = pytest.importorskip("pyccl")
cobaya_mod = pytest.importorskip("cobaya")

import numpy as np  # noqa: E402

from tissage_cosmique.computations.covariance import compute_covariance_from_data_vector  # noqa: E402
from tissage_cosmique.computations.data_vector import make_data_vector_result  # noqa: E402
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
