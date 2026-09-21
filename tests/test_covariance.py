"""Tests for analytic Gaussian covariance matrix."""

import pytest

pyccl = pytest.importorskip("pyccl")

import numpy as np  # noqa: E402

from tissage_cosmique.computations.covariance import (  # noqa: E402
    compute_covariance_from_data_vector,
    noise_power,
)

PLANCK = dict(
    Omega_c=0.2589, Omega_b=0.0486, h=0.6774,
    n_s=0.9667, sigma8=0.8159, Omega_k=0.0, w0=-1.0, wa=0.0,
)

Z = np.linspace(0.0, 2.0, 100).tolist()
NZ = np.exp(-0.5 * ((np.array(Z) - 0.5) / 0.15) ** 2).tolist()
ELL = np.arange(10, 200).astype(float)

WL_CONFIG = dict(
    tracer_type="weak_lensing",
    dndz_z=Z, dndz_nz=NZ, has_shear=True,
    n_gal=10.0, sigma_epsilon=0.26,
)

NC_CONFIG = dict(
    tracer_type="number_counts",
    dndz_z=Z, dndz_nz=NZ,
    bias_model="constant", bias_params={"a0": 1.5},
    has_rsd=False,
    n_gal=30.0,
)

CMB_CONFIG = dict(
    tracer_type="cmb_lensing", z_source=1100.0,
)


class TestNoisePower:

    def test_wl_noise(self):
        n_ell = noise_power(WL_CONFIG, ELL)
        assert n_ell.shape == ELL.shape
        assert np.all(n_ell > 0)
        assert len(set(n_ell)) == 1

    def test_nc_noise(self):
        n_ell = noise_power(NC_CONFIG, ELL)
        assert np.all(n_ell > 0)
        assert n_ell[0] < noise_power(
            {**NC_CONFIG, "n_gal": 1.0}, ELL,
        )[0]

    def test_cmb_noise_zero(self):
        n_ell = noise_power(CMB_CONFIG, ELL)
        assert np.all(n_ell == 0)

    def test_no_ngal_gives_zero(self):
        config = {**WL_CONFIG}
        del config["n_gal"]
        n_ell = noise_power(config, ELL)
        assert np.all(n_ell == 0)


class TestGaussianCovariance:

    def test_symmetric(self):
        tracer_configs = {"wl": WL_CONFIG, "nc": NC_CONFIG}
        dv_config = {"tracer_names": ["wl", "nc"]}
        _, _, cov = compute_covariance_from_data_vector(
            PLANCK, tracer_configs, dv_config, ELL, f_sky=0.3,
        )
        np.testing.assert_allclose(cov, cov.T, atol=1e-30)

    def test_positive_diagonal(self):
        tracer_configs = {"wl": WL_CONFIG, "nc": NC_CONFIG}
        dv_config = {"tracer_names": ["wl", "nc"]}
        _, _, cov = compute_covariance_from_data_vector(
            PLANCK, tracer_configs, dv_config, ELL, f_sky=0.3,
        )
        assert np.all(np.diag(cov) > 0)

    def test_shape(self):
        tracer_configs = {"wl": WL_CONFIG, "nc": NC_CONFIG}
        dv_config = {"tracer_names": ["wl", "nc"]}
        dv, _, cov = compute_covariance_from_data_vector(
            PLANCK, tracer_configs, dv_config, ELL, f_sky=0.3,
        )
        n = len(dv)
        assert cov.shape == (n, n)

    def test_positive_semidefinite(self):
        tracer_configs = {"wl": WL_CONFIG}
        dv_config = {
            "tracer_names": ["wl"],
            "include_pairs": [["wl", "wl"]],
        }
        _, _, cov = compute_covariance_from_data_vector(
            PLANCK, tracer_configs, dv_config, ELL, f_sky=0.3,
        )
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-15)

    def test_larger_fsky_smaller_variance(self):
        tracer_configs = {"wl": WL_CONFIG}
        dv_config = {
            "tracer_names": ["wl"],
            "include_pairs": [["wl", "wl"]],
        }
        _, _, cov_small = compute_covariance_from_data_vector(
            PLANCK, tracer_configs, dv_config, ELL, f_sky=0.1,
        )
        _, _, cov_large = compute_covariance_from_data_vector(
            PLANCK, tracer_configs, dv_config, ELL, f_sky=0.5,
        )
        assert np.all(np.diag(cov_large) < np.diag(cov_small))

    def test_more_noise_larger_variance(self):
        noisy = {**WL_CONFIG, "n_gal": 1.0}
        quiet = {**WL_CONFIG, "n_gal": 100.0}
        dv_config = {
            "tracer_names": ["wl"],
            "include_pairs": [["wl", "wl"]],
        }
        _, _, cov_noisy = compute_covariance_from_data_vector(
            PLANCK, {"wl": noisy}, dv_config, ELL, f_sky=0.3,
        )
        _, _, cov_quiet = compute_covariance_from_data_vector(
            PLANCK, {"wl": quiet}, dv_config, ELL, f_sky=0.3,
        )
        assert np.all(np.diag(cov_noisy) > np.diag(cov_quiet))

    def test_snr_makes_sense(self):
        tracer_configs = {"wl": WL_CONFIG}
        dv_config = {
            "tracer_names": ["wl"],
            "include_pairs": [["wl", "wl"]],
        }
        dv, _, cov = compute_covariance_from_data_vector(
            PLANCK, tracer_configs, dv_config, ELL, f_sky=0.3,
        )
        snr = dv / np.sqrt(np.diag(cov))
        assert np.all(np.isfinite(snr))
        assert np.any(snr > 1)
