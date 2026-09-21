"""Tests for Fisher matrix forecast."""

import numpy as np
import pytest

from tissage_cosmique.computations.fisher import compute_fisher_matrix


def _linear_model(params: dict) -> np.ndarray:
    """Simple model: y = a * x + b, evaluated at x = [1, 2, 3, 4, 5]."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    return params["a"] * x + params["b"]


class TestComputeFisherMatrix:

    def test_shape(self):
        cov = np.eye(5) * 0.01
        cov_inv = np.linalg.inv(cov)
        result = compute_fisher_matrix(
            _linear_model,
            {"a": 1.0, "b": 0.0},
            cov_inv,
            ["a", "b"],
        )
        assert result.fisher_matrix.shape == (2, 2)

    def test_symmetric(self):
        cov_inv = np.eye(5) * 100.0
        result = compute_fisher_matrix(
            _linear_model,
            {"a": 1.0, "b": 0.0},
            cov_inv,
            ["a", "b"],
        )
        np.testing.assert_allclose(
            result.fisher_matrix, result.fisher_matrix.T, atol=1e-10,
        )

    def test_positive_diagonal(self):
        cov_inv = np.eye(5) * 100.0
        result = compute_fisher_matrix(
            _linear_model,
            {"a": 1.0, "b": 0.0},
            cov_inv,
            ["a", "b"],
        )
        assert np.all(np.diag(result.fisher_matrix) > 0)

    def test_derivatives_stored(self):
        cov_inv = np.eye(5)
        result = compute_fisher_matrix(
            _linear_model,
            {"a": 1.0, "b": 0.0},
            cov_inv,
            ["a", "b"],
        )
        assert "a" in result.derivatives
        assert "b" in result.derivatives
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_allclose(result.derivatives["a"], x, rtol=0.01)
        np.testing.assert_allclose(result.derivatives["b"], np.ones(5), rtol=0.01)


class TestFisherResult:

    @pytest.fixture()
    def result(self):
        cov_inv = np.eye(5) * 100.0
        return compute_fisher_matrix(
            _linear_model,
            {"a": 2.0, "b": 1.0},
            cov_inv,
            ["a", "b"],
        )

    def test_sigma(self, result):
        s_a = result.sigma("a")
        s_b = result.sigma("b")
        assert s_a > 0
        assert s_b > 0

    def test_sigma_conditional(self, result):
        s_cond = result.sigma_conditional("a")
        s_marg = result.sigma("a")
        assert s_cond <= s_marg

    def test_correlation_matrix(self, result):
        corr = result.correlation_matrix()
        assert corr.shape == (2, 2)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-10)
        assert np.all(np.abs(corr) <= 1.0 + 1e-10)

    def test_ellipse(self, result):
        x, y = result.ellipse("a", "b", n_sigma=1.0)
        assert len(x) == 100
        assert len(y) == 100
        assert np.mean(x) == pytest.approx(2.0, abs=0.1)
        assert np.mean(y) == pytest.approx(1.0, abs=0.1)

    def test_tighter_cov_smaller_sigma(self):
        cov_tight = np.eye(5) * 10000.0
        cov_loose = np.eye(5) * 1.0
        r_tight = compute_fisher_matrix(
            _linear_model, {"a": 1.0, "b": 0.0}, cov_tight, ["a", "b"],
        )
        r_loose = compute_fisher_matrix(
            _linear_model, {"a": 1.0, "b": 0.0}, cov_loose, ["a", "b"],
        )
        assert r_tight.sigma("a") < r_loose.sigma("a")


class TestFisherWithPyccl:

    def test_cosmological_fisher(self):
        pytest.importorskip("pyccl")
        from tissage_cosmique.computations.covariance import (
            compute_covariance_from_data_vector,
        )

        z = np.linspace(0.0, 2.0, 100).tolist()
        nz = np.exp(-0.5 * ((np.array(z) - 0.5) / 0.15) ** 2).tolist()
        tracer_configs = {
            "wl": dict(
                tracer_type="weak_lensing",
                dndz_z=z, dndz_nz=nz, has_shear=True,
                n_gal=10.0, sigma_epsilon=0.26,
            ),
        }
        dv_config = {
            "tracer_names": ["wl"],
            "include_pairs": [["wl", "wl"]],
        }
        ell = np.arange(10, 200).astype(float)

        fiducial = dict(
            Omega_c=0.2589, Omega_b=0.0486, h=0.6774,
            n_s=0.9667, sigma8=0.8159, Omega_k=0.0, w0=-1.0, wa=0.0,
        )

        _, _, cov = compute_covariance_from_data_vector(
            fiducial, tracer_configs, dv_config, ell, f_sky=0.3,
        )
        cov_inv = np.linalg.inv(cov)

        from tissage_cosmique.computations.data_vector import compute_data_vector

        def model_fn(params: dict) -> np.ndarray:
            full = {**fiducial, **params}
            return compute_data_vector(full, tracer_configs, dv_config, ell)

        result = compute_fisher_matrix(
            model_fn, fiducial, cov_inv,
            param_names=["Omega_c", "sigma8"],
            step_sizes={"Omega_c": 0.005, "sigma8": 0.005},
        )

        assert result.fisher_matrix.shape == (2, 2)
        assert result.sigma("Omega_c") > 0
        assert result.sigma("sigma8") > 0
        assert result.sigma("Omega_c") < 1.0
