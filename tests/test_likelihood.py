"""Tests for Gaussian likelihood computation."""

import numpy as np

from tissage_cosmique.computations.likelihood import chi_squared, gaussian_log_likelihood


class TestChiSquared:

    def test_zero_residual(self):
        data = np.array([1.0, 2.0, 3.0])
        cov_inv = np.eye(3)
        assert chi_squared(data, data, cov_inv) == 0.0

    def test_unit_residual_identity_cov(self):
        data = np.array([1.0, 0.0])
        model = np.array([0.0, 0.0])
        cov_inv = np.eye(2)
        assert chi_squared(data, model, cov_inv) == 1.0

    def test_scales_with_cov(self):
        data = np.array([1.0])
        model = np.array([0.0])
        cov_inv_tight = np.array([[100.0]])
        cov_inv_loose = np.array([[1.0]])
        assert chi_squared(data, model, cov_inv_tight) > chi_squared(data, model, cov_inv_loose)

    def test_symmetric(self):
        data = np.array([1.0, 2.0])
        model = np.array([3.0, 4.0])
        cov_inv = np.array([[2.0, 0.5], [0.5, 1.0]])
        assert chi_squared(data, model, cov_inv) == chi_squared(model, data, cov_inv)


class TestGaussianLogLikelihood:

    def test_zero_residual_gives_zero(self):
        data = np.array([1.0, 2.0])
        cov_inv = np.eye(2)
        assert gaussian_log_likelihood(data, data, cov_inv) == 0.0

    def test_negative(self):
        data = np.array([1.0])
        model = np.array([0.0])
        cov_inv = np.array([[1.0]])
        assert gaussian_log_likelihood(data, model, cov_inv) < 0.0

    def test_is_minus_half_chi2(self):
        data = np.array([1.0, 2.0])
        model = np.array([0.5, 1.5])
        cov_inv = np.array([[2.0, 0.5], [0.5, 3.0]])
        chi2 = chi_squared(data, model, cov_inv)
        log_l = gaussian_log_likelihood(data, model, cov_inv)
        assert log_l == -0.5 * chi2
