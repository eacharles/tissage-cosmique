"""Tests for generalized multi-grid training data utilities."""

import pytest

pyccl = pytest.importorskip("pyccl")
pytest.importorskip("camb")

import numpy as np  # noqa: E402

from tissage_cosmique.computations.distances import comoving_angular_distance  # noqa: E402
from tissage_cosmique.computations.power import linear_matter_power  # noqa: E402
from tissage_cosmique.emulators import build_training_data, params_to_feature_matrix  # noqa: E402

PARAM_NAMES = ["Omega_c", "h", "sigma8"]
FIXED_PARAMS = dict(Omega_b=0.0486, n_s=0.9667, Omega_k=0.0, w0=-1.0, wa=0.0)

PLANCK = dict(Omega_c=0.2589, h=0.6774, sigma8=0.8159, **FIXED_PARAMS)


class TestSingleGrid:
    """Backward compatibility: single-grid case works as before."""

    def test_shape(self):
        a_grid = np.linspace(0.2, 0.8, 10)
        samples = [PLANCK]
        X, y = build_training_data(comoving_angular_distance, samples, a_grid, param_names=PARAM_NAMES)
        assert X.shape == (10, 4)
        assert y.shape == (10,)

    def test_feature_matrix_shape(self):
        a_grid = np.linspace(0.2, 0.8, 10)
        X = params_to_feature_matrix(PLANCK, a_grid, param_names=PARAM_NAMES)
        assert X.shape == (10, 4)


class TestTwoGrids:
    """Power spectrum: two grid variables (k, a)."""

    def test_build_training_data_shape(self):
        k_grid = np.logspace(-2, 0, 5)
        a_grid = np.array([0.5, 1.0])
        samples = [PLANCK]
        X, y = build_training_data(
            linear_matter_power, samples, k_grid, a_grid,
            param_names=PARAM_NAMES,
            result_axes_order=[1, 0],
        )
        assert X.shape == (2 * 5, 5)
        assert y.shape == (2 * 5,)

    def test_build_training_data_multiple_samples(self):
        k_grid = np.logspace(-2, 0, 4)
        a_grid = np.array([0.5, 0.8, 1.0])
        samples = [PLANCK, {**PLANCK, "Omega_c": 0.30}]
        X, y = build_training_data(
            linear_matter_power, samples, k_grid, a_grid,
            param_names=PARAM_NAMES,
            result_axes_order=[1, 0],
        )
        assert X.shape == (2 * 3 * 4, 5)
        assert y.shape == (2 * 3 * 4,)

    def test_feature_columns_correct(self):
        k_grid = np.logspace(-2, 0, 3)
        a_grid = np.array([0.5, 1.0])
        X, y = build_training_data(
            linear_matter_power, [PLANCK], k_grid, a_grid,
            param_names=PARAM_NAMES,
            result_axes_order=[1, 0],
        )
        k_col = X[:, 3]
        a_col = X[:, 4]
        assert set(np.round(k_col, 10)) == set(np.round(k_grid, 10))
        assert set(np.round(a_col, 10)) == set(np.round(a_grid, 10))

    def test_values_match_direct_computation(self):
        k_grid = np.logspace(-2, 0, 4)
        a_grid = np.array([0.5, 1.0])
        X, y = build_training_data(
            linear_matter_power, [PLANCK], k_grid, a_grid,
            param_names=PARAM_NAMES,
            result_axes_order=[1, 0],
        )
        pk_direct = linear_matter_power(PLANCK, k_grid, a_grid)
        assert np.allclose(y, pk_direct.ravel())

    def test_feature_matrix_shape(self):
        k_grid = np.logspace(-2, 0, 5)
        a_grid = np.array([0.5, 0.8, 1.0])
        X = params_to_feature_matrix(
            PLANCK, k_grid, a_grid,
            param_names=PARAM_NAMES,
            result_axes_order=[1, 0],
        )
        assert X.shape == (3 * 5, 5)

    def test_all_positive_power(self):
        k_grid = np.logspace(-2, 0, 5)
        a_grid = np.array([0.5, 1.0])
        X, y = build_training_data(
            linear_matter_power, [PLANCK], k_grid, a_grid,
            param_names=PARAM_NAMES,
            result_axes_order=[1, 0],
        )
        assert np.all(y > 0)
