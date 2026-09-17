"""Tests for emulator validation tools."""

import pytest

pyccl = pytest.importorskip("pyccl")

import numpy as np  # noqa: E402

from tissage_cosmique.computations.distances import comoving_angular_distance  # noqa: E402
from tissage_cosmique.emulators import (  # noqa: E402
    GPEmulator,
    build_training_data,
    check_calibration,
    cross_validate,
    validate_against_computation,
    validate_emulator,
)

PARAM_NAMES = ["Omega_c", "h", "sigma8"]
A_GRID = np.linspace(0.2, 0.8, 10)
FIXED_PARAMS = dict(Omega_b=0.0486, n_s=0.9667, Omega_k=0.0, w0=-1.0, wa=0.0)


def _make_param_samples(n: int, rng: np.random.Generator) -> list[dict]:
    samples = []
    for _ in range(n):
        p = {
            "Omega_c": rng.uniform(0.22, 0.32),
            "h": rng.uniform(0.62, 0.75),
            "sigma8": rng.uniform(0.77, 0.87),
            **FIXED_PARAMS,
        }
        samples.append(p)
    return samples


@pytest.fixture()
def training_xy():
    rng = np.random.default_rng(42)
    samples = _make_param_samples(30, rng)
    X, y = build_training_data(comoving_angular_distance, samples, A_GRID, PARAM_NAMES)
    return X, y


@pytest.fixture()
def fitted_emulator(training_xy):
    X, y = training_xy
    emu = GPEmulator(feature_names=PARAM_NAMES + ["a"])
    emu.fit(X, y)
    return emu


class TestValidateEmulator:

    def test_returns_correct_types(self, fitted_emulator, training_xy):
        X, y = training_xy
        result = validate_emulator(fitted_emulator, X, y)
        assert isinstance(result.mae, float)
        assert isinstance(result.rmse, float)
        assert isinstance(result.r2_score, float)
        assert result.n_test == len(y)

    def test_residuals_shape(self, fitted_emulator, training_xy):
        X, y = training_xy
        result = validate_emulator(fitted_emulator, X, y)
        assert result.residuals.shape == (len(y),)

    def test_training_set_r2(self, fitted_emulator, training_xy):
        X, y = training_xy
        result = validate_emulator(fitted_emulator, X, y)
        assert result.r2_score > 0.99

    def test_mae_positive(self, fitted_emulator, training_xy):
        X, y = training_xy
        result = validate_emulator(fitted_emulator, X, y)
        assert result.mae >= 0
        assert result.rmse >= result.mae


class TestValidateAgainstComputation:

    def test_holdout_accuracy(self, fitted_emulator):
        rng = np.random.default_rng(999)
        test_params = _make_param_samples(5, rng)
        result = validate_against_computation(
            fitted_emulator, comoving_angular_distance, test_params, A_GRID, PARAM_NAMES,
        )
        assert result.n_test == 5 * len(A_GRID)
        assert result.r2_score > 0.9

    def test_relative_error_bounded(self, fitted_emulator):
        rng = np.random.default_rng(999)
        test_params = _make_param_samples(5, rng)
        result = validate_against_computation(
            fitted_emulator, comoving_angular_distance, test_params, A_GRID, PARAM_NAMES,
        )
        assert result.mean_relative_error < 0.05


class TestCrossValidate:

    def test_returns_correct_number_of_folds(self, training_xy):
        X, y = training_xy
        results = cross_validate(
            lambda: GPEmulator(feature_names=PARAM_NAMES + ["a"]),
            X, y, n_folds=3,
        )
        assert len(results) == 3

    def test_all_folds_have_results(self, training_xy):
        X, y = training_xy
        results = cross_validate(
            lambda: GPEmulator(feature_names=PARAM_NAMES + ["a"]),
            X, y, n_folds=3,
        )
        total_test = sum(r.n_test for r in results)
        assert total_test == len(y)
        for r in results:
            assert r.n_test > 0
            assert r.r2_score > 0.8


class TestCheckCalibration:

    def test_returns_calibration_result(self, fitted_emulator, training_xy):
        X, y = training_xy
        result = check_calibration(fitted_emulator, X, y)
        assert len(result.expected_coverage) == 3
        assert len(result.observed_coverage) == 3
        assert result.mean_std > 0

    def test_coverage_reasonable(self, fitted_emulator, training_xy):
        X, y = training_xy
        result = check_calibration(fitted_emulator, X, y)
        for exp, obs in zip(result.expected_coverage, result.observed_coverage):
            assert 0.0 <= obs <= 1.0

    def test_raises_for_no_predict_with_std(self, training_xy):
        from tissage_cosmique.emulators.base import Emulator

        class DummyEmulator(Emulator):
            def fit(self, X: np.ndarray, y: np.ndarray) -> None:
                pass

            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.zeros(len(X))

            def save(self, path: str) -> None:
                pass

            @classmethod
            def load(cls, path: str) -> "DummyEmulator":
                return cls()

            @property
            def metadata(self) -> dict:
                return {}

        X, y = training_xy
        emu = DummyEmulator()
        with pytest.raises(TypeError, match="does not support predict_with_std"):
            check_calibration(emu, X, y)
