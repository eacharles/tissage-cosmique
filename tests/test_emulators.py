"""Tests for the emulator protocol, GP backend, and training utilities."""

import pytest

pyccl = pytest.importorskip("pyccl")

import numpy as np  # noqa: E402

from tissage_cosmique.computations.distances import comoving_angular_distance  # noqa: E402
from tissage_cosmique.emulators import GPEmulator, build_training_data, params_to_feature_matrix  # noqa: E402

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
def training_data():
    rng = np.random.default_rng(42)
    samples = _make_param_samples(30, rng)
    X, y = build_training_data(comoving_angular_distance, samples, A_GRID, PARAM_NAMES)
    return X, y, samples


@pytest.fixture()
def fitted_emulator(training_data):
    X, y, _ = training_data
    emu = GPEmulator(feature_names=PARAM_NAMES + ["a"])
    emu.fit(X, y)
    return emu


class TestBuildTrainingData:

    def test_shape(self, training_data):
        X, y, samples = training_data
        expected_rows = len(samples) * len(A_GRID)
        assert X.shape == (expected_rows, len(PARAM_NAMES) + 1)
        assert y.shape == (expected_rows,)

    def test_values_positive(self, training_data):
        _, y, _ = training_data
        assert np.all(y >= 0)


class TestParamsToFeatureMatrix:

    def test_shape(self):
        params = {"Omega_c": 0.25, "h": 0.70, "sigma8": 0.81}
        a = np.linspace(0.2, 0.8, 20)
        X = params_to_feature_matrix(params, a, PARAM_NAMES)
        assert X.shape == (20, len(PARAM_NAMES) + 1)

    def test_last_column_is_a(self):
        params = {"Omega_c": 0.25, "h": 0.70, "sigma8": 0.81}
        a = np.array([0.3, 0.5, 0.8])
        X = params_to_feature_matrix(params, a, PARAM_NAMES)
        np.testing.assert_array_equal(X[:, -1], a)

    def test_param_columns_constant(self):
        params = {"Omega_c": 0.25, "h": 0.70, "sigma8": 0.81}
        a = np.array([0.3, 0.5, 0.8])
        X = params_to_feature_matrix(params, a, PARAM_NAMES)
        for col in range(len(PARAM_NAMES)):
            assert len(set(X[:, col])) == 1


class TestGPEmulator:

    def test_fit_sets_state(self, fitted_emulator):
        assert fitted_emulator.is_fitted
        assert fitted_emulator.n_training_samples == 30 * len(A_GRID)

    def test_predict_shape(self, fitted_emulator):
        params = {"Omega_c": 0.27, "h": 0.68, "sigma8": 0.81}
        X = params_to_feature_matrix(params, A_GRID, PARAM_NAMES)
        pred = fitted_emulator.predict(X)
        assert pred.shape == (len(A_GRID),)

    def test_prediction_accuracy(self, fitted_emulator):
        rng = np.random.default_rng(123)
        test_samples = _make_param_samples(5, rng)
        for params in test_samples:
            truth = comoving_angular_distance({**params, **FIXED_PARAMS}, A_GRID)
            X = params_to_feature_matrix(params, A_GRID, PARAM_NAMES)
            pred = fitted_emulator.predict(X)
            large = truth > 100.0
            rel_err = np.abs(pred[large] - truth[large]) / truth[large]
            assert np.max(rel_err) < 0.05, f"Max relative error {np.max(rel_err):.4f} exceeds 5%"

    def test_predict_with_std(self, fitted_emulator):
        params = {"Omega_c": 0.27, "h": 0.68, "sigma8": 0.81}
        X = params_to_feature_matrix(params, A_GRID, PARAM_NAMES)
        mean, std = fitted_emulator.predict_with_std(X)
        assert mean.shape == (len(A_GRID),)
        assert std.shape == (len(A_GRID),)
        assert np.all(std >= 0)

    def test_metadata(self, fitted_emulator):
        meta = fitted_emulator.metadata
        assert meta["type"] == "gp"
        assert meta["is_fitted"] is True
        assert meta["n_training_samples"] == 30 * len(A_GRID)
        assert "kernel" in meta
        assert "training_score" in meta
        assert meta["feature_names"] == PARAM_NAMES + ["a"]

    def test_save_load_roundtrip(self, fitted_emulator, tmp_path):
        path = tmp_path / "gp_emulator.joblib"
        fitted_emulator.save(path)
        loaded = GPEmulator.load(path)
        assert loaded.is_fitted
        assert loaded.feature_names == fitted_emulator.feature_names
        assert loaded.n_training_samples == fitted_emulator.n_training_samples

        params = {"Omega_c": 0.27, "h": 0.68, "sigma8": 0.81}
        X = params_to_feature_matrix(params, A_GRID, PARAM_NAMES)
        np.testing.assert_array_equal(
            fitted_emulator.predict(X),
            loaded.predict(X),
        )

    def test_unfitted_state(self):
        emu = GPEmulator(feature_names=["a", "b"])
        assert not emu.is_fitted
        assert emu.n_training_samples == 0
        meta = emu.metadata
        assert meta["is_fitted"] is False
        assert "kernel" not in meta
