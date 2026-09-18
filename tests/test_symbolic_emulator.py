"""Tests for the PySR symbolic regression emulator backend."""

import pytest

pysr_mod = pytest.importorskip("pysr")
pyccl = pytest.importorskip("pyccl")

import numpy as np  # noqa: E402

from tissage_cosmique.computations.distances import comoving_angular_distance  # noqa: E402
from tissage_cosmique.emulators import SymbolicEmulator, build_training_data, params_to_feature_matrix  # noqa: E402

PARAM_NAMES = ["Omega_c", "h", "sigma8"]
A_GRID = np.linspace(0.2, 0.8, 10)
FIXED_PARAMS = dict(Omega_b=0.0486, n_s=0.9667, Omega_k=0.0, w0=-1.0, wa=0.0)


def _make_param_samples(n: int, rng: np.random.Generator) -> list[dict]:
    return [
        {"Omega_c": rng.uniform(0.22, 0.32), "h": rng.uniform(0.62, 0.75),
         "sigma8": rng.uniform(0.77, 0.87), **FIXED_PARAMS}
        for _ in range(n)
    ]


@pytest.fixture(scope="module")
def training_xy():
    rng = np.random.default_rng(42)
    samples = _make_param_samples(30, rng)
    return build_training_data(comoving_angular_distance, samples, A_GRID, param_names=PARAM_NAMES)


@pytest.fixture(scope="module")
def fitted_emulator(training_xy):
    X, y = training_xy
    emu = SymbolicEmulator(feature_names=PARAM_NAMES + ["a"], niterations=5, maxsize=15)
    emu.fit(X, y)
    return emu


class TestSymbolicEmulator:

    def test_fit_sets_state(self, fitted_emulator):
        assert fitted_emulator.is_fitted
        assert fitted_emulator.n_training_samples == 30 * len(A_GRID)

    def test_predict_shape(self, fitted_emulator):
        params = {"Omega_c": 0.27, "h": 0.68, "sigma8": 0.81}
        X = params_to_feature_matrix(params, A_GRID, param_names=PARAM_NAMES)
        pred = fitted_emulator.predict(X)
        assert pred.shape == (len(A_GRID),)

    def test_metadata_has_expression(self, fitted_emulator):
        meta = fitted_emulator.metadata
        assert meta["type"] == "symbolic"
        assert meta["is_fitted"] is True
        assert "best_expression" in meta
        assert isinstance(meta["best_expression"], str)
        assert len(meta["best_expression"]) > 0

    def test_save_load_roundtrip(self, fitted_emulator, tmp_path):
        path = tmp_path / "symbolic_emulator.joblib"
        fitted_emulator.save(path)
        loaded = SymbolicEmulator.load(path)
        assert loaded.is_fitted
        assert loaded.feature_names == fitted_emulator.feature_names
        assert loaded.metadata["best_expression"] == fitted_emulator.metadata["best_expression"]

        params = {"Omega_c": 0.27, "h": 0.68, "sigma8": 0.81}
        X = params_to_feature_matrix(params, A_GRID, param_names=PARAM_NAMES)
        np.testing.assert_allclose(
            fitted_emulator.predict(X),
            loaded.predict(X),
            rtol=1e-10,
        )
