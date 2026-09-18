"""Tests for emulator inversion — recovering input parameters from target outputs."""

import pytest

pyccl = pytest.importorskip("pyccl")

import numpy as np  # noqa: E402

from tissage_cosmique.computations.distances import comoving_angular_distance  # noqa: E402
from tissage_cosmique.emulators import (  # noqa: E402
    GPEmulator,
    InversionResult,
    build_training_data,
    invert_minimize,
)

PARAM_NAMES = ["Omega_c", "h", "sigma8"]
A_GRID = np.linspace(0.2, 0.8, 10)
FIXED_PARAMS = dict(Omega_b=0.0486, n_s=0.9667, Omega_k=0.0, w0=-1.0, wa=0.0)

PLANCK = dict(Omega_c=0.2589, h=0.6774, sigma8=0.8159, **FIXED_PARAMS)
BOUNDS = {
    "Omega_c": (0.20, 0.35),
    "h": (0.60, 0.80),
    "sigma8": (0.70, 0.90),
}


def _make_param_samples(n: int, rng: np.random.Generator) -> list[dict]:
    return [
        {"Omega_c": rng.uniform(0.22, 0.32), "h": rng.uniform(0.62, 0.75),
         "sigma8": rng.uniform(0.77, 0.87), **FIXED_PARAMS}
        for _ in range(n)
    ]


@pytest.fixture(scope="module")
def gp_emulator():
    rng = np.random.default_rng(42)
    samples = _make_param_samples(40, rng)
    X, y = build_training_data(comoving_angular_distance, samples, A_GRID, param_names=PARAM_NAMES)
    emu = GPEmulator(feature_names=PARAM_NAMES + ["a"])
    emu.fit(X, y)
    return emu


class TestInvertMinimize:

    def test_recover_single_param(self, gp_emulator):
        y_target = comoving_angular_distance(PLANCK, A_GRID)
        result = invert_minimize(
            gp_emulator, y_target,
            free_params=["Omega_c"],
            fixed_params={"h": 0.6774, "sigma8": 0.8159, "a": A_GRID},
            x0={"Omega_c": 0.27},
            bounds={"Omega_c": (0.20, 0.35)},
        )
        assert isinstance(result, InversionResult)
        assert result.success
        assert abs(result.x_solution["Omega_c"] - 0.2589) < 0.01

    def test_recover_two_params(self, gp_emulator):
        y_target = comoving_angular_distance(PLANCK, A_GRID)
        result = invert_minimize(
            gp_emulator, y_target,
            free_params=["Omega_c", "h"],
            fixed_params={"sigma8": 0.8159, "a": A_GRID},
            x0={"Omega_c": 0.27, "h": 0.68},
            bounds={"Omega_c": (0.20, 0.35), "h": (0.60, 0.80)},
        )
        assert result.success
        assert abs(result.x_solution["Omega_c"] - 0.2589) < 0.02
        assert abs(result.x_solution["h"] - 0.6774) < 0.02

    def test_returns_inversion_result(self, gp_emulator):
        y_target = comoving_angular_distance(PLANCK, A_GRID)
        result = gp_emulator.invert(
            y_target,
            free_params=["Omega_c"],
            fixed_params={"h": 0.6774, "sigma8": 0.8159, "a": A_GRID},
            x0={"Omega_c": 0.27},
            bounds={"Omega_c": (0.20, 0.35)},
        )
        assert isinstance(result, InversionResult)
        assert "Omega_c" in result.x_solution
        assert result.y_predicted.shape == y_target.shape
        np.testing.assert_array_equal(result.y_target, y_target)


class TestGPInversion:

    def test_uncertainty_weighted(self, gp_emulator):
        y_target = comoving_angular_distance(PLANCK, A_GRID)
        result = gp_emulator.invert(
            y_target,
            free_params=["Omega_c"],
            fixed_params={"h": 0.6774, "sigma8": 0.8159, "a": A_GRID},
            x0={"Omega_c": 0.27},
            bounds={"Omega_c": (0.20, 0.35)},
        )
        assert isinstance(result, InversionResult)
        assert "Omega_c" in result.x_solution
        assert result.y_predicted.shape == y_target.shape
        assert result.residual < 0.1


class TestPyTorchInversion:

    def test_gradient_based(self):
        pytest.importorskip("torch")
        from tissage_cosmique.emulators import PyTorchEmulator

        rng = np.random.default_rng(42)
        samples = _make_param_samples(40, rng)
        X, y = build_training_data(
            comoving_angular_distance, samples, A_GRID, param_names=PARAM_NAMES,
        )
        emu = PyTorchEmulator(
            feature_names=PARAM_NAMES + ["a"], hidden_layers=[64, 64], n_epochs=300, seed=42,
        )
        emu.fit(X, y)

        y_target = comoving_angular_distance(PLANCK, A_GRID)
        result = emu.invert(
            y_target,
            free_params=["Omega_c"],
            fixed_params={"h": 0.6774, "sigma8": 0.8159, "a": A_GRID},
            x0={"Omega_c": 0.27},
            bounds={"Omega_c": (0.20, 0.35)},
            n_steps=500,
        )
        assert isinstance(result, InversionResult)
        assert abs(result.x_solution["Omega_c"] - 0.2589) < 0.05


class TestTensorFlowInversion:

    def test_gradient_based(self):
        pytest.importorskip("tensorflow")
        from tissage_cosmique.emulators import TensorFlowEmulator

        rng = np.random.default_rng(42)
        samples = _make_param_samples(40, rng)
        X, y = build_training_data(
            comoving_angular_distance, samples, A_GRID, param_names=PARAM_NAMES,
        )
        emu = TensorFlowEmulator(
            feature_names=PARAM_NAMES + ["a"], hidden_layers=[64, 64], n_epochs=300, seed=42,
        )
        emu.fit(X, y)

        y_target = comoving_angular_distance(PLANCK, A_GRID)
        result = emu.invert(
            y_target,
            free_params=["Omega_c"],
            fixed_params={"h": 0.6774, "sigma8": 0.8159, "a": A_GRID},
            x0={"Omega_c": 0.27},
            bounds={"Omega_c": (0.20, 0.35)},
            n_steps=500,
        )
        assert isinstance(result, InversionResult)
        assert abs(result.x_solution["Omega_c"] - 0.2589) < 0.05
