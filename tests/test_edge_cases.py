"""Tests for uncovered edge cases across the emulator modules."""

import pytest

pyccl = pytest.importorskip("pyccl")
pytest.importorskip("camb")

import numpy as np  # noqa: E402

from tissage_cosmique.computations.distances import comoving_angular_distance  # noqa: E402
from tissage_cosmique.computations.power import linear_matter_power  # noqa: E402
from tissage_cosmique.emulators import GPEmulator, build_training_data  # noqa: E402
from tissage_cosmique.emulators.codecs.pca import PCACodec  # noqa: E402

PARAM_NAMES = ["Omega_c", "h", "sigma8"]
FIXED = dict(Omega_b=0.0486, n_s=0.9667, Omega_k=0.0, w0=-1.0, wa=0.0)
A_GRID = np.linspace(0.2, 0.8, 10)


def _make_samples(n: int, rng: np.random.Generator) -> list[dict]:
    return [
        {"Omega_c": rng.uniform(0.22, 0.32), "h": rng.uniform(0.62, 0.75),
         "sigma8": rng.uniform(0.77, 0.87), **FIXED}
        for _ in range(n)
    ]


class TestPCACodecUnfitted:

    def test_explained_variance_unfitted(self):
        codec = PCACodec(n_components=3)
        assert codec.explained_variance_ratio is None

    def test_cumulative_variance_unfitted(self):
        codec = PCACodec(n_components=3)
        assert codec.cumulative_variance is None


class TestLatentEmulator2DOutput:

    def test_fit_with_2d_output(self):
        from tissage_cosmique.emulators import LatentEmulator, PCACodec

        rng = np.random.default_rng(42)
        samples = _make_samples(20, rng)
        k_grid = np.logspace(-2, 0, 5)
        a_grid = np.array([0.5, 1.0])

        le = LatentEmulator(
            codec=PCACodec(n_components=3),
            emulator_factory=lambda: GPEmulator(feature_names=PARAM_NAMES),
        )
        le.fit(samples, linear_matter_power, k_grid, a_grid, param_names=PARAM_NAMES)
        assert le.is_fitted
        curve = le.predict(samples[0])
        assert curve.shape == (2 * 5,)


class TestPyTorchInversionSingleTarget:

    def test_single_target_inversion(self):
        pytest.importorskip("torch")
        from tissage_cosmique.emulators import PyTorchEmulator

        rng = np.random.default_rng(42)
        samples = _make_samples(30, rng)
        X, y = build_training_data(
            comoving_angular_distance, samples, A_GRID, param_names=PARAM_NAMES,
        )
        emu = PyTorchEmulator(
            feature_names=PARAM_NAMES + ["a"], hidden_layers=[32, 32], n_epochs=200, seed=42,
        )
        emu.fit(X, y)

        truth = comoving_angular_distance(
            {"Omega_c": 0.27, "h": 0.68, "sigma8": 0.81, **FIXED},
            np.array([0.5]),
        )
        result = emu.invert(
            y_target=truth,
            free_params=["Omega_c"],
            fixed_params={"h": 0.68, "sigma8": 0.81, "a": np.array([0.5])},
            x0={"Omega_c": 0.27},
            bounds={"Omega_c": (0.20, 0.35)},
            n_steps=200,
        )
        assert "Omega_c" in result.x_solution


class TestInversionWithBoundsOnly:

    def test_invert_with_bounds_no_x0(self):
        rng = np.random.default_rng(42)
        samples = _make_samples(30, rng)
        X, y = build_training_data(
            comoving_angular_distance, samples, A_GRID, param_names=PARAM_NAMES,
        )
        emu = GPEmulator(feature_names=PARAM_NAMES + ["a"])
        emu.fit(X, y)

        truth = comoving_angular_distance(
            {"Omega_c": 0.27, "h": 0.68, "sigma8": 0.81, **FIXED}, A_GRID,
        )
        from tissage_cosmique.emulators import invert_minimize

        result = invert_minimize(
            emu, truth,
            free_params=["Omega_c"],
            fixed_params={"h": 0.68, "sigma8": 0.81, "a": A_GRID},
            bounds={"Omega_c": (0.20, 0.35)},
        )
        assert "Omega_c" in result.x_solution
