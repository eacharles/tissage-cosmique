"""Tests for the LatentEmulator (codec + per-dimension emulators)."""

import pytest

pyccl = pytest.importorskip("pyccl")

import numpy as np  # noqa: E402

from tissage_cosmique.computations.distances import comoving_angular_distance  # noqa: E402
from tissage_cosmique.emulators import GPEmulator, LatentEmulator, PCACodec  # noqa: E402

PARAM_NAMES = ["Omega_c", "h", "sigma8"]
FIXED = dict(Omega_b=0.0486, n_s=0.9667, Omega_k=0.0, w0=-1.0, wa=0.0)
A_GRID = np.linspace(0.2, 0.8, 30)


def _make_samples(n: int, rng: np.random.Generator) -> list[dict]:
    return [
        {"Omega_c": rng.uniform(0.22, 0.32), "h": rng.uniform(0.62, 0.75),
         "sigma8": rng.uniform(0.77, 0.87), **FIXED}
        for _ in range(n)
    ]


@pytest.fixture(scope="module")
def pca_latent_emulator():
    rng = np.random.default_rng(42)
    samples = _make_samples(50, rng)
    codec = PCACodec(n_components=3)
    le = LatentEmulator(codec, lambda: GPEmulator(feature_names=PARAM_NAMES))
    le.fit(samples, comoving_angular_distance, A_GRID, param_names=PARAM_NAMES)
    return le, samples


class TestLatentEmulatorPCA:

    def test_is_fitted(self, pca_latent_emulator):
        le, _ = pca_latent_emulator
        assert le.is_fitted
        assert le.codec.is_fitted
        assert len(le.emulators) == 3

    def test_predict_shape(self, pca_latent_emulator):
        le, _ = pca_latent_emulator
        params = {"Omega_c": 0.27, "h": 0.68, "sigma8": 0.81, **FIXED}
        curve = le.predict(params)
        assert curve.shape == (len(A_GRID),)

    def test_predict_latent_shape(self, pca_latent_emulator):
        le, _ = pca_latent_emulator
        params = {"Omega_c": 0.27, "h": 0.68, "sigma8": 0.81, **FIXED}
        z = le.predict_latent(params)
        assert z.shape == (3,)

    def test_accuracy_on_holdout(self, pca_latent_emulator):
        le, _ = pca_latent_emulator
        rng = np.random.default_rng(999)
        test_samples = _make_samples(10, rng)
        for params in test_samples:
            truth = comoving_angular_distance(params, A_GRID)
            pred = le.predict(params)
            rel_err = np.abs(pred - truth) / np.maximum(np.abs(truth), 1.0)
            assert np.mean(rel_err) < 0.05

    def test_save_load(self, pca_latent_emulator, tmp_path):
        le, _ = pca_latent_emulator
        le.save(tmp_path / "latent_emu")
        loaded = LatentEmulator.load(tmp_path / "latent_emu", PCACodec, GPEmulator)
        assert loaded.is_fitted

        params = {"Omega_c": 0.27, "h": 0.68, "sigma8": 0.81, **FIXED}
        np.testing.assert_allclose(le.predict(params), loaded.predict(params), rtol=1e-5)

    def test_metadata(self, pca_latent_emulator):
        le, _ = pca_latent_emulator
        meta = le.metadata
        assert meta["is_fitted"]
        assert meta["n_latent"] == 3
        assert meta["codec"]["type"] == "pca"
        assert meta["emulator_type"] == "gp"


class TestLatentEmulatorAutoencoder:

    def test_fit_and_predict(self):
        pytest.importorskip("torch")
        from tissage_cosmique.emulators import AutoencoderCodec

        rng = np.random.default_rng(42)
        samples = _make_samples(40, rng)
        codec = AutoencoderCodec(n_latent=3, hidden_layers=[32], n_epochs=100, seed=42)
        le = LatentEmulator(codec, lambda: GPEmulator(feature_names=PARAM_NAMES))
        le.fit(samples, comoving_angular_distance, A_GRID, param_names=PARAM_NAMES)

        assert le.is_fitted
        params = {"Omega_c": 0.27, "h": 0.68, "sigma8": 0.81, **FIXED}
        curve = le.predict(params)
        assert curve.shape == (len(A_GRID),)
