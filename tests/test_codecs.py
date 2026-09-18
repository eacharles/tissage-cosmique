"""Tests for latent space codecs (PCA and autoencoder)."""

import pytest

pyccl = pytest.importorskip("pyccl")

import numpy as np  # noqa: E402

from tissage_cosmique.computations.distances import comoving_angular_distance  # noqa: E402
from tissage_cosmique.emulators.codecs import AutoencoderCodec, PCACodec  # noqa: E402

FIXED = dict(Omega_b=0.0486, n_s=0.9667, Omega_k=0.0, w0=-1.0, wa=0.0)
A_GRID = np.linspace(0.2, 0.8, 30)


def _make_curve_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    samples = [
        {"Omega_c": rng.uniform(0.22, 0.32), "h": rng.uniform(0.62, 0.75),
         "sigma8": rng.uniform(0.77, 0.87), **FIXED}
        for _ in range(n)
    ]
    return np.array([comoving_angular_distance(p, A_GRID) for p in samples])


@pytest.fixture(scope="module")
def curve_matrix():
    return _make_curve_matrix(50, np.random.default_rng(42))


class TestPCACodec:

    def test_fit_and_shapes(self, curve_matrix):
        codec = PCACodec(n_components=5)
        codec.fit(curve_matrix)
        assert codec.is_fitted
        assert codec.n_latent == 5
        assert codec.n_features == 30

    def test_encode_decode_roundtrip(self, curve_matrix):
        codec = PCACodec(n_components=5)
        codec.fit(curve_matrix)
        Z = codec.encode(curve_matrix)
        assert Z.shape == (50, 5)
        Y_recon = codec.decode(Z)
        assert Y_recon.shape == curve_matrix.shape
        np.testing.assert_allclose(Y_recon, curve_matrix, rtol=1e-4)

    def test_variance_explained(self, curve_matrix):
        codec = PCACodec(n_components=3)
        codec.fit(curve_matrix)
        evr = codec.explained_variance_ratio
        assert evr is not None
        assert len(evr) == 3
        assert np.sum(evr) > 0.999

    def test_variance_threshold(self, curve_matrix):
        codec = PCACodec(n_components=0.9999)
        codec.fit(curve_matrix)
        assert codec.n_latent >= 1
        cumvar = codec.cumulative_variance
        assert cumvar is not None
        assert cumvar[-1] >= 0.9999

    def test_save_load(self, curve_matrix, tmp_path):
        codec = PCACodec(n_components=3)
        codec.fit(curve_matrix)
        Z_orig = codec.encode(curve_matrix[:5])

        codec.save(tmp_path / "pca.joblib")
        loaded = PCACodec.load(tmp_path / "pca.joblib")
        Z_loaded = loaded.encode(curve_matrix[:5])
        np.testing.assert_array_equal(Z_orig, Z_loaded)

    def test_metadata(self, curve_matrix):
        codec = PCACodec(n_components=3)
        codec.fit(curve_matrix)
        meta = codec.metadata
        assert meta["type"] == "pca"
        assert meta["n_latent"] == 3
        assert meta["total_explained"] > 0.999


class TestAutoencoderCodec:

    def test_fit_and_shapes(self, curve_matrix):
        pytest.importorskip("torch")
        codec = AutoencoderCodec(n_latent=5, hidden_layers=[32], n_epochs=100, seed=42)
        codec.fit(curve_matrix)
        assert codec.is_fitted
        assert codec.n_latent == 5

    def test_encode_decode(self, curve_matrix):
        pytest.importorskip("torch")
        codec = AutoencoderCodec(n_latent=5, hidden_layers=[32], n_epochs=200, seed=42)
        codec.fit(curve_matrix)
        Z = codec.encode(curve_matrix)
        assert Z.shape == (50, 5)
        Y_recon = codec.decode(Z)
        assert Y_recon.shape == curve_matrix.shape
        rel_err = np.abs(Y_recon - curve_matrix) / np.maximum(np.abs(curve_matrix), 1.0)
        assert np.mean(rel_err) < 0.1

    def test_save_load(self, curve_matrix, tmp_path):
        pytest.importorskip("torch")
        codec = AutoencoderCodec(n_latent=3, hidden_layers=[32], n_epochs=50, seed=42)
        codec.fit(curve_matrix)
        Z_orig = codec.encode(curve_matrix[:5])

        codec.save(tmp_path / "ae.joblib")
        loaded = AutoencoderCodec.load(tmp_path / "ae.joblib")
        Z_loaded = loaded.encode(curve_matrix[:5])
        np.testing.assert_allclose(Z_orig, Z_loaded, rtol=1e-5)

    def test_metadata(self, curve_matrix):
        pytest.importorskip("torch")
        codec = AutoencoderCodec(n_latent=3, hidden_layers=[32], n_epochs=50, seed=42)
        codec.fit(curve_matrix)
        meta = codec.metadata
        assert meta["type"] == "autoencoder"
        assert meta["n_latent"] == 3
        assert "training_loss" in meta
