"""PCA-based codec for latent space encoding of output curves."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from .base import Codec


class PCACodec(Codec):
    """Codec using Principal Component Analysis.

    For smooth cosmological functions, 3-5 PCA components typically capture
    >99.99% of variance, making this an effective and fast compression.

    Parameters
    ----------
    n_components
        Number of PCA components. If float < 1, interpreted as minimum
        explained variance ratio (e.g., 0.9999).
    """

    def __init__(self, n_components: int | float = 5) -> None:
        n_latent = n_components if isinstance(n_components, int) else 0
        super().__init__(n_latent=n_latent)
        self._n_components_init = n_components
        self._pca: PCA | None = None

    def fit(self, Y: np.ndarray) -> None:
        self._n_features = Y.shape[1]
        self._pca = PCA(n_components=self._n_components_init)
        self._pca.fit(Y)
        self._n_latent = self._pca.n_components_
        self._is_fitted = True

    def encode(self, Y: np.ndarray) -> np.ndarray:
        assert self._pca is not None
        return self._pca.transform(Y)

    def decode(self, Z: np.ndarray) -> np.ndarray:
        assert self._pca is not None
        return self._pca.inverse_transform(Z)

    @property
    def explained_variance_ratio(self) -> np.ndarray | None:
        if self._pca is None:
            return None
        return self._pca.explained_variance_ratio_

    @property
    def cumulative_variance(self) -> np.ndarray | None:
        if self._pca is None:
            return None
        return np.cumsum(self._pca.explained_variance_ratio_)

    def save(self, path: str | Path) -> None:
        import joblib

        data = {
            "pca": self._pca,
            "n_components_init": self._n_components_init,
            "n_latent": self._n_latent,
            "n_features": self._n_features,
            "is_fitted": self._is_fitted,
        }
        joblib.dump(data, path)

    @classmethod
    def load(cls, path: str | Path) -> PCACodec:
        import joblib

        data = joblib.load(path)
        codec = cls(n_components=data["n_components_init"])
        codec._pca = data["pca"]
        codec._n_latent = data["n_latent"]
        codec._n_features = data["n_features"]
        codec._is_fitted = data["is_fitted"]
        return codec

    @property
    def metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": "pca",
            "n_latent": self._n_latent,
            "n_features": self._n_features,
            "is_fitted": self._is_fitted,
        }
        if self._is_fitted and self._pca is not None:
            result["explained_variance_ratio"] = self._pca.explained_variance_ratio_.tolist()
            result["total_explained"] = float(np.sum(self._pca.explained_variance_ratio_))
        return result
