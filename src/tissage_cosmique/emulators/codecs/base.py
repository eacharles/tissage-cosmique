"""Abstract base class for latent space codecs (encoder/decoder)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class Codec(ABC):
    """Base class for encoding output curves into a latent representation.

    A codec compresses high-dimensional output vectors (e.g., distances at
    30 scale factors) into a low-dimensional latent code, and reconstructs
    the full output from the code.
    """

    def __init__(self, n_latent: int) -> None:
        self._n_latent = n_latent
        self._n_features: int = 0
        self._is_fitted = False

    @property
    def n_latent(self) -> int:
        return self._n_latent

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @abstractmethod
    def fit(self, Y: np.ndarray) -> None:
        """Fit the codec to a matrix of output curves.

        Parameters
        ----------
        Y
            Output matrix of shape ``(n_samples, n_grid)``.
        """

    @abstractmethod
    def encode(self, Y: np.ndarray) -> np.ndarray:
        """Encode output curves to latent vectors.

        Parameters
        ----------
        Y
            Output matrix of shape ``(n_samples, n_grid)``.

        Returns
        -------
        np.ndarray
            Latent codes of shape ``(n_samples, n_latent)``.
        """

    @abstractmethod
    def decode(self, Z: np.ndarray) -> np.ndarray:
        """Decode latent vectors to output curves.

        Parameters
        ----------
        Z
            Latent codes of shape ``(n_samples, n_latent)``.

        Returns
        -------
        np.ndarray
            Reconstructed output of shape ``(n_samples, n_grid)``.
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Serialize the codec to a file."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> Codec:
        """Load a previously saved codec."""

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return a dict describing the codec state."""
