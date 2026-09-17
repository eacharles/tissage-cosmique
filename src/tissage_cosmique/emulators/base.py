"""Abstract base class for emulators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class Emulator(ABC):
    """Base class for surrogate models that replace tracked computations.

    Subclasses implement a specific ML backend (GP, neural net, symbolic
    regression, etc.) while exposing a uniform fit/predict/save/load interface.
    """

    def __init__(self, feature_names: list[str] | None = None) -> None:
        self.feature_names = feature_names
        self._n_training_samples = 0
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the emulator.

        Parameters
        ----------
        X
            Feature matrix of shape (n_samples, n_features).
        y
            Target array of shape (n_samples,).
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets for new inputs.

        Parameters
        ----------
        X
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predictions of shape (n_samples,).
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Serialize the emulator to a file."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> Emulator:
        """Load a previously saved emulator."""

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def n_training_samples(self) -> int:
        return self._n_training_samples

    def invert(
        self,
        y_target: np.ndarray,
        free_params: list[str],
        fixed_params: dict[str, float | np.ndarray],
        *,
        x0: dict[str, float] | None = None,
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> Any:
        """Find free parameter values that reproduce target outputs.

        Default implementation uses scipy.optimize.minimize. Backends may
        override with specialized methods (gradients, uncertainty weighting,
        symbolic algebra).
        """
        from .inversion import invert_minimize

        return invert_minimize(self, y_target, free_params, fixed_params, x0=x0, bounds=bounds)

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return a dict describing the emulator state."""
