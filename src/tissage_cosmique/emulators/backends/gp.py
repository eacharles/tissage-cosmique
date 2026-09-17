"""Gaussian Process emulator backend using scikit-learn."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler

from ..base import Emulator


class GPEmulator(Emulator):
    """Emulator backed by a scikit-learn Gaussian Process.

    Parameters
    ----------
    feature_names
        Names of the input features (used for metadata and dict-to-array conversion).
    kernel
        A scikit-learn kernel instance. Defaults to ``RBF() + WhiteKernel()``.
    n_restarts_optimizer
        Number of optimizer restarts for kernel hyperparameter tuning.
    alpha
        Noise level added to the diagonal of the kernel matrix.
    normalize
        If True, standardize X and y before fitting (recommended).
    """

    def __init__(
        self,
        feature_names: list[str] | None = None,
        *,
        kernel: Any | None = None,
        n_restarts_optimizer: int = 5,
        alpha: float = 1e-6,
        normalize: bool = True,
    ) -> None:
        super().__init__(feature_names=feature_names)
        self._kernel = kernel or (ConstantKernel() * RBF())
        self._gp = GaussianProcessRegressor(
            kernel=self._kernel,
            n_restarts_optimizer=n_restarts_optimizer,
            alpha=alpha,
            normalize_y=False,
        )
        self._normalize = normalize
        self._x_scaler: StandardScaler | None = StandardScaler() if normalize else None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._training_score: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_fit = X
        if self._x_scaler is not None:
            X_fit = self._x_scaler.fit_transform(X)
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y)) or 1.0
        y_fit = (y - self._y_mean) / self._y_std
        self._gp.fit(X_fit, y_fit)
        self._n_training_samples = len(y)
        self._is_fitted = True
        self._training_score = float(self._gp.score(X_fit, y_fit))

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_pred = X
        if self._x_scaler is not None:
            X_pred = self._x_scaler.transform(X)
        y_scaled = self._gp.predict(X_pred)
        return y_scaled * self._y_std + self._y_mean

    def predict_with_std(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty estimates.

        Returns
        -------
        mean
            Predicted means of shape ``(n_samples,)``.
        std
            Predicted standard deviations of shape ``(n_samples,)``.
        """
        X_pred = X
        if self._x_scaler is not None:
            X_pred = self._x_scaler.transform(X)
        y_scaled, std_scaled = self._gp.predict(X_pred, return_std=True)
        return y_scaled * self._y_std + self._y_mean, std_scaled * self._y_std

    def save(self, path: str | Path) -> None:
        import joblib

        data = {
            "gp": self._gp,
            "x_scaler": self._x_scaler,
            "y_mean": self._y_mean,
            "y_std": self._y_std,
            "feature_names": self.feature_names,
            "n_training_samples": self._n_training_samples,
            "is_fitted": self._is_fitted,
            "training_score": self._training_score,
            "normalize": self._normalize,
        }
        joblib.dump(data, path)

    @classmethod
    def load(cls, path: str | Path) -> GPEmulator:
        import joblib

        data = joblib.load(path)
        emu = cls(
            feature_names=data["feature_names"],
            normalize=data["normalize"],
        )
        emu._gp = data["gp"]
        emu._x_scaler = data["x_scaler"]
        emu._y_mean = data["y_mean"]
        emu._y_std = data["y_std"]
        emu._n_training_samples = data["n_training_samples"]
        emu._is_fitted = data["is_fitted"]
        emu._training_score = data["training_score"]
        return emu

    @property
    def metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": "gp",
            "feature_names": self.feature_names,
            "is_fitted": self._is_fitted,
            "n_training_samples": self._n_training_samples,
            "normalize": self._normalize,
        }
        if self._is_fitted:
            result["training_score"] = self._training_score
            result["kernel"] = str(self._gp.kernel_)
        return result
