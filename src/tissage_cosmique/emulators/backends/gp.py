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

    def invert(
        self,
        y_target: np.ndarray,
        free_params: list[str],
        fixed_params: dict[str, float | np.ndarray],
        *,
        x0: dict[str, float] | None = None,
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> Any:
        """Uncertainty-weighted inversion using GP predicted std."""
        from scipy.optimize import minimize as scipy_minimize

        from macon.common import unexpected

        from ..inversion import InversionResult, _build_feature_matrix, _resolve_indices

        feature_names = self.feature_names
        if unexpected(feature_names is None):
            return super().invert(y_target, free_params, fixed_params, x0=x0, bounds=bounds)
        assert feature_names is not None

        y_target = np.atleast_1d(y_target)
        n_targets = len(y_target)
        n_features = len(feature_names)

        free_indices, _ = _resolve_indices(feature_names, free_params, fixed_params)
        fixed_idx_val = {feature_names.index(k): v for k, v in fixed_params.items()}

        if x0 is not None:
            x0_arr = np.array([x0[p] for p in free_params])
        elif bounds is not None:
            x0_arr = np.array([(bounds[p][0] + bounds[p][1]) / 2 for p in free_params])
        else:  # pragma: no cover — inversion without x0 or bounds would start from zeros, unlikely to converge
            x0_arr = np.zeros(len(free_params))

        scipy_bounds = [bounds[p] for p in free_params] if bounds else None

        def objective(free_values: np.ndarray) -> float:
            X = _build_feature_matrix(free_values, free_indices, fixed_idx_val, n_features, n_targets)
            y_pred, y_std = self.predict_with_std(X)
            y_std = np.maximum(y_std, 1e-10)
            return float(np.sum(((y_pred - y_target) / y_std) ** 2))

        result = scipy_minimize(objective, x0_arr, method="L-BFGS-B", bounds=scipy_bounds)

        X_final = _build_feature_matrix(result.x, free_indices, fixed_idx_val, n_features, n_targets)
        y_pred = self.predict(X_final)
        y_scale = np.maximum(np.abs(y_target).mean(), 1e-10)
        rel_residual = float(np.sqrt(np.mean(((y_pred - y_target) / y_scale) ** 2)))

        return InversionResult(
            x_solution={p: float(v) for p, v in zip(free_params, result.x)},
            y_predicted=y_pred,
            y_target=y_target,
            residual=rel_residual,
            success=bool(result.success),
            message=str(result.message),
        )

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
