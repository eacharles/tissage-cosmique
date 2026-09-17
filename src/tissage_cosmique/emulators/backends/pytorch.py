"""PyTorch feedforward neural network emulator backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from ..base import Emulator


class PyTorchEmulator(Emulator):
    """Emulator backed by a PyTorch feedforward neural network.

    Parameters
    ----------
    feature_names
        Names of the input features.
    hidden_layers
        List of hidden layer sizes (default: [64, 64]).
    n_epochs
        Number of training epochs (default: 500).
    batch_size
        Training batch size (default: 32).
    learning_rate
        Adam optimizer learning rate (default: 1e-3).
    """

    def __init__(
        self,
        feature_names: list[str] | None = None,
        *,
        hidden_layers: list[int] | None = None,
        n_epochs: int = 500,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__(feature_names=feature_names)
        self._hidden_layers = hidden_layers or [64, 64]
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._x_scaler = StandardScaler()
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._model: Any = None
        self._n_features: int = 0
        self._training_loss: float | None = None

    def _build_model(self, n_features: int) -> Any:
        import torch

        layers: list[Any] = []
        in_size = n_features
        for units in self._hidden_layers:
            layers.append(torch.nn.Linear(in_size, units))
            layers.append(torch.nn.ReLU())
            in_size = units
        layers.append(torch.nn.Linear(in_size, 1))
        return torch.nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        import torch

        X_scaled = self._x_scaler.fit_transform(X)
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y)) or 1.0
        y_scaled = (y - self._y_mean) / self._y_std

        self._n_features = X.shape[1]
        self._model = self._build_model(self._n_features)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._learning_rate)
        loss_fn = torch.nn.MSELoss()

        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(y_scaled, dtype=torch.float32).unsqueeze(1)

        self._model.train()
        n = len(X_t)
        for _ in range(self._n_epochs):
            perm = torch.randperm(n)
            for start in range(0, n, self._batch_size):
                idx = perm[start : start + self._batch_size]
                pred = self._model(X_t[idx])
                loss = loss_fn(pred, y_t[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self._model.eval()
        with torch.no_grad():
            final_pred = self._model(X_t)
            self._training_loss = float(loss_fn(final_pred, y_t).item())

        self._n_training_samples = len(y)
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        X_scaled = self._x_scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        self._model.eval()
        with torch.no_grad():
            y_scaled = self._model(X_t).numpy().flatten()
        return y_scaled * self._y_std + self._y_mean

    def save(self, path: str | Path) -> None:
        import joblib

        data = {
            "state_dict": {k: v.cpu().numpy() for k, v in self._model.state_dict().items()},
            "x_scaler": self._x_scaler,
            "y_mean": self._y_mean,
            "y_std": self._y_std,
            "feature_names": self.feature_names,
            "hidden_layers": self._hidden_layers,
            "n_features": self._n_features,
            "n_epochs": self._n_epochs,
            "batch_size": self._batch_size,
            "learning_rate": self._learning_rate,
            "n_training_samples": self._n_training_samples,
            "is_fitted": self._is_fitted,
            "training_loss": self._training_loss,
        }
        joblib.dump(data, path)

    @classmethod
    def load(cls, path: str | Path) -> PyTorchEmulator:
        import joblib
        import torch

        data = joblib.load(path)
        emu = cls(
            feature_names=data["feature_names"],
            hidden_layers=data["hidden_layers"],
            n_epochs=data["n_epochs"],
            batch_size=data["batch_size"],
            learning_rate=data["learning_rate"],
        )
        emu._x_scaler = data["x_scaler"]
        emu._y_mean = data["y_mean"]
        emu._y_std = data["y_std"]
        emu._n_features = data["n_features"]
        emu._model = emu._build_model(data["n_features"])
        state = {k: torch.tensor(v) for k, v in data["state_dict"].items()}
        emu._model.load_state_dict(state)
        emu._model.eval()
        emu._n_training_samples = data["n_training_samples"]
        emu._is_fitted = data["is_fitted"]
        emu._training_loss = data["training_loss"]
        return emu

    @property
    def metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": "pytorch",
            "feature_names": self.feature_names,
            "is_fitted": self._is_fitted,
            "n_training_samples": self._n_training_samples,
            "hidden_layers": self._hidden_layers,
            "n_epochs": self._n_epochs,
            "learning_rate": self._learning_rate,
        }
        if self._is_fitted:
            result["training_loss"] = self._training_loss
        return result
