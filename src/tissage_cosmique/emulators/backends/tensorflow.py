"""TensorFlow/Keras feedforward neural network emulator backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from ..base import Emulator


class TensorFlowEmulator(Emulator):
    """Emulator backed by a Keras feedforward neural network.

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
    validation_split
        Fraction of training data for validation (default: 0.1).
    """

    def __init__(
        self,
        feature_names: list[str] | None = None,
        *,
        hidden_layers: list[int] | None = None,
        n_epochs: int = 500,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        validation_split: float = 0.1,
    ) -> None:
        super().__init__(feature_names=feature_names)
        self._hidden_layers = hidden_layers or [64, 64]
        self._n_epochs = n_epochs
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._validation_split = validation_split
        self._x_scaler = StandardScaler()
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._model: Any = None
        self._training_loss: float | None = None

    def _build_model(self, n_features: int) -> Any:
        import tensorflow as tf

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(n_features,)))
        for units in self._hidden_layers:
            model.add(tf.keras.layers.Dense(units, activation="relu"))
        model.add(tf.keras.layers.Dense(1))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self._learning_rate),
            loss="mse",
        )
        return model

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_scaled = self._x_scaler.fit_transform(X)
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y)) or 1.0
        y_scaled = (y - self._y_mean) / self._y_std

        self._model = self._build_model(X.shape[1])
        history = self._model.fit(
            X_scaled,
            y_scaled,
            epochs=self._n_epochs,
            batch_size=self._batch_size,
            validation_split=self._validation_split,
            verbose=0,
        )
        self._training_loss = float(history.history["loss"][-1])
        self._n_training_samples = len(y)
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._x_scaler.transform(X)
        y_scaled = self._model.predict(X_scaled, verbose=0).flatten()
        return y_scaled * self._y_std + self._y_mean

    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        model_path = path.with_suffix(".keras")
        self._model.save(model_path)
        data = {
            "model_path": str(model_path),
            "x_scaler": self._x_scaler,
            "y_mean": self._y_mean,
            "y_std": self._y_std,
            "feature_names": self.feature_names,
            "hidden_layers": self._hidden_layers,
            "n_epochs": self._n_epochs,
            "batch_size": self._batch_size,
            "learning_rate": self._learning_rate,
            "validation_split": self._validation_split,
            "n_training_samples": self._n_training_samples,
            "is_fitted": self._is_fitted,
            "training_loss": self._training_loss,
        }
        joblib.dump(data, path)

    @classmethod
    def load(cls, path: str | Path) -> TensorFlowEmulator:
        import joblib
        import tensorflow as tf

        path = Path(path)
        data = joblib.load(path)
        emu = cls(
            feature_names=data["feature_names"],
            hidden_layers=data["hidden_layers"],
            n_epochs=data["n_epochs"],
            batch_size=data["batch_size"],
            learning_rate=data["learning_rate"],
            validation_split=data["validation_split"],
        )
        emu._x_scaler = data["x_scaler"]
        emu._y_mean = data["y_mean"]
        emu._y_std = data["y_std"]
        emu._model = tf.keras.models.load_model(data["model_path"])
        emu._n_training_samples = data["n_training_samples"]
        emu._is_fitted = data["is_fitted"]
        emu._training_loss = data["training_loss"]
        return emu

    def invert(
        self,
        y_target: np.ndarray,
        free_params: list[str],
        fixed_params: dict[str, float | np.ndarray],
        *,
        x0: dict[str, float] | None = None,
        bounds: dict[str, tuple[float, float]] | None = None,
        n_steps: int = 1000,
        lr: float = 0.01,
    ) -> Any:
        """Gradient-based inversion via TensorFlow GradientTape."""
        import tensorflow as tf

        from ..inversion import InversionResult, _build_feature_matrix, _resolve_indices

        feature_names = self.feature_names
        if feature_names is None:
            return super().invert(y_target, free_params, fixed_params, x0=x0, bounds=bounds)

        y_target = np.atleast_1d(y_target)
        n_targets = len(y_target)
        n_features = len(feature_names)

        free_indices, _ = _resolve_indices(feature_names, free_params, fixed_params)
        fixed_idx_val = {feature_names.index(k): v for k, v in fixed_params.items()}

        if x0 is not None:
            init = np.array([x0[p] for p in free_params], dtype=np.float32)
        elif bounds is not None:
            init = np.array([(bounds[p][0] + bounds[p][1]) / 2 for p in free_params], dtype=np.float32)
        else:
            init = np.zeros(len(free_params), dtype=np.float32)

        free_var = tf.Variable(init)
        y_target_tf = tf.constant(y_target, dtype=tf.float32)
        optimizer = tf.optimizers.Adam(learning_rate=lr)

        for _ in range(n_steps):
            with tf.GradientTape() as tape:
                X_np = _build_feature_matrix(
                    free_var.numpy(), free_indices, fixed_idx_val, n_features, n_targets,
                )
                X_scaled = self._x_scaler.transform(X_np).astype(np.float32)
                X_tf = tf.constant(X_scaled)
                y_scaled = tf.reshape(self._model(X_tf, training=False), [-1])
                y_pred_tf = y_scaled * self._y_std + self._y_mean
                loss = tf.reduce_mean((y_pred_tf - y_target_tf) ** 2)

            grads = tape.gradient(loss, [free_var])
            if grads[0] is not None:
                optimizer.apply_gradients(zip(grads, [free_var]))

            if bounds:
                free_var.assign(tf.clip_by_value(
                    free_var,
                    [bounds[p][0] for p in free_params],
                    [bounds[p][1] for p in free_params],
                ))

        X_final = _build_feature_matrix(
            free_var.numpy(), free_indices, fixed_idx_val, n_features, n_targets,
        )
        y_pred = self.predict(X_final)
        y_scale = max(float(np.abs(y_target).mean()), 1e-10)
        rel_residual = float(np.sqrt(np.mean(((y_pred - y_target) / y_scale) ** 2)))

        return InversionResult(
            x_solution={p: float(free_var[i].numpy()) for i, p in enumerate(free_params)},
            y_predicted=y_pred,
            y_target=y_target,
            residual=rel_residual,
            success=rel_residual < 0.01,
            message=f"TensorFlow inversion completed in {n_steps} steps",
        )

    @property
    def metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": "tensorflow",
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
