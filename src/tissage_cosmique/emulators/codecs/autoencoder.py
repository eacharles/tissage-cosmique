"""PyTorch autoencoder codec for nonlinear latent space encoding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from .base import Codec


class AutoencoderCodec(Codec):
    """Codec using a PyTorch autoencoder (encoder + decoder networks).

    More expressive than PCA for functions with sharp features or
    nonlinear structure in the output space.

    Parameters
    ----------
    n_latent
        Dimensionality of the latent space.
    hidden_layers
        Hidden layer sizes for both encoder and decoder (default: [64, 32]).
    n_epochs
        Training epochs (default: 500).
    learning_rate
        Adam optimizer learning rate (default: 1e-3).
    seed
        Random seed for reproducible training.
    """

    def __init__(
        self,
        n_latent: int = 5,
        *,
        hidden_layers: list[int] | None = None,
        n_epochs: int = 500,
        learning_rate: float = 1e-3,
        seed: int | None = None,
    ) -> None:
        super().__init__(n_latent=n_latent)
        self._hidden_layers = hidden_layers or [64, 32]
        self._n_epochs = n_epochs
        self._learning_rate = learning_rate
        self._seed = seed
        self._scaler = StandardScaler()
        self._encoder: Any = None
        self._decoder: Any = None
        self._training_loss: float | None = None

    def _build_networks(self, n_features: int) -> tuple[Any, Any]:
        import torch

        enc_layers: list[Any] = []
        in_size = n_features
        for units in self._hidden_layers:
            enc_layers.append(torch.nn.Linear(in_size, units))
            enc_layers.append(torch.nn.ReLU())
            in_size = units
        enc_layers.append(torch.nn.Linear(in_size, self._n_latent))
        encoder = torch.nn.Sequential(*enc_layers)

        dec_layers: list[Any] = []
        in_size = self._n_latent
        for units in reversed(self._hidden_layers):
            dec_layers.append(torch.nn.Linear(in_size, units))
            dec_layers.append(torch.nn.ReLU())
            in_size = units
        dec_layers.append(torch.nn.Linear(in_size, n_features))
        decoder = torch.nn.Sequential(*dec_layers)

        return encoder, decoder

    def fit(self, Y: np.ndarray) -> None:
        import torch

        if self._seed is not None:
            torch.manual_seed(self._seed)
            np.random.seed(self._seed)

        self._n_features = Y.shape[1]
        Y_scaled = self._scaler.fit_transform(Y)

        self._encoder, self._decoder = self._build_networks(self._n_features)
        params = list(self._encoder.parameters()) + list(self._decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=self._learning_rate)
        loss_fn = torch.nn.MSELoss()

        Y_t = torch.tensor(Y_scaled, dtype=torch.float32)

        for _ in range(self._n_epochs):
            z = self._encoder(Y_t)
            y_recon = self._decoder(z)
            loss = loss_fn(y_recon, Y_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self._training_loss = float(loss.item())
        self._is_fitted = True

    def encode(self, Y: np.ndarray) -> np.ndarray:
        import torch

        Y_scaled = self._scaler.transform(Y)
        Y_t = torch.tensor(Y_scaled, dtype=torch.float32)
        self._encoder.eval()
        with torch.no_grad():
            return self._encoder(Y_t).numpy()

    def decode(self, Z: np.ndarray) -> np.ndarray:
        import torch

        Z_t = torch.tensor(Z, dtype=torch.float32)
        self._decoder.eval()
        with torch.no_grad():
            Y_scaled = self._decoder(Z_t).numpy()
        return self._scaler.inverse_transform(Y_scaled)

    def save(self, path: str | Path) -> None:
        import joblib

        data = {
            "encoder_state": {k: v.cpu().numpy() for k, v in self._encoder.state_dict().items()},
            "decoder_state": {k: v.cpu().numpy() for k, v in self._decoder.state_dict().items()},
            "scaler": self._scaler,
            "n_latent": self._n_latent,
            "n_features": self._n_features,
            "hidden_layers": self._hidden_layers,
            "n_epochs": self._n_epochs,
            "learning_rate": self._learning_rate,
            "seed": self._seed,
            "is_fitted": self._is_fitted,
            "training_loss": self._training_loss,
        }
        joblib.dump(data, path)

    @classmethod
    def load(cls, path: str | Path) -> AutoencoderCodec:
        import joblib
        import torch

        data = joblib.load(path)
        codec = cls(
            n_latent=data["n_latent"],
            hidden_layers=data["hidden_layers"],
            n_epochs=data["n_epochs"],
            learning_rate=data["learning_rate"],
            seed=data.get("seed"),
        )
        codec._n_features = data["n_features"]
        codec._scaler = data["scaler"]
        codec._encoder, codec._decoder = codec._build_networks(data["n_features"])
        codec._encoder.load_state_dict({k: torch.tensor(v) for k, v in data["encoder_state"].items()})
        codec._decoder.load_state_dict({k: torch.tensor(v) for k, v in data["decoder_state"].items()})
        codec._encoder.eval()
        codec._decoder.eval()
        codec._is_fitted = data["is_fitted"]
        codec._training_loss = data["training_loss"]
        return codec

    @property
    def metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": "autoencoder",
            "n_latent": self._n_latent,
            "n_features": self._n_features,
            "hidden_layers": self._hidden_layers,
            "n_epochs": self._n_epochs,
            "is_fitted": self._is_fitted,
        }
        if self._is_fitted:
            result["training_loss"] = self._training_loss
        return result
