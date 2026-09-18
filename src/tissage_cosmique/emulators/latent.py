"""Latent space emulator: codec (encoder/decoder) + per-dimension emulators."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .base import Emulator
from .codecs.base import Codec


class LatentEmulator:
    """Composite emulator that works in a compressed latent space.

    Compresses output curves via a codec (PCA or autoencoder), then trains
    one scalar emulator per latent dimension. Prediction decodes the
    emulated latent vector back to the full output curve.

    Parameters
    ----------
    codec
        A fitted or unfitted :class:`Codec` instance.
    emulator_factory
        Callable that returns a fresh :class:`Emulator` instance (called
        once per latent dimension).
    """

    def __init__(
        self,
        codec: Codec,
        emulator_factory: Callable[[], Emulator],
    ) -> None:
        self._codec = codec
        self._emulator_factory = emulator_factory
        self._emulators: list[Emulator] = []
        self._param_names: list[str] = []
        self._is_fitted = False

    @property
    def codec(self) -> Codec:
        return self._codec

    @property
    def emulators(self) -> list[Emulator]:
        return self._emulators

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(
        self,
        param_samples: list[dict[str, Any]],
        computation_fn: Callable[..., np.ndarray],
        *grids: np.ndarray,
        param_names: list[str],
    ) -> None:
        """Run computations, fit codec, and train per-dimension emulators.

        Parameters
        ----------
        param_samples
            List of cosmological parameter dictionaries.
        computation_fn
            Function with signature ``(params_dict, *grids) -> np.ndarray``.
        *grids
            Grid arrays passed to the computation function.
        param_names
            Ordered list of parameter keys for the emulator feature matrix.
        """
        self._param_names = param_names

        Y = np.array([computation_fn(p, *grids) for p in param_samples])
        if Y.ndim > 2:
            Y = Y.reshape(len(param_samples), -1)

        if not self._codec.is_fitted:
            self._codec.fit(Y)

        Z = self._codec.encode(Y)

        X = np.array([[p[k] for k in param_names] for p in param_samples])

        self._emulators = []
        for k in range(self._codec.n_latent):
            emu = self._emulator_factory()
            emu.fit(X, Z[:, k])
            self._emulators.append(emu)

        self._is_fitted = True

    def predict(self, cosmo_params: dict[str, Any]) -> np.ndarray:
        """Predict the full output curve for a parameter set.

        Parameters
        ----------
        cosmo_params
            Cosmological parameter dictionary.

        Returns
        -------
        np.ndarray
            Reconstructed output curve.
        """
        x = np.array([[cosmo_params[k] for k in self._param_names]])
        z = np.array([[emu.predict(x)[0] for emu in self._emulators]])
        return self._codec.decode(z)[0]

    def predict_latent(self, cosmo_params: dict[str, Any]) -> np.ndarray:
        """Predict only the latent code (without decoding).

        Returns
        -------
        np.ndarray
            Latent vector of shape ``(n_latent,)``.
        """
        x = np.array([[cosmo_params[k] for k in self._param_names]])
        return np.array([emu.predict(x)[0] for emu in self._emulators])

    def save(self, path: str | Path) -> None:
        """Save the latent emulator (codec + all emulators) to a directory."""
        import joblib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self._codec.save(path / "codec.joblib")

        for k, emu in enumerate(self._emulators):
            emu.save(path / f"emulator_{k}.joblib")

        meta = {
            "n_latent": self._codec.n_latent,
            "param_names": self._param_names,
            "n_emulators": len(self._emulators),
            "is_fitted": self._is_fitted,
        }
        joblib.dump(meta, path / "meta.joblib")

    @classmethod
    def load(
        cls,
        path: str | Path,
        codec_cls: type[Codec],
        emulator_cls: type[Emulator],
    ) -> LatentEmulator:
        """Load a saved latent emulator.

        Parameters
        ----------
        path
            Directory containing codec.joblib, emulator_*.joblib, meta.joblib.
        codec_cls
            The Codec subclass to use for loading (e.g., PCACodec).
        emulator_cls
            The Emulator subclass to use for loading (e.g., GPEmulator).
        """
        import joblib

        path = Path(path)
        meta = joblib.load(path / "meta.joblib")

        codec = codec_cls.load(path / "codec.joblib")
        emulators = [
            emulator_cls.load(path / f"emulator_{k}.joblib")
            for k in range(meta["n_emulators"])
        ]

        le = cls(codec=codec, emulator_factory=lambda: emulator_cls())
        le._emulators = emulators
        le._param_names = meta["param_names"]
        le._is_fitted = meta["is_fitted"]
        return le

    @property
    def metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "is_fitted": self._is_fitted,
            "param_names": self._param_names,
            "codec": self._codec.metadata,
            "n_latent": self._codec.n_latent,
        }
        if self._emulators:
            result["emulator_type"] = self._emulators[0].metadata.get("type", "unknown")
        return result
