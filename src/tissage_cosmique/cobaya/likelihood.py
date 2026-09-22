"""Cobaya Likelihood classes for tissage-cosmique data vectors."""

from __future__ import annotations

from typing import Any

import numpy as np
from cobaya.likelihood import Likelihood

from ..computations.likelihood import gaussian_log_likelihood


class TissageGaussianLikelihood(Likelihood):
    """Gaussian likelihood for a tissage-cosmique data vector.

    Compares a model data vector (from ``TissageCCLTheory``) against
    observed data using a precomputed covariance matrix.

    Configuration (in cobaya YAML or dict)::

        likelihood:
          tissage_cosmique.cobaya.TissageGaussianLikelihood:
            data_vector: [...]        # observed C(ℓ) values
            covariance: [[...], ...]  # covariance matrix
            # OR:
            sacc_file: path/to/data.sacc
    """

    data_vector: list[float] | None = None
    covariance: list[list[float]] | None = None
    sacc_file: str | None = None

    _data: np.ndarray
    _cov_inv: np.ndarray

    def initialize(self) -> None:
        if self.sacc_file is not None:
            self._load_from_sacc()
        elif self.data_vector is not None and self.covariance is not None:
            self._data = np.array(self.data_vector)
            cov = np.array(self.covariance)
            self._cov_inv = np.linalg.inv(cov)
        else:
            raise ValueError("Provide either sacc_file or data_vector + covariance")

    def _load_from_sacc(self) -> None:
        from ..computations.sacc_io import load_sacc

        _, _, _, cl_dict, covariance = load_sacc(self.sacc_file)  # type: ignore[arg-type]
        self._data = np.concatenate(list(cl_dict.values()))
        if covariance is not None:
            self._cov_inv = np.linalg.inv(covariance)
        else:
            raise ValueError(f"SACC file {self.sacc_file} has no covariance")

    def get_requirements(self) -> dict[str, Any]:
        return {"tcl_data_vector": None}

    def logp(self, **params_values: Any) -> float:
        model = self.provider.get_result("tcl_data_vector")
        return gaussian_log_likelihood(self._data, model, self._cov_inv)


class TissageSaccLikelihood(TissageGaussianLikelihood):
    """Convenience: Gaussian likelihood loaded entirely from a SACC file.

    Configuration::

        likelihood:
          tissage_cosmique.cobaya.TissageSaccLikelihood:
            sacc_file: path/to/data.sacc
    """

    sacc_file: str

    def initialize(self) -> None:
        self._load_from_sacc()
