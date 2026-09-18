"""PySR symbolic regression emulator backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from ..base import Emulator


class SymbolicEmulator(Emulator):
    """Emulator backed by PySR symbolic regression.

    Discovers a closed-form mathematical expression that approximates the
    training data. The resulting expression is compact, fast, and inspectable.

    Parameters
    ----------
    feature_names
        Names of the input features.
    niterations
        Number of PySR search iterations (default: 40).
    binary_operators
        Binary operators to search over (default: ["+", "-", "*", "/"]).
    unary_operators
        Unary operators to search over (default: ["exp", "log", "sqrt", "square"]).
    maxsize
        Maximum complexity of expressions (default: 25).
    """

    def __init__(
        self,
        feature_names: list[str] | None = None,
        *,
        niterations: int = 40,
        binary_operators: list[str] | None = None,
        unary_operators: list[str] | None = None,
        maxsize: int = 25,
    ) -> None:
        super().__init__(feature_names=feature_names)
        self._niterations = niterations
        self._binary_operators = binary_operators or ["+", "-", "*", "/"]
        self._unary_operators = unary_operators or ["exp", "log", "sqrt", "square"]
        self._maxsize = maxsize
        self._x_scaler = StandardScaler()
        self._model: Any = None
        self._best_expression: str | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from pysr import PySRRegressor

        X_scaled = self._x_scaler.fit_transform(X)
        self._model = PySRRegressor(
            niterations=self._niterations,
            binary_operators=self._binary_operators,
            unary_operators=self._unary_operators,
            maxsize=self._maxsize,
            temp_equation_file=True,
            verbosity=0,
        )
        self._model.fit(X_scaled, y, variable_names=self.feature_names)
        self._best_expression = str(self._model.sympy())
        self._n_training_samples = len(y)
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._x_scaler.transform(X)
        return self._model.predict(X_scaled)

    def save(self, path: str | Path) -> None:
        import joblib

        data = {
            "model": self._model,
            "x_scaler": self._x_scaler,
            "feature_names": self.feature_names,
            "niterations": self._niterations,
            "binary_operators": self._binary_operators,
            "unary_operators": self._unary_operators,
            "maxsize": self._maxsize,
            "n_training_samples": self._n_training_samples,
            "is_fitted": self._is_fitted,
            "best_expression": self._best_expression,
        }
        joblib.dump(data, path)

    @classmethod
    def load(cls, path: str | Path) -> SymbolicEmulator:
        import joblib

        data = joblib.load(path)
        emu = cls(
            feature_names=data["feature_names"],
            niterations=data["niterations"],
            binary_operators=data["binary_operators"],
            unary_operators=data["unary_operators"],
            maxsize=data["maxsize"],
        )
        emu._model = data["model"]
        emu._x_scaler = data["x_scaler"]
        emu._n_training_samples = data["n_training_samples"]
        emu._is_fitted = data["is_fitted"]
        emu._best_expression = data["best_expression"]
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
        """Symbolic inversion: try sympy.solve for single free param, else minimize."""
        if len(free_params) == 1 and self._best_expression is not None:
            try:
                return self._sympy_invert(y_target, free_params, fixed_params, bounds)
            except Exception as uexc:  # noqa: F841
                pass
        return super().invert(y_target, free_params, fixed_params, x0=x0, bounds=bounds)

    def _sympy_invert(
        self,
        y_target: np.ndarray,
        free_params: list[str],
        fixed_params: dict[str, float | np.ndarray],
        bounds: dict[str, tuple[float, float]] | None,
    ) -> Any:
        import sympy

        from ..inversion import InversionResult

        feature_names = self.feature_names or []
        expr = sympy.sympify(self._best_expression)
        symbols = {str(s): s for s in expr.free_symbols}

        from macon.common import unexpected

        free_name = free_params[0]
        free_sym = symbols.get(free_name)
        if unexpected(free_sym is None):
            raise ValueError(f"Symbol {free_name} not found in expression")

        for name, val in fixed_params.items():
            if name in symbols:
                arr = np.asarray(val)
                scalar_val = float(arr.flat[0]) if arr.ndim > 0 else float(arr)
                expr = expr.subs(symbols[name], scalar_val)

        y_val = float(y_target[0])
        solutions = sympy.solve(expr - y_val, free_sym)

        if unexpected(not solutions):
            raise ValueError("sympy.solve found no solutions")

        best = None
        for sol in solutions:
            sol_f = complex(sol)
            if sol_f.imag != 0:
                continue
            val = sol_f.real
            if bounds and free_name in bounds:
                lo, hi = bounds[free_name]
                if not (lo <= val <= hi):
                    continue
            best = val
            break

        if unexpected(best is None):
            best = float(solutions[0])
        assert best is not None

        from ..inversion import _build_feature_matrix, _resolve_indices

        free_indices, _ = _resolve_indices(feature_names, free_params, fixed_params)
        fixed_idx_val = {feature_names.index(k): v for k, v in fixed_params.items()}
        X = _build_feature_matrix(
            np.array([best]), free_indices, fixed_idx_val, len(feature_names), len(y_target),
        )
        y_pred = self.predict(X)
        y_scale = max(float(np.abs(y_target).mean()), 1e-10)
        rel_residual = float(np.sqrt(np.mean(((y_pred - y_target) / y_scale) ** 2)))

        return InversionResult(
            x_solution={free_name: best},
            y_predicted=y_pred,
            y_target=y_target,
            residual=rel_residual,
            success=True,
            message="Symbolic inversion via sympy.solve",
        )

    @property
    def metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": "symbolic",
            "feature_names": self.feature_names,
            "is_fitted": self._is_fitted,
            "n_training_samples": self._n_training_samples,
            "niterations": self._niterations,
            "binary_operators": self._binary_operators,
            "unary_operators": self._unary_operators,
            "maxsize": self._maxsize,
        }
        if self._is_fitted:
            result["best_expression"] = self._best_expression
        return result
