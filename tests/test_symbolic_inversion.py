"""Tests for symbolic emulator inversion via sympy."""

import numpy as np

from tissage_cosmique.emulators.backends.symbolic import SymbolicEmulator
from tissage_cosmique.emulators.inversion import InversionResult


class TestSymbolicInversion:
    """Test the sympy inversion path using a mock-fitted SymbolicEmulator."""

    def _make_mock_emulator(self) -> SymbolicEmulator:
        """Create a SymbolicEmulator with a known expression, without running PySR."""
        emu = SymbolicEmulator(feature_names=["x", "y"])
        emu._best_expression = "2*x + 3*y"
        emu._is_fitted = True
        emu._n_training_samples = 100

        from unittest.mock import MagicMock

        from sklearn.preprocessing import StandardScaler

        emu._x_scaler = StandardScaler()
        emu._x_scaler.mean_ = np.array([0.0, 0.0])
        emu._x_scaler.scale_ = np.array([1.0, 1.0])
        emu._x_scaler.var_ = np.array([1.0, 1.0])
        emu._x_scaler.n_features_in_ = 2
        emu._x_scaler.n_samples_seen_ = 100

        mock_model = MagicMock()
        mock_model.predict = lambda X: 2 * X[:, 0] + 3 * X[:, 1]
        emu._model = mock_model
        return emu

    def test_sympy_solve_single_param(self):
        emu = self._make_mock_emulator()
        result = emu.invert(
            y_target=np.array([11.0]),
            free_params=["x"],
            fixed_params={"y": 1.0},
        )
        assert isinstance(result, InversionResult)
        assert abs(result.x_solution["x"] - 4.0) < 0.01

    def test_sympy_solve_with_bounds(self):
        emu = self._make_mock_emulator()
        result = emu.invert(
            y_target=np.array([11.0]),
            free_params=["x"],
            fixed_params={"y": 1.0},
            bounds={"x": (0.0, 10.0)},
        )
        assert isinstance(result, InversionResult)
        assert abs(result.x_solution["x"] - 4.0) < 0.01

    def test_falls_back_for_multi_param(self):
        emu = self._make_mock_emulator()
        result = emu.invert(
            y_target=np.array([11.0]),
            free_params=["x", "y"],
            fixed_params={},
            x0={"x": 1.0, "y": 1.0},
            bounds={"x": (0.0, 10.0), "y": (0.0, 10.0)},
        )
        assert isinstance(result, InversionResult)

    def test_sympy_invert_returns_result(self):
        emu = self._make_mock_emulator()
        result = emu.invert(
            y_target=np.array([11.0]),
            free_params=["x"],
            fixed_params={"y": 1.0},
        )
        assert isinstance(result, InversionResult)
        assert result.success
