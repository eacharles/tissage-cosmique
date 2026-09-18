"""Utilities for generating emulator training data from tracked computations."""

from collections.abc import Callable
from typing import Any

import numpy as np
from macon.common import unexpected


def build_training_data(
    computation_fn: Callable[..., np.ndarray],
    param_samples: list[dict[str, Any]],
    *grids: np.ndarray,
    param_names: list[str],
    grid_names: list[str] | None = None,
    result_axes_order: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a computation over a parameter grid and collect (X, y) training pairs.

    Supports computations with one or more grid variables (e.g., scale factor,
    wavenumber). Each training point maps to a single scalar output.

    Parameters
    ----------
    computation_fn
        A function with signature ``(params_dict, *grids) -> result_array``.
        The result can be 1-D (single grid) or N-D (multiple grids).
    param_samples
        List of parameter dictionaries, one per cosmology.
    *grids
        One or more 1-D arrays of grid values (e.g., ``a_grid`` or
        ``k_grid, a_grid``).
    param_names
        Ordered list of parameter keys to extract from each dict.
    grid_names
        Names for the grid columns in X (for metadata). Defaults to
        ``["grid_0", "grid_1", ...]``.
    result_axes_order
        Maps each result axis to the corresponding grid index. For example,
        ``pyccl.linear_matter_power(cosmo, k, a)`` returns shape ``(n_a, n_k)``
        — axis 0 corresponds to grid 1 (a) and axis 1 to grid 0 (k), so pass
        ``result_axes_order=[1, 0]``. Default: identity ``[0, 1, ..., n-1]``.

    Returns
    -------
    X
        Feature matrix of shape ``(n_total, len(param_names) + len(grids))``.
        Grid columns appear in the same order as ``*grids``.
    y
        Target array of shape ``(n_total,)``.

    Examples
    --------
    Single grid (distances)::

        X, y = build_training_data(
            comoving_angular_distance, samples, a_grid,
            param_names=["Omega_c", "h", "sigma8"],
        )
        # X columns: [Omega_c, h, sigma8, a]

    Two grids (power spectrum)::

        X, y = build_training_data(
            linear_matter_power, samples, k_grid, a_grid,
            param_names=["Omega_c", "h", "sigma8"],
            result_axes_order=[1, 0],  # result shape (n_a, n_k)
        )
        # X columns: [Omega_c, h, sigma8, k, a]
    """
    n_grids = len(grids)
    if unexpected(n_grids == 0):
        raise ValueError("At least one grid array is required")

    if result_axes_order is None:
        result_axes_order = list(range(n_grids))

    axis_to_grid = {axis: grid_idx for axis, grid_idx in enumerate(result_axes_order)}

    X_rows: list[list[float]] = []
    y_values: list[float] = []

    for params in param_samples:
        result = computation_fn(params, *grids)
        result = np.asarray(result)
        pvals = [params[k] for k in param_names]

        if unexpected(result.ndim == 0):
            X_rows.append(pvals + [float(g[0]) for g in grids])
            y_values.append(float(result))
        elif result.ndim == 1 and n_grids == 1:
            for i, val in enumerate(result):
                X_rows.append(pvals + [float(grids[0][i])])
                y_values.append(float(val))
        else:
            for idx in np.ndindex(result.shape):
                grid_vals = [0.0] * n_grids
                for axis, pos in enumerate(idx):
                    grid_idx = axis_to_grid[axis]
                    grid_vals[grid_idx] = float(grids[grid_idx][pos])
                X_rows.append(pvals + grid_vals)
                y_values.append(float(result[idx]))

    return np.array(X_rows), np.array(y_values)


def params_to_feature_matrix(
    cosmo_params: dict[str, Any],
    *grids: np.ndarray,
    param_names: list[str],
    result_axes_order: list[int] | None = None,
) -> np.ndarray:
    """Build a feature matrix for one parameter set over one or more grids.

    Parameters
    ----------
    cosmo_params
        Cosmological parameter dictionary.
    *grids
        One or more 1-D grid arrays.
    param_names
        Ordered list of parameter keys (must match training order).
    result_axes_order
        Same as in :func:`build_training_data`. Determines how grid
        combinations are enumerated to match the flattened result order.

    Returns
    -------
    np.ndarray
        Feature matrix of shape ``(n_total, len(param_names) + len(grids))``.
    """
    n_grids = len(grids)
    if unexpected(n_grids == 0):
        raise ValueError("At least one grid array is required")

    if result_axes_order is None:
        result_axes_order = list(range(n_grids))

    axis_to_grid = {axis: grid_idx for axis, grid_idx in enumerate(result_axes_order)}

    pvals = [cosmo_params[k] for k in param_names]

    if n_grids == 1:
        n = len(grids[0])
        X = np.empty((n, len(param_names) + 1))
        X[:, :-1] = pvals
        X[:, -1] = grids[0]
        return X

    grid_shapes = tuple(len(grids[result_axes_order[ax]]) for ax in range(n_grids))
    n_total = int(np.prod(grid_shapes))
    X = np.empty((n_total, len(param_names) + n_grids))
    X[:, :len(param_names)] = pvals

    row = 0
    for idx in np.ndindex(grid_shapes):
        for axis, pos in enumerate(idx):
            grid_idx = axis_to_grid[axis]
            X[row, len(param_names) + grid_idx] = grids[grid_idx][pos]
        row += 1

    return X
