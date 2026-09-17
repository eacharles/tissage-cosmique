"""Build emulator training data from tisserande provenance stored in the database."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID

import numpy as np
from macon.models.filtering import Filter, FilterOp


def run_tracked_batch(
    computation_fn: Callable[..., np.ndarray],
    param_samples: list[dict[str, Any]],
    a_grid: np.ndarray,
    *,
    db_url: str | None = None,
) -> list[UUID]:
    """Run tracked computations for a batch of parameter sets.

    Configures tisserande tracking (if not already configured), then calls
    ``computation_fn(params, a_grid)`` for each parameter set. The ``@track``
    decorator on the computation function stores provenance automatically.

    Parameters
    ----------
    computation_fn
        A ``@track``-decorated function with signature ``(params_dict, a_array) -> array``.
    param_samples
        List of cosmological parameter dictionaries.
    a_grid
        Scale factor grid to evaluate at.
    db_url
        Database URL for tracking storage. If None, uses tisserande's default config.

    Returns
    -------
    list[UUID]
        Execution IDs for each computation run.
    """
    from tisserande.tracking.backends import LocalSyncBackend
    from tisserande.tracking.decorator import configure, get_backend

    if get_backend() is None:
        configure(db_url=db_url) if db_url is not None else configure(backend=LocalSyncBackend())

    from tisserande.local_sync import execution as exec_ops

    initial_count = exec_ops.count_rows()
    for params in param_samples:
        computation_fn(params, a_grid)

    all_execs = exec_ops.get_rows()
    return [e.id_ for e in all_execs[initial_count:]]


def query_computation_results(
    param_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Query stored computation results and build (X, y) training pairs.

    Reconstructs training data from tisserande's provenance graph: for each
    successful execution, extracts the input cosmological parameters (from
    ``config_dict`` nodes) and scale factors (from ``array`` nodes with
    ``arg_name='a'``), along with the output distances (from ``array`` nodes
    with ``arg_name='return'``).

    Parameters
    ----------
    param_names
        Ordered list of cosmological parameter keys to extract from stored
        config_dict nodes (e.g., ``["Omega_c", "h", "sigma8"]``).

    Returns
    -------
    X
        Feature matrix of shape ``(n_total, len(param_names) + 1)``.
    y
        Target array of shape ``(n_total,)``.
    """
    from tisserande.local_sync import edge as edge_ops
    from tisserande.local_sync import execution as exec_ops
    from tisserande.local_sync import node as node_ops

    execs = exec_ops.filter_rows(
        [Filter(field="status", op=FilterOp.EQ, value="success")],
    )

    X_rows: list[list[float]] = []
    y_values: list[float] = []

    for exec_rec in execs:
        if exec_rec.function_node_id is None:
            continue

        input_edges = edge_ops.filter_rows([
            Filter(field="to_id", op=FilterOp.EQ, value=exec_rec.function_node_id),
            Filter(field="execution_id", op=FilterOp.EQ, value=exec_rec.id_),
        ])
        output_edges = edge_ops.filter_rows([
            Filter(field="from_id", op=FilterOp.EQ, value=exec_rec.function_node_id),
            Filter(field="execution_id", op=FilterOp.EQ, value=exec_rec.id_),
        ])

        input_nodes = [node_ops.get_row(e.from_id) for e in input_edges]
        output_nodes = [node_ops.get_row(e.to_id) for e in output_edges]

        cosmo_params: dict[str, Any] | None = None
        a_array: list[float] | None = None
        result_array: list[float] | None = None

        for n in input_nodes:
            if n.arg_name == "cosmo_params" and hasattr(n, "config_data"):
                cosmo_params = n.config_data
            elif n.arg_name == "a" and hasattr(n, "value_json"):
                a_array = n.value_json

        for n in output_nodes:
            if n.arg_name == "return" and hasattr(n, "value_json"):
                result_array = n.value_json

        if cosmo_params is None or a_array is None or result_array is None:
            continue

        pvals = [cosmo_params[k] for k in param_names]
        for i, a_val in enumerate(a_array):
            X_rows.append(pvals + [float(a_val)])
            y_values.append(float(result_array[i]))

    return np.array(X_rows), np.array(y_values)
