"""Integration tests: tracked computations → DB → training data reconstruction."""

import pytest

pyccl = pytest.importorskip("pyccl")

import numpy as np  # noqa: E402

from tissage_cosmique.computations.distances import comoving_angular_distance  # noqa: E402
from tissage_cosmique.emulators import (  # noqa: E402
    build_training_data,
    query_computation_results,
    run_tracked_batch,
)

PARAM_NAMES = ["Omega_c", "h", "sigma8"]
FIXED_PARAMS = dict(Omega_b=0.0486, n_s=0.9667, Omega_k=0.0, w0=-1.0, wa=0.0)
A_GRID = np.linspace(0.3, 0.7, 5)


def _make_param_samples(n: int, rng: np.random.Generator) -> list[dict]:
    return [
        {"Omega_c": rng.uniform(0.22, 0.32), "h": rng.uniform(0.62, 0.75),
         "sigma8": rng.uniform(0.77, 0.87), **FIXED_PARAMS}
        for _ in range(n)
    ]


class TestRunTrackedBatch:

    def test_returns_execution_ids(self):
        rng = np.random.default_rng(42)
        samples = _make_param_samples(3, rng)
        exec_ids = run_tracked_batch(comoving_angular_distance, samples, A_GRID)
        assert len(exec_ids) == 3

    def test_executions_stored_in_db(self):
        from tisserande.local_sync import execution

        rng = np.random.default_rng(42)
        samples = _make_param_samples(3, rng)
        run_tracked_batch(comoving_angular_distance, samples, A_GRID)
        execs = execution.get_rows()
        assert len(execs) >= 3


class TestQueryComputationResults:

    def test_roundtrip_matches_direct_computation(self):
        rng = np.random.default_rng(99)
        samples = _make_param_samples(5, rng)
        run_tracked_batch(comoving_angular_distance, samples, A_GRID)

        X_db, y_db = query_computation_results(PARAM_NAMES)
        X_direct, y_direct = build_training_data(
            comoving_angular_distance, samples, A_GRID, param_names=PARAM_NAMES,
        )

        assert X_db.shape == X_direct.shape
        assert y_db.shape == y_direct.shape
        np.testing.assert_allclose(X_db, X_direct, rtol=1e-10)
        np.testing.assert_allclose(y_db, y_direct, rtol=1e-10)

    def test_empty_db_returns_empty(self):
        X, y = query_computation_results(PARAM_NAMES)
        assert X.shape == (0,)
        assert y.shape == (0,)

    def test_query_preserves_param_order(self):
        rng = np.random.default_rng(77)
        samples = _make_param_samples(2, rng)
        run_tracked_batch(comoving_angular_distance, samples, A_GRID)

        X_db, _ = query_computation_results(PARAM_NAMES)
        for i, sample in enumerate(samples):
            row_start = i * len(A_GRID)
            for j, pname in enumerate(PARAM_NAMES):
                assert X_db[row_start, j] == pytest.approx(sample[pname])
