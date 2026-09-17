"""Tests for the hot-swap registry, dispatch, and context manager."""

import pytest

pyccl = pytest.importorskip("pyccl")

import numpy as np  # noqa: E402

from tissage_cosmique.computations.distances import comoving_angular_distance  # noqa: E402
from tissage_cosmique.emulators import GPEmulator, build_training_data, params_to_feature_matrix  # noqa: E402
from tissage_cosmique.swap import (  # noqa: E402
    disable,
    disable_all,
    enable,
    enable_all,
    get_entry,
    is_registered,
    list_registered,
    original,
    register,
    reset,
    swappable,
    unregister,
)

PARAM_NAMES = ["Omega_c", "h", "sigma8"]
A_GRID = np.linspace(0.2, 0.8, 10)
FIXED_PARAMS = dict(Omega_b=0.0486, n_s=0.9667, Omega_k=0.0, w0=-1.0, wa=0.0)
TEST_PARAMS = dict(Omega_c=0.27, h=0.68, sigma8=0.81, **FIXED_PARAMS)


def _make_param_samples(n: int, rng: np.random.Generator) -> list[dict]:
    return [
        {"Omega_c": rng.uniform(0.22, 0.32), "h": rng.uniform(0.62, 0.75),
         "sigma8": rng.uniform(0.77, 0.87), **FIXED_PARAMS}
        for _ in range(n)
    ]


@pytest.fixture(scope="module")
def fitted_emulator():
    rng = np.random.default_rng(42)
    samples = _make_param_samples(30, rng)
    X, y = build_training_data(comoving_angular_distance, samples, A_GRID, PARAM_NAMES)
    emu = GPEmulator(feature_names=PARAM_NAMES + ["a"])
    emu.fit(X, y)
    return emu


@pytest.fixture(autouse=True)
def _clean_registry():
    reset()
    yield
    reset()


class TestRegistry:

    def test_register_and_lookup(self, fitted_emulator):
        register(comoving_angular_distance, fitted_emulator, PARAM_NAMES)
        assert is_registered(comoving_angular_distance)
        entry = get_entry(comoving_angular_distance)
        assert entry is not None
        assert entry.emulator is fitted_emulator
        assert entry.param_names == PARAM_NAMES
        assert entry.enabled is True

    def test_unregister(self, fitted_emulator):
        register(comoving_angular_distance, fitted_emulator, PARAM_NAMES)
        unregister(comoving_angular_distance)
        assert not is_registered(comoving_angular_distance)

    def test_enable_disable(self, fitted_emulator):
        register(comoving_angular_distance, fitted_emulator, PARAM_NAMES)
        disable(comoving_angular_distance)
        assert get_entry(comoving_angular_distance).enabled is False
        enable(comoving_angular_distance)
        assert get_entry(comoving_angular_distance).enabled is True

    def test_enable_disable_all(self, fitted_emulator):
        register(comoving_angular_distance, fitted_emulator, PARAM_NAMES)
        disable_all()
        assert get_entry(comoving_angular_distance).enabled is False
        enable_all()
        assert get_entry(comoving_angular_distance).enabled is True

    def test_list_registered(self, fitted_emulator):
        register(comoving_angular_distance, fitted_emulator, PARAM_NAMES)
        listing = list_registered()
        assert len(listing) == 1
        assert comoving_angular_distance.__qualname__ in listing

    def test_reset(self, fitted_emulator):
        register(comoving_angular_distance, fitted_emulator, PARAM_NAMES)
        reset()
        assert not is_registered(comoving_angular_distance)


class TestSwappable:

    def test_falls_through_when_not_registered(self):
        compute = swappable(comoving_angular_distance)
        result = compute(TEST_PARAMS, A_GRID)
        truth = comoving_angular_distance(TEST_PARAMS, A_GRID)
        np.testing.assert_array_equal(result, truth)

    def test_routes_through_emulator(self, fitted_emulator):
        compute = swappable(comoving_angular_distance)
        register(comoving_angular_distance, fitted_emulator, PARAM_NAMES)
        result = compute(TEST_PARAMS, A_GRID)
        X = params_to_feature_matrix(TEST_PARAMS, A_GRID, PARAM_NAMES)
        expected = fitted_emulator.predict(X)
        np.testing.assert_array_equal(result, expected)

    def test_falls_through_when_disabled(self, fitted_emulator):
        compute = swappable(comoving_angular_distance)
        register(comoving_angular_distance, fitted_emulator, PARAM_NAMES, enabled=False)
        result = compute(TEST_PARAMS, A_GRID)
        truth = comoving_angular_distance(TEST_PARAMS, A_GRID)
        np.testing.assert_array_equal(result, truth)

    def test_preserves_function_name(self):
        compute = swappable(comoving_angular_distance)
        assert compute.__name__ == comoving_angular_distance.__name__

    def test_original_attribute(self):
        compute = swappable(comoving_angular_distance)
        assert compute._original is comoving_angular_distance


class TestOriginalContext:

    def test_bypasses_emulator(self, fitted_emulator):
        compute = swappable(comoving_angular_distance)
        register(comoving_angular_distance, fitted_emulator, PARAM_NAMES)

        with original(comoving_angular_distance):
            result = compute(TEST_PARAMS, A_GRID)

        truth = comoving_angular_distance(TEST_PARAMS, A_GRID)
        np.testing.assert_array_equal(result, truth)

    def test_restores_enabled_after_context(self, fitted_emulator):
        register(comoving_angular_distance, fitted_emulator, PARAM_NAMES)
        assert get_entry(comoving_angular_distance).enabled is True

        with original(comoving_angular_distance):
            assert get_entry(comoving_angular_distance).enabled is False

        assert get_entry(comoving_angular_distance).enabled is True

    def test_noop_when_not_registered(self):
        with original(comoving_angular_distance):
            pass


class TestAccuracy:

    def test_swap_matches_direct_emulator_predict(self, fitted_emulator):
        compute = swappable(comoving_angular_distance)
        register(comoving_angular_distance, fitted_emulator, PARAM_NAMES)

        swap_result = compute(TEST_PARAMS, A_GRID)
        X = params_to_feature_matrix(TEST_PARAMS, A_GRID, PARAM_NAMES)
        direct_result = fitted_emulator.predict(X)

        np.testing.assert_array_equal(swap_result, direct_result)
