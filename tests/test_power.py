"""Tests for tracked linear matter power spectrum computation."""

import pytest

pyccl = pytest.importorskip("pyccl")
pytest.importorskip("camb")

import numpy as np  # noqa: E402

from tissage_cosmique.computations.power import linear_matter_power  # noqa: E402

PLANCK2018 = dict(
    name="Planck2018",
    Omega_c=0.2589,
    Omega_b=0.0486,
    h=0.6774,
    n_s=0.9667,
    sigma8=0.8159,
    Omega_k=0.0,
    w0=-1.0,
    wa=0.0,
)


class TestLinearMatterPower:

    def test_array_output(self):
        k = np.logspace(-3, 0, 20)
        pk = linear_matter_power(PLANCK2018, k, 1.0)
        assert isinstance(pk, np.ndarray)
        assert pk.shape == (20,)

    def test_positive_values(self):
        k = np.logspace(-3, 0, 20)
        pk = linear_matter_power(PLANCK2018, k, 1.0)
        assert np.all(pk > 0)

    def test_growth_with_scale_factor(self):
        k = np.logspace(-2, -1, 10)
        pk_today = linear_matter_power(PLANCK2018, k, 1.0)
        pk_past = linear_matter_power(PLANCK2018, k, 0.5)
        assert np.all(pk_today > pk_past)

    def test_peak_shape(self):
        k = np.logspace(-4, 1, 100)
        pk = linear_matter_power(PLANCK2018, k, 1.0)
        peak_idx = np.argmax(pk)
        assert 0 < peak_idx < len(pk) - 1

    def test_minimal_params(self):
        params = dict(Omega_c=0.25, Omega_b=0.05, h=0.70, n_s=0.96, sigma8=0.81)
        k = np.logspace(-2, 0, 10)
        pk = linear_matter_power(params, k, 1.0)
        assert pk.shape == (10,)
        assert np.all(pk > 0)
