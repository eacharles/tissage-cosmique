"""Tests for tracked distance computations."""

import pytest

pyccl = pytest.importorskip("pyccl")

import numpy as np  # noqa: E402

from tissage_cosmique.computations.cosmology import make_pyccl_cosmology  # noqa: E402
from tissage_cosmique.computations.distances import comoving_angular_distance  # noqa: E402

PLANCK2018 = dict(
    name="Planck2018",
    Omega_c=0.2589,
    Omega_b=0.0486,
    h=0.6774,
    n_s=0.9667,
    sigma8=0.8159,
    A_s=None,
    Omega_k=0.0,
    w0=-1.0,
    wa=0.0,
)


class TestMakePycclCosmology:

    def test_creates_cosmology(self):
        cosmo = make_pyccl_cosmology(PLANCK2018)
        assert isinstance(cosmo, pyccl.Cosmology)

    def test_strips_non_pyccl_keys(self):
        params = {**PLANCK2018, "id_": "some-uuid", "extra_key": 42}
        cosmo = make_pyccl_cosmology(params)
        assert isinstance(cosmo, pyccl.Cosmology)

    def test_skips_none_values(self):
        params = dict(Omega_c=0.25, Omega_b=0.05, h=0.70, n_s=0.96, sigma8=0.81, A_s=None)
        cosmo = make_pyccl_cosmology(params)
        assert isinstance(cosmo, pyccl.Cosmology)


class TestComovingAngularDistance:

    def test_array_input(self):
        a = np.linspace(0.2, 1.0, 50)
        result = comoving_angular_distance(PLANCK2018, a)
        assert isinstance(result, np.ndarray)
        assert result.shape == (50,)

    def test_distance_zero_at_present(self):
        a = np.array([1.0])
        result = comoving_angular_distance(PLANCK2018, a)
        assert result[0] == 0.0

    def test_distance_increases_with_redshift(self):
        a = np.array([0.3, 0.5, 0.8, 1.0])
        result = comoving_angular_distance(PLANCK2018, a)
        for i in range(len(result) - 1):
            assert result[i] > result[i + 1]

    def test_scalar_input(self):
        result = comoving_angular_distance(PLANCK2018, np.array([0.5]))
        assert result.shape == (1,)
        assert result[0] > 0

    def test_flat_lcdm_defaults(self):
        params = dict(Omega_c=0.25, Omega_b=0.05, h=0.70, n_s=0.96, sigma8=0.81)
        a = np.array([0.5])
        result = comoving_angular_distance(params, a)
        assert result[0] > 0
