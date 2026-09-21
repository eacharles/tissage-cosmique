"""Tests for tracked angular power spectrum computation and tracer helpers."""

import pytest

pyccl = pytest.importorskip("pyccl")

import numpy as np  # noqa: E402

from tissage_cosmique.computations.angular_cl import angular_cl  # noqa: E402
from tissage_cosmique.computations.cosmology import make_pyccl_cosmology  # noqa: E402
from tissage_cosmique.computations.tracers import make_pyccl_tracer  # noqa: E402

PLANCK = dict(
    Omega_c=0.2589, Omega_b=0.0486, h=0.6774,
    n_s=0.9667, sigma8=0.8159, Omega_k=0.0, w0=-1.0, wa=0.0,
)

Z = np.linspace(0.0, 2.0, 100).tolist()
NZ = np.exp(-0.5 * ((np.array(Z) - 0.5) / 0.15) ** 2).tolist()
BIAS_BZ = [1.2] * 100

WL_CONFIG = dict(
    name="wl_test", tracer_type="weak_lensing",
    dndz_z=Z, dndz_nz=NZ, has_shear=True,
)

NC_CONFIG = dict(
    name="nc_test", tracer_type="number_counts",
    dndz_z=Z, dndz_nz=NZ, bias_z=Z, bias_bz=BIAS_BZ, has_rsd=False,
)

CMB_CONFIG = dict(
    name="cmb_test", tracer_type="cmb_lensing", z_source=1100.0,
)

ELL = np.arange(10, 500).astype(float)


class TestMakePycclTracer:

    def test_weak_lensing(self):
        cosmo = make_pyccl_cosmology(PLANCK)
        tracer = make_pyccl_tracer(cosmo, WL_CONFIG)
        assert tracer is not None

    def test_number_counts(self):
        cosmo = make_pyccl_cosmology(PLANCK)
        tracer = make_pyccl_tracer(cosmo, NC_CONFIG)
        assert tracer is not None

    def test_cmb_lensing(self):
        cosmo = make_pyccl_cosmology(PLANCK)
        tracer = make_pyccl_tracer(cosmo, CMB_CONFIG)
        assert tracer is not None

    def test_unknown_type_raises(self):
        cosmo = make_pyccl_cosmology(PLANCK)
        with pytest.raises(ValueError, match="Unknown tracer type"):
            make_pyccl_tracer(cosmo, {"tracer_type": "bogus"})


class TestAngularCl:

    def test_wl_auto(self):
        cl = angular_cl(PLANCK, WL_CONFIG, WL_CONFIG, ELL)
        assert cl.shape == ELL.shape
        assert np.all(cl >= 0)

    def test_nc_auto(self):
        cl = angular_cl(PLANCK, NC_CONFIG, NC_CONFIG, ELL)
        assert cl.shape == ELL.shape
        assert np.all(cl >= 0)

    def test_wl_nc_cross(self):
        cl = angular_cl(PLANCK, WL_CONFIG, NC_CONFIG, ELL)
        assert cl.shape == ELL.shape

    def test_cmb_auto(self):
        cl = angular_cl(PLANCK, CMB_CONFIG, CMB_CONFIG, ELL)
        assert cl.shape == ELL.shape
        assert np.all(cl >= 0)

    def test_wl_cmb_cross(self):
        cl = angular_cl(PLANCK, WL_CONFIG, CMB_CONFIG, ELL)
        assert cl.shape == ELL.shape
