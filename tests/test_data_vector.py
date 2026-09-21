"""Tests for data vector computation and result helpers."""

import pytest

pyccl = pytest.importorskip("pyccl")

import numpy as np  # noqa: E402

from tissage_cosmique.computations.data_vector import (  # noqa: E402
    compute_data_vector,
    get_data_vector_pairs,
    make_data_vector_result,
)

PLANCK = dict(
    Omega_c=0.2589, Omega_b=0.0486, h=0.6774,
    n_s=0.9667, sigma8=0.8159, Omega_k=0.0, w0=-1.0, wa=0.0,
)

Z = np.linspace(0.0, 2.0, 100).tolist()
NZ = np.exp(-0.5 * ((np.array(Z) - 0.5) / 0.15) ** 2).tolist()

TRACER_CONFIGS = {
    "wl": dict(
        tracer_type="weak_lensing",
        dndz_z=Z, dndz_nz=NZ, has_shear=True,
    ),
    "nc": dict(
        tracer_type="number_counts",
        dndz_z=Z, dndz_nz=NZ,
        bias_model="constant", bias_params={"a0": 1.5},
        has_rsd=False,
    ),
}

ELL = np.arange(10, 200).astype(float)


class TestGetDataVectorPairs:

    def test_all_pairs_two_tracers(self):
        pairs = get_data_vector_pairs(["wl", "nc"])
        assert pairs == [("wl", "wl"), ("wl", "nc"), ("nc", "nc")]

    def test_all_pairs_three_tracers(self):
        pairs = get_data_vector_pairs(["a", "b", "c"])
        assert len(pairs) == 6
        assert pairs[0] == ("a", "a")
        assert pairs[-1] == ("c", "c")

    def test_explicit_pairs(self):
        pairs = get_data_vector_pairs(
            ["wl", "nc"],
            include_pairs=[["wl", "nc"], ["nc", "nc"]],
        )
        assert pairs == [("wl", "nc"), ("nc", "nc")]


class TestComputeDataVector:

    def test_shape_all_pairs(self):
        dv_config = {
            "tracer_names": ["wl", "nc"],
            "include_pairs": None,
        }
        dv = compute_data_vector(PLANCK, TRACER_CONFIGS, dv_config, ELL)
        n_pairs = 3
        assert dv.shape == (n_pairs * len(ELL),)

    def test_shape_explicit_pairs(self):
        dv_config = {
            "tracer_names": ["wl", "nc"],
            "include_pairs": [["wl", "nc"]],
        }
        dv = compute_data_vector(PLANCK, TRACER_CONFIGS, dv_config, ELL)
        assert dv.shape == (1 * len(ELL),)

    def test_auto_spectra_positive(self):
        dv_config = {
            "tracer_names": ["wl", "nc"],
            "include_pairs": [["wl", "wl"], ["nc", "nc"]],
        }
        dv = compute_data_vector(PLANCK, TRACER_CONFIGS, dv_config, ELL)
        assert np.all(dv >= 0)


class TestDataVectorResult:

    def test_get_cl(self):
        dv_config = {
            "tracer_names": ["wl", "nc"],
            "include_pairs": None,
        }
        dv = compute_data_vector(PLANCK, TRACER_CONFIGS, dv_config, ELL)
        result = make_data_vector_result(dv, dv_config, ELL)

        cl_wl = result.get_cl("wl", "wl")
        assert cl_wl.shape == (len(ELL),)
        assert np.all(cl_wl >= 0)

        cl_cross = result.get_cl("wl", "nc")
        assert cl_cross.shape == (len(ELL),)

    def test_get_cl_symmetric(self):
        dv_config = {
            "tracer_names": ["wl", "nc"],
            "include_pairs": None,
        }
        dv = compute_data_vector(PLANCK, TRACER_CONFIGS, dv_config, ELL)
        result = make_data_vector_result(dv, dv_config, ELL)

        np.testing.assert_array_equal(
            result.get_cl("wl", "nc"),
            result.get_cl("nc", "wl"),
        )

    def test_get_cl_missing_raises(self):
        dv_config = {
            "tracer_names": ["wl"],
            "include_pairs": [["wl", "wl"]],
        }
        dv = compute_data_vector(PLANCK, TRACER_CONFIGS, dv_config, ELL)
        result = make_data_vector_result(dv, dv_config, ELL)

        with pytest.raises(KeyError):
            result.get_cl("wl", "nc")

    def test_reshape_matrix(self):
        dv_config = {
            "tracer_names": ["wl", "nc"],
            "include_pairs": None,
        }
        dv = compute_data_vector(PLANCK, TRACER_CONFIGS, dv_config, ELL)
        result = make_data_vector_result(dv, dv_config, ELL)

        matrix = result.reshape_matrix()
        assert matrix.shape == (3, len(ELL))
