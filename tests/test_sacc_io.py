"""Tests for SACC file export."""

import pytest

pyccl = pytest.importorskip("pyccl")
sacc_mod = pytest.importorskip("sacc")

import numpy as np  # noqa: E402

from tissage_cosmique.computations.covariance import compute_covariance_from_data_vector  # noqa: E402
from tissage_cosmique.computations.data_vector import make_data_vector_result  # noqa: E402
from tissage_cosmique.computations.sacc_io import build_sacc, load_sacc, load_sacc_to_db, save_sacc  # noqa: E402

PLANCK = dict(
    Omega_c=0.2589, Omega_b=0.0486, h=0.6774,
    n_s=0.9667, sigma8=0.8159, Omega_k=0.0, w0=-1.0, wa=0.0,
)

Z = np.linspace(0.0, 2.0, 100).tolist()
NZ = np.exp(-0.5 * ((np.array(Z) - 0.5) / 0.15) ** 2).tolist()
ELL = np.arange(10, 200).astype(float)

TRACER_CONFIGS = {
    "wl": dict(
        tracer_type="weak_lensing",
        dndz_z=Z, dndz_nz=NZ, has_shear=True,
        n_gal=10.0, sigma_epsilon=0.26,
    ),
    "nc": dict(
        tracer_type="number_counts",
        dndz_z=Z, dndz_nz=NZ,
        bias_model="constant", bias_params={"a0": 1.5},
        has_rsd=False, n_gal=30.0,
    ),
}


class TestBuildSacc:

    def test_creates_sacc_object(self):
        dv_config = {"tracer_names": ["wl", "nc"]}
        dv, dv_result, cov = compute_covariance_from_data_vector(
            PLANCK, TRACER_CONFIGS, dv_config, ELL, f_sky=0.3,
        )
        dv_result = make_data_vector_result(dv, dv_config, ELL)
        s = build_sacc(dv_result, TRACER_CONFIGS, cov)
        assert s is not None
        assert len(s.mean) == len(dv)

    def test_tracers_registered(self):
        dv_config = {"tracer_names": ["wl", "nc"]}
        dv, dv_result, _ = compute_covariance_from_data_vector(
            PLANCK, TRACER_CONFIGS, dv_config, ELL, f_sky=0.3,
        )
        dv_result = make_data_vector_result(dv, dv_config, ELL)
        s = build_sacc(dv_result, TRACER_CONFIGS)
        assert "wl" in s.tracers
        assert "nc" in s.tracers

    def test_data_types_correct(self):
        dv_config = {"tracer_names": ["wl", "nc"]}
        dv, dv_result, _ = compute_covariance_from_data_vector(
            PLANCK, TRACER_CONFIGS, dv_config, ELL, f_sky=0.3,
        )
        dv_result = make_data_vector_result(dv, dv_config, ELL)
        s = build_sacc(dv_result, TRACER_CONFIGS)
        dtypes = set(d.data_type for d in s.data)
        assert "galaxy_shear_cl_ee" in dtypes
        assert "galaxy_density_cl" in dtypes

    def test_covariance_attached(self):
        dv_config = {"tracer_names": ["wl"]}
        dv, dv_result, cov = compute_covariance_from_data_vector(
            PLANCK, TRACER_CONFIGS, dv_config, ELL, f_sky=0.3,
        )
        dv_result = make_data_vector_result(dv, dv_config, ELL)
        s = build_sacc(dv_result, TRACER_CONFIGS, cov)
        assert s.covariance is not None

    def test_no_covariance(self):
        dv_config = {"tracer_names": ["wl"]}
        dv, dv_result, _ = compute_covariance_from_data_vector(
            PLANCK, TRACER_CONFIGS, dv_config, ELL, f_sky=0.3,
        )
        dv_result = make_data_vector_result(dv, dv_config, ELL)
        s = build_sacc(dv_result, TRACER_CONFIGS)
        assert s.covariance is None


class TestSaveSacc:

    def test_save_and_reload(self, tmp_path):
        dv_config = {"tracer_names": ["wl", "nc"]}
        dv, dv_result, cov = compute_covariance_from_data_vector(
            PLANCK, TRACER_CONFIGS, dv_config, ELL, f_sky=0.3,
        )
        dv_result = make_data_vector_result(dv, dv_config, ELL)

        path = str(tmp_path / "test.sacc")
        save_sacc(path, dv_result, TRACER_CONFIGS, cov)

        import sacc

        loaded = sacc.Sacc.load_fits(path)
        assert len(loaded.mean) == len(dv)
        assert loaded.covariance is not None
        np.testing.assert_allclose(loaded.mean, dv, rtol=1e-6)


class TestLoadSacc:

    def _make_sacc_file(self, tmp_path):
        dv_config = {"tracer_names": ["wl", "nc"]}
        dv, dv_result, cov = compute_covariance_from_data_vector(
            PLANCK, TRACER_CONFIGS, dv_config, ELL, f_sky=0.3,
        )
        dv_result = make_data_vector_result(dv, dv_config, ELL)
        path = str(tmp_path / "roundtrip.sacc")
        save_sacc(path, dv_result, TRACER_CONFIGS, cov)
        return path, dv, cov

    def test_load_extracts_tracers(self, tmp_path):
        path, _, _ = self._make_sacc_file(tmp_path)
        configs, pairs, ell_dict, cl_dict, cov = load_sacc(path)
        assert "wl" in configs
        assert "nc" in configs
        assert configs["wl"]["tracer_type"] in ("weak_lensing", "number_counts")
        assert "dndz_z" in configs["wl"]

    def test_load_extracts_pairs(self, tmp_path):
        path, _, _ = self._make_sacc_file(tmp_path)
        _, pairs, ell_dict, cl_dict, _ = load_sacc(path)
        assert len(pairs) >= 2
        assert len(ell_dict) >= 2
        assert len(cl_dict) >= 2

    def test_load_extracts_covariance(self, tmp_path):
        path, dv, cov_orig = self._make_sacc_file(tmp_path)
        _, _, _, _, cov_loaded = load_sacc(path)
        assert cov_loaded is not None
        assert cov_loaded.shape == cov_orig.shape

    def test_roundtrip_cl_values(self, tmp_path):
        path, dv, _ = self._make_sacc_file(tmp_path)
        _, _, _, cl_dict, _ = load_sacc(path)
        all_cl = np.concatenate(list(cl_dict.values()))
        assert len(all_cl) == len(dv)
        np.testing.assert_allclose(sorted(all_cl), sorted(dv), rtol=1e-6)


class TestLoadSaccToDb:

    def test_stores_tracers(self, tmp_path):
        dv_config = {"tracer_names": ["wl", "nc"]}
        dv, dv_result, cov = compute_covariance_from_data_vector(
            PLANCK, TRACER_CONFIGS, dv_config, ELL, f_sky=0.3,
        )
        dv_result = make_data_vector_result(dv, dv_config, ELL)
        path = str(tmp_path / "todb.sacc")
        save_sacc(path, dv_result, TRACER_CONFIGS, cov)

        summary = load_sacc_to_db(path, prefix="test_")
        assert "wl" in summary["tracer_names"]
        assert "nc" in summary["tracer_names"]
        assert summary["n_data_points"] > 0

        from tissage_cosmique.local_sync import tracer_config

        rows = tracer_config.get_rows()
        names = [r.name for r in rows]
        assert "test_wl" in names
        assert "test_nc" in names
