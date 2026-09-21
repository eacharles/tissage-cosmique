"""Tests for TracerConfig CRUD operations."""

import numpy as np

from tissage_cosmique.local_sync import tracer_config

Z = np.linspace(0.0, 2.0, 50).tolist()
NZ = np.exp(-0.5 * ((np.array(Z) - 0.5) / 0.15) ** 2).tolist()


class TestTracerConfigCRUD:

    def test_create_weak_lensing(self):
        row = tracer_config.create_row(
            name="wl_source_bin1",
            tracer_type="weak_lensing",
            dndz_z=Z,
            dndz_nz=NZ,
            has_shear=True,
        )
        assert row.name == "wl_source_bin1"
        assert row.tracer_type == "weak_lensing"
        assert row.has_shear is True
        assert len(row.dndz_z) == 50

    def test_create_number_counts(self):
        bias_bz = [1.2] * 50
        row = tracer_config.create_row(
            name="nc_lens_bin1",
            tracer_type="number_counts",
            dndz_z=Z,
            dndz_nz=NZ,
            bias_z=Z,
            bias_bz=bias_bz,
            has_rsd=False,
        )
        assert row.tracer_type == "number_counts"
        assert row.bias_bz == bias_bz

    def test_create_cmb_lensing(self):
        row = tracer_config.create_row(
            name="cmb_kappa",
            tracer_type="cmb_lensing",
            z_source=1100.0,
        )
        assert row.tracer_type == "cmb_lensing"
        assert row.z_source == 1100.0
        assert row.dndz_z is None

    def test_get_and_update(self):
        row = tracer_config.create_row(
            name="wl_update_test",
            tracer_type="weak_lensing",
            dndz_z=Z,
            dndz_nz=NZ,
        )
        tracer_config.update_row(row.id_, n_samples=512)
        updated = tracer_config.get_row(row.id_)
        assert updated.n_samples == 512

    def test_delete(self):
        row = tracer_config.create_row(
            name="wl_delete_test",
            tracer_type="weak_lensing",
            dndz_z=Z,
            dndz_nz=NZ,
        )
        tracer_config.delete_row(row.id_)
        rows = tracer_config.get_rows()
        assert not any(r.id_ == row.id_ for r in rows)

    def test_multiple_tracers(self):
        tracer_config.create_row(name="multi_wl", tracer_type="weak_lensing", dndz_z=Z, dndz_nz=NZ)
        tracer_config.create_row(
            name="multi_nc", tracer_type="number_counts", dndz_z=Z, dndz_nz=NZ, has_rsd=True,
        )
        tracer_config.create_row(name="multi_cmb", tracer_type="cmb_lensing", z_source=1100.0)
        rows = tracer_config.get_rows()
        assert len(rows) >= 3
