"""Tests for DataVectorConfig CRUD operations."""

from tissage_cosmique.local_sync import data_vector_config


class TestDataVectorConfigCRUD:

    def test_create_and_get(self):
        row = data_vector_config.create_row(
            name="3x2pt_v1",
            tracer_names=["source_bin1", "lens_bin1", "cmb_kappa"],
            ell=list(range(10, 500)),
        )
        assert row.name == "3x2pt_v1"
        assert row.tracer_names == ["source_bin1", "lens_bin1", "cmb_kappa"]
        assert len(row.ell) == 490

        fetched = data_vector_config.get_row(row.id_)
        assert fetched.tracer_names == row.tracer_names

    def test_create_with_explicit_pairs(self):
        row = data_vector_config.create_row(
            name="2x2pt_v1",
            tracer_names=["wl", "nc"],
            ell=list(range(10, 200)),
            include_pairs=[["wl", "wl"], ["wl", "nc"]],
        )
        assert row.include_pairs == [["wl", "wl"], ["wl", "nc"]]

    def test_update(self):
        row = data_vector_config.create_row(
            name="dv_update_test",
            tracer_names=["wl"],
            ell=[10, 20, 30],
        )
        data_vector_config.update_row(row.id_, ell=[10, 20, 30, 40, 50])
        updated = data_vector_config.get_row(row.id_)
        assert len(updated.ell) == 5

    def test_delete(self):
        row = data_vector_config.create_row(
            name="dv_delete_test",
            tracer_names=["wl"],
            ell=[10, 20],
        )
        data_vector_config.delete_row(row.id_)
        rows = data_vector_config.get_rows()
        assert not any(r.id_ == row.id_ for r in rows)
