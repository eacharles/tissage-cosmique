"""Tests for EmulatorRecord CRUD operations."""

from datetime import UTC, datetime

from tissage_cosmique.local_sync import codec_record, emulator_record


class TestEmulatorRecordCRUD:

    def test_create_and_get(self):
        row = emulator_record.create_row(
            name="gp_distances_v1",
            emulator_type="gp",
            function_name="comoving_angular_distance",
            param_names=["Omega_c", "h", "sigma8"],
            feature_names=["Omega_c", "h", "sigma8", "a"],
            artifact_path="/tmp/gp_v1.joblib",
            training_score=0.9999,
            n_training_samples=900,
            param_bounds={"Omega_c": [0.22, 0.32], "h": [0.62, 0.75]},
            created_at=datetime.now(UTC),
        )
        assert row.name == "gp_distances_v1"
        assert row.function_name == "comoving_angular_distance"
        assert row.param_names == ["Omega_c", "h", "sigma8"]
        assert row.id_ is not None

        fetched = emulator_record.get_row(row.id_)
        assert fetched.training_score == 0.9999
        assert fetched.param_bounds["Omega_c"] == [0.22, 0.32]

    def test_create_latent_with_codec_fk(self):
        codec = codec_record.create_row(
            name="pca_for_latent_test",
            codec_type="pca",
            n_latent=3,
            n_features=30,
            artifact_path="/tmp/pca.joblib",
        )
        row = emulator_record.create_row(
            name="latent_distances_v1",
            emulator_type="latent",
            function_name="comoving_angular_distance",
            param_names=["Omega_c", "h", "sigma8"],
            artifact_path="/tmp/latent_v1/",
            n_training_samples=50,
            codec_id=codec.id_,
        )
        assert row.codec_id == codec.id_
        assert row.emulator_type == "latent"

    def test_create_with_execution_ids(self):
        row = emulator_record.create_row(
            name="gp_tracked_v1",
            emulator_type="gp",
            function_name="comoving_angular_distance",
            param_names=["Omega_c", "h", "sigma8"],
            artifact_path="/tmp/gp_tracked.joblib",
            execution_ids=["uuid-1", "uuid-2", "uuid-3"],
            n_training_samples=90,
        )
        assert row.execution_ids == ["uuid-1", "uuid-2", "uuid-3"]

    def test_update(self):
        row = emulator_record.create_row(
            name="gp_update_test",
            emulator_type="gp",
            function_name="comoving_angular_distance",
            param_names=["Omega_c"],
            artifact_path="/tmp/old.joblib",
        )
        emulator_record.update_row(row.id_, training_score=0.9998)
        updated = emulator_record.get_row(row.id_)
        assert updated.training_score == 0.9998

    def test_delete(self):
        row = emulator_record.create_row(
            name="gp_delete_test",
            emulator_type="gp",
            function_name="test_fn",
            param_names=["x"],
            artifact_path="/tmp/del.joblib",
        )
        emulator_record.delete_row(row.id_)
        rows = emulator_record.get_rows()
        assert not any(r.id_ == row.id_ for r in rows)

    def test_multiple_and_get_rows(self):
        emulator_record.create_row(
            name="emu_multi_1",
            emulator_type="gp",
            function_name="fn_a",
            param_names=["x"],
            artifact_path="/tmp/1.joblib",
        )
        emulator_record.create_row(
            name="emu_multi_2",
            emulator_type="pytorch",
            function_name="fn_b",
            param_names=["x", "y"],
            artifact_path="/tmp/2.joblib",
        )
        rows = emulator_record.get_rows()
        assert len(rows) >= 2
