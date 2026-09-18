"""Tests for CodecRecord CRUD operations."""

from datetime import UTC, datetime

from tissage_cosmique.local_sync import codec_record


class TestCodecRecordCRUD:

    def test_create_and_get(self):
        row = codec_record.create_row(
            name="pca_distances_v1",
            codec_type="pca",
            n_latent=3,
            n_features=30,
            artifact_path="/tmp/pca_v1.joblib",
            explained_variance=[0.993, 0.006, 0.001],
            created_at=datetime.now(UTC),
        )
        assert row.name == "pca_distances_v1"
        assert row.codec_type == "pca"
        assert row.n_latent == 3
        assert row.id_ is not None

        fetched = codec_record.get_row(row.id_)
        assert fetched.artifact_path == "/tmp/pca_v1.joblib"
        assert fetched.explained_variance == [0.993, 0.006, 0.001]

    def test_create_autoencoder(self):
        row = codec_record.create_row(
            name="ae_power_v1",
            codec_type="autoencoder",
            n_latent=5,
            n_features=60,
            artifact_path="/tmp/ae_v1.joblib",
            reconstruction_loss=0.00038,
            metadata_json={"hidden_layers": [64, 32], "n_epochs": 500},
        )
        assert row.codec_type == "autoencoder"
        assert row.reconstruction_loss == 0.00038
        assert row.metadata_json["hidden_layers"] == [64, 32]

    def test_update(self):
        row = codec_record.create_row(
            name="pca_update_test",
            codec_type="pca",
            n_latent=5,
            n_features=30,
            artifact_path="/tmp/old.joblib",
        )
        codec_record.update_row(row.id_, artifact_path="/tmp/new.joblib")
        updated = codec_record.get_row(row.id_)
        assert updated.artifact_path == "/tmp/new.joblib"

    def test_delete(self):
        row = codec_record.create_row(
            name="pca_delete_test",
            codec_type="pca",
            n_latent=3,
            n_features=30,
            artifact_path="/tmp/del.joblib",
        )
        codec_record.delete_row(row.id_)
        rows = codec_record.get_rows()
        assert not any(r.id_ == row.id_ for r in rows)
