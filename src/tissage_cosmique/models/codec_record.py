"""Pydantic models for codec records (tracking fitted codecs in the DB)."""

from datetime import datetime
from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class CodecRecordBase(BaseModel):
    """Shared fields for a codec record."""

    name: str = Field(description="Unique name for this codec")
    codec_type: str = Field(description="Codec type: pca, autoencoder")
    n_latent: int = Field(description="Latent space dimensionality")
    n_features: int = Field(description="Original output dimensionality")
    artifact_path: str = Field(description="Path to saved codec artifact")
    explained_variance: list[float] | None = Field(default=None, description="PCA variance ratios")
    reconstruction_loss: float | None = Field(default=None, description="Autoencoder reconstruction loss")
    metadata_json: dict[str, Any] | None = Field(default=None, description="Additional codec metadata")
    created_at: datetime | None = Field(default=None, description="Creation timestamp")


class CodecRecordCreate(CodecRecordBase):
    """Fields used to create a CodecRecord."""


class CodecRecord(CodecRecordBase):
    """CodecRecord response model."""

    model_config = ConfigDict(from_attributes=True)
    col_names_for_table: ClassVar[list[str]] = [
        "id_",
        "name",
        "codec_type",
        "n_latent",
        "n_features",
        "created_at",
    ]

    id_: UUID
