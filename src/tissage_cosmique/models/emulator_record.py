"""Pydantic models for emulator records (tracking trained emulators in the DB)."""

from datetime import datetime
from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class EmulatorRecordBase(BaseModel):
    """Shared fields for an emulator record."""

    name: str = Field(description="Unique name for this emulator")
    emulator_type: str = Field(description="Backend type: gp, tensorflow, pytorch, symbolic, latent")
    function_name: str = Field(description="Computation function this emulator replaces")
    param_names: list[str] = Field(description="Parameter names used for training")
    feature_names: list[str] | None = Field(default=None, description="Full feature names (direct emulators)")
    artifact_path: str = Field(description="Path to saved emulator artifact")
    training_score: float | None = Field(default=None, description="Training R2 or similar metric")
    n_training_samples: int = Field(default=0, description="Number of training points")
    param_bounds: dict[str, list[float]] | None = Field(
        default=None, description="Parameter ranges {name: [lo, hi]}",
    )
    grid_config: dict[str, list[float]] | None = Field(
        default=None, description="Grid arrays used for training",
    )
    execution_ids: list[str] | None = Field(
        default=None, description="Tisserande execution IDs",
    )
    codec_id: UUID | None = Field(default=None, description="FK to CodecRecord (latent emulators)")
    metadata_json: dict[str, Any] | None = Field(default=None, description="Additional emulator metadata")
    created_at: datetime | None = Field(default=None, description="Creation timestamp")


class EmulatorRecordCreate(EmulatorRecordBase):
    """Fields used to create an EmulatorRecord."""


class EmulatorRecord(EmulatorRecordBase):
    """EmulatorRecord response model."""

    model_config = ConfigDict(from_attributes=True)
    col_names_for_table: ClassVar[list[str]] = [
        "id_",
        "name",
        "emulator_type",
        "function_name",
        "training_score",
        "n_training_samples",
        "created_at",
    ]

    id_: UUID
