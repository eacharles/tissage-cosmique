"""Pydantic models for tracer configurations (pyccl tracer definitions stored in DB)."""

from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class TracerConfigBase(BaseModel):
    """Shared fields for a tracer configuration."""

    name: str = Field(description="Unique name for this tracer")
    tracer_type: str = Field(description="Tracer type: weak_lensing, number_counts, cmb_lensing")
    dndz_z: list[float] | None = Field(default=None, description="Redshift array for dN/dz")
    dndz_nz: list[float] | None = Field(default=None, description="N(z) array for dN/dz")
    bias_z: list[float] | None = Field(default=None, description="Galaxy bias redshift array")
    bias_bz: list[float] | None = Field(default=None, description="Galaxy bias b(z) array")
    ia_bias_z: list[float] | None = Field(default=None, description="IA amplitude redshift array")
    ia_bias_az: list[float] | None = Field(default=None, description="IA amplitude A_IA(z) array")
    mag_bias_z: list[float] | None = Field(default=None, description="Magnification bias redshift array")
    mag_bias_sz: list[float] | None = Field(default=None, description="Magnification bias s(z) array")
    has_shear: bool = Field(default=True, description="Include lensing shear (WeakLensing)")
    has_rsd: bool = Field(default=False, description="Include RSD (NumberCounts)")
    use_A_ia: bool = Field(default=True, description="Use conventional IA normalization")
    z_source: float | None = Field(default=None, description="Source redshift (CMBLensing)")
    n_samples: int = Field(default=256, description="Kernel resolution samples")
    metadata_json: dict[str, Any] | None = Field(default=None, description="Additional metadata")


class TracerConfigCreate(TracerConfigBase):
    """Fields used to create a TracerConfig."""


class TracerConfig(TracerConfigBase):
    """TracerConfig response model."""

    model_config = ConfigDict(from_attributes=True)
    col_names_for_table: ClassVar[list[str]] = [
        "id_",
        "name",
        "tracer_type",
        "has_shear",
        "has_rsd",
        "z_source",
    ]

    id_: UUID
