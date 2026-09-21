"""Pydantic models for tracer configurations (pyccl tracer definitions stored in DB)."""

from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class TracerConfigBase(BaseModel):
    """Shared fields for a tracer configuration."""

    name: str = Field(description="Unique name for this tracer")
    tracer_type: str = Field(
        description="Tracer type: weak_lensing, number_counts, cmb_lensing",
    )

    # Static (z, f(z)) arrays — used when no parametric/functional model is set
    dndz_z: list[float] | None = Field(default=None, description="Redshift array for dN/dz")
    dndz_nz: list[float] | None = Field(default=None, description="N(z) array for dN/dz")
    bias_z: list[float] | None = Field(default=None, description="Galaxy bias z array")
    bias_bz: list[float] | None = Field(default=None, description="Galaxy bias b(z) array")
    ia_bias_z: list[float] | None = Field(default=None, description="IA amplitude z array")
    ia_bias_az: list[float] | None = Field(default=None, description="IA amplitude A_IA(z) array")
    mag_bias_z: list[float] | None = Field(default=None, description="Magnification bias z array")
    mag_bias_sz: list[float] | None = Field(default=None, description="Magnification bias s(z) array")

    # Parametric nuisance models (override static arrays)
    bias_model: str | None = Field(
        default=None, description="Bias model: constant, linear, quadratic, power_law",
    )
    bias_params: dict[str, float] | None = Field(
        default=None, description="Bias model parameters {a0, a1, ...}",
    )
    ia_bias_model: str | None = Field(default=None, description="IA bias model name")
    ia_bias_params: dict[str, float] | None = Field(
        default=None, description="IA bias model parameters",
    )
    mag_bias_model: str | None = Field(default=None, description="Mag bias model name")
    mag_bias_params: dict[str, float] | None = Field(
        default=None, description="Mag bias model parameters",
    )

    # Functional nuisance models (z-node amplitudes, interpolated)
    bias_nodes_z: list[float] | None = Field(
        default=None, description="Bias z-nodes for functional interpolation",
    )
    bias_nodes_amp: list[float] | None = Field(
        default=None, description="Bias amplitudes at z-nodes",
    )
    ia_bias_nodes_z: list[float] | None = Field(default=None, description="IA z-nodes")
    ia_bias_nodes_amp: list[float] | None = Field(
        default=None, description="IA amplitudes at z-nodes",
    )
    mag_bias_nodes_z: list[float] | None = Field(default=None, description="Mag bias z-nodes")
    mag_bias_nodes_amp: list[float] | None = Field(
        default=None, description="Mag bias amplitudes at z-nodes",
    )

    # Boolean flags and scalar params
    has_shear: bool = Field(default=True, description="Include lensing shear")
    has_rsd: bool = Field(default=False, description="Include RSD")
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
        "bias_model",
        "ia_bias_model",
        "has_shear",
        "has_rsd",
        "z_source",
    ]

    id_: UUID
