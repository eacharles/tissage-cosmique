"""Pydantic models for data vector configurations."""

from typing import Any, ClassVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DataVectorConfigBase(BaseModel):
    """Shared fields for a data vector configuration."""

    name: str = Field(description="Unique name for this data vector config")
    tracer_names: list[str] = Field(description="Ordered list of tracer config names")
    ell: list[float] = Field(description="Default shared multipole array")
    include_pairs: list[list[str]] | None = Field(
        default=None,
        description="Explicit tracer pairs [[t1, t2], ...], or None for all",
    )

    # Per-pair ℓ ranges (override shared ell)
    ell_per_pair: dict[str, list[float]] | None = Field(
        default=None,
        description="Per-pair ell arrays, keyed by 't1-t2'",
    )

    # ℓ binning
    bin_edges: list[float] | None = Field(
        default=None, description="Shared bin edges for bandpower binning",
    )
    bin_edges_per_pair: dict[str, list[float]] | None = Field(
        default=None,
        description="Per-pair bin edges, keyed by 't1-t2'",
    )
    bin_weights: str = Field(
        default="2ell+1",
        description="Binning weight scheme: '2ell+1' or 'uniform'",
    )

    metadata_json: dict[str, Any] | None = Field(
        default=None, description="Additional metadata",
    )


class DataVectorConfigCreate(DataVectorConfigBase):
    """Fields used to create a DataVectorConfig."""


class DataVectorConfig(DataVectorConfigBase):
    """DataVectorConfig response model."""

    model_config = ConfigDict(from_attributes=True)
    col_names_for_table: ClassVar[list[str]] = [
        "id_",
        "name",
        "tracer_names",
    ]

    id_: UUID
