"""Pydantic models for cosmological parameter sets."""

from typing import ClassVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class CosmologyParamsBase(BaseModel):
    """Shared fields for a cosmological parameter set."""

    name: str = Field(description="Unique name for this parameter set")
    Omega_c: float = Field(description="Cold dark matter density fraction")
    Omega_b: float = Field(description="Baryon density fraction")
    h: float = Field(description="Hubble constant / 100 km/s/Mpc")
    n_s: float = Field(description="Scalar spectral index")
    sigma8: float | None = Field(default=None, description="RMS matter fluctuations at 8 Mpc/h")
    A_s: float | None = Field(default=None, description="Scalar amplitude of primordial power spectrum")
    Omega_k: float = Field(default=0.0, description="Curvature density fraction")
    w0: float = Field(default=-1.0, description="Dark energy equation of state parameter w0")
    wa: float = Field(default=0.0, description="Dark energy equation of state parameter wa")


class CosmologyParamsCreate(CosmologyParamsBase):
    """Fields used to create a CosmologyParams record."""


class CosmologyParams(CosmologyParamsBase):
    """CosmologyParams response model."""

    model_config = ConfigDict(from_attributes=True)
    col_names_for_table: ClassVar[list[str]] = [
        "id_",
        "name",
        "Omega_c",
        "Omega_b",
        "h",
        "n_s",
        "sigma8",
        "w0",
        "wa",
    ]

    id_: UUID
