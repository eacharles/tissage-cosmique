"""ORM model for the TracerConfig table."""

import uuid

from pydantic import BaseModel
from sqlalchemy import Boolean, Float, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from .. import models
from .base import Base
from .utils import uuid7 as _uuid7


class TracerConfigTable(Base):
    """ORM table for tracer configurations."""

    __tablename__ = "tracer_config"

    id_: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=_uuid7)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    tracer_type: Mapped[str] = mapped_column(String(50))

    # Static arrays
    dndz_z: Mapped[list | None] = mapped_column(JSON, nullable=True)
    dndz_nz: Mapped[list | None] = mapped_column(JSON, nullable=True)
    dndz_model: Mapped[str | None] = mapped_column(String(50), nullable=True)
    dndz_params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    dndz_basis_z: Mapped[list | None] = mapped_column(JSON, nullable=True)
    dndz_basis_amp: Mapped[list | None] = mapped_column(JSON, nullable=True)
    bias_z: Mapped[list | None] = mapped_column(JSON, nullable=True)
    bias_bz: Mapped[list | None] = mapped_column(JSON, nullable=True)
    ia_bias_z: Mapped[list | None] = mapped_column(JSON, nullable=True)
    ia_bias_az: Mapped[list | None] = mapped_column(JSON, nullable=True)
    mag_bias_z: Mapped[list | None] = mapped_column(JSON, nullable=True)
    mag_bias_sz: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Parametric nuisance models
    bias_model: Mapped[str | None] = mapped_column(String(50), nullable=True)
    bias_params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    ia_bias_model: Mapped[str | None] = mapped_column(String(50), nullable=True)
    ia_bias_params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    mag_bias_model: Mapped[str | None] = mapped_column(String(50), nullable=True)
    mag_bias_params: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Functional nuisance models (z-nodes)
    bias_nodes_z: Mapped[list | None] = mapped_column(JSON, nullable=True)
    bias_nodes_amp: Mapped[list | None] = mapped_column(JSON, nullable=True)
    ia_bias_nodes_z: Mapped[list | None] = mapped_column(JSON, nullable=True)
    ia_bias_nodes_amp: Mapped[list | None] = mapped_column(JSON, nullable=True)
    mag_bias_nodes_z: Mapped[list | None] = mapped_column(JSON, nullable=True)
    mag_bias_nodes_amp: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Flags and scalars
    has_shear: Mapped[bool] = mapped_column(Boolean, default=True)
    has_rsd: Mapped[bool] = mapped_column(Boolean, default=False)
    use_A_ia: Mapped[bool] = mapped_column(Boolean, default=True)
    z_source: Mapped[float | None] = mapped_column(Float, nullable=True)
    n_samples: Mapped[int] = mapped_column(Integer, default=256)
    n_gal: Mapped[float | None] = mapped_column(Float, nullable=True)
    sigma_epsilon: Mapped[float | None] = mapped_column(Float, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    @classmethod
    def pydantic_create_class(cls) -> type[BaseModel]:
        return models.TracerConfigCreate

    @classmethod
    def pydantic_model_class(cls) -> type[BaseModel]:
        return models.TracerConfig

    @classmethod
    def class_string(cls) -> str:
        return cls.__tablename__
