"""ORM model for the CosmologyParams table."""

import uuid

from pydantic import BaseModel
from sqlalchemy import Float, String
from sqlalchemy.orm import Mapped, mapped_column

from .. import models
from .base import Base
from .utils import uuid7 as _uuid7


class CosmologyParamsTable(Base):
    """ORM table for cosmological parameter sets."""

    __tablename__ = "cosmology_params"

    id_: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=_uuid7)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    Omega_c: Mapped[float] = mapped_column(Float)
    Omega_b: Mapped[float] = mapped_column(Float)
    h: Mapped[float] = mapped_column(Float)
    n_s: Mapped[float] = mapped_column(Float)
    sigma8: Mapped[float | None] = mapped_column(Float, nullable=True)
    A_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    Omega_k: Mapped[float] = mapped_column(Float, default=0.0)
    w0: Mapped[float] = mapped_column(Float, default=-1.0)
    wa: Mapped[float] = mapped_column(Float, default=0.0)

    @classmethod
    def pydantic_create_class(cls) -> type[BaseModel]:
        return models.CosmologyParamsCreate

    @classmethod
    def pydantic_model_class(cls) -> type[BaseModel]:
        return models.CosmologyParams

    @classmethod
    def class_string(cls) -> str:
        return cls.__tablename__
