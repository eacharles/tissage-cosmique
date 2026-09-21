"""ORM model for the DataVectorConfig table."""

import uuid

from pydantic import BaseModel
from sqlalchemy import JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from .. import models
from .base import Base
from .utils import uuid7 as _uuid7


class DataVectorConfigTable(Base):
    """ORM table for data vector configurations."""

    __tablename__ = "data_vector_config"

    id_: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=_uuid7)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    tracer_names: Mapped[list] = mapped_column(JSON)
    ell: Mapped[list] = mapped_column(JSON)
    include_pairs: Mapped[list | None] = mapped_column(JSON, nullable=True)
    ell_per_pair: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    bin_edges: Mapped[list | None] = mapped_column(JSON, nullable=True)
    bin_edges_per_pair: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    bin_weights: Mapped[str] = mapped_column(String(20), default="2ell+1")
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    @classmethod
    def pydantic_create_class(cls) -> type[BaseModel]:
        return models.DataVectorConfigCreate

    @classmethod
    def pydantic_model_class(cls) -> type[BaseModel]:
        return models.DataVectorConfig

    @classmethod
    def class_string(cls) -> str:
        return cls.__tablename__
