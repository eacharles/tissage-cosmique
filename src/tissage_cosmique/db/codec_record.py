"""ORM model for the CodecRecord table."""

import uuid
from datetime import datetime

from pydantic import BaseModel
from sqlalchemy import DateTime, Float, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from .. import models
from .base import Base
from .utils import uuid7 as _uuid7


class CodecRecordTable(Base):
    """ORM table for codec records."""

    __tablename__ = "codec_record"

    id_: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=_uuid7)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    codec_type: Mapped[str] = mapped_column(String(50))
    n_latent: Mapped[int] = mapped_column(Integer)
    n_features: Mapped[int] = mapped_column(Integer)
    artifact_path: Mapped[str] = mapped_column(String(1024))
    explained_variance: Mapped[list | None] = mapped_column(JSON, nullable=True)
    reconstruction_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    @classmethod
    def pydantic_create_class(cls) -> type[BaseModel]:
        return models.CodecRecordCreate

    @classmethod
    def pydantic_model_class(cls) -> type[BaseModel]:
        return models.CodecRecord

    @classmethod
    def class_string(cls) -> str:
        return cls.__tablename__
