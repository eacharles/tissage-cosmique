"""ORM model for the EmulatorRecord table."""

import uuid
from datetime import datetime

from pydantic import BaseModel
from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from .. import models
from .base import Base
from .utils import uuid7 as _uuid7


class EmulatorRecordTable(Base):
    """ORM table for emulator records."""

    __tablename__ = "emulator_record"

    id_: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=_uuid7)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    emulator_type: Mapped[str] = mapped_column(String(50))
    function_name: Mapped[str] = mapped_column(String(255))
    param_names: Mapped[list] = mapped_column(JSON)
    feature_names: Mapped[list | None] = mapped_column(JSON, nullable=True)
    artifact_path: Mapped[str] = mapped_column(String(1024))
    training_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    n_training_samples: Mapped[int] = mapped_column(Integer, default=0)
    param_bounds: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    grid_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    execution_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    codec_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("codec_record.id_"), nullable=True,
    )
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    @classmethod
    def pydantic_create_class(cls) -> type[BaseModel]:
        return models.EmulatorRecordCreate

    @classmethod
    def pydantic_model_class(cls) -> type[BaseModel]:
        return models.EmulatorRecord

    @classmethod
    def class_string(cls) -> str:
        return cls.__tablename__
