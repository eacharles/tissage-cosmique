"""Configuration for tissage-cosmique, loaded from environment variables."""

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfiguration(BaseModel):
    """Database configuration for tissage-cosmique."""

    url: str = Field(
        default="sqlite+aiosqlite:///tissage_cosmique.db",
        description="The URL for the tissage-cosmique database",
    )

    echo: bool = Field(
        default=False,
        description="SQLAlchemy engine echo setting",
    )


class EmulatorConfiguration(BaseModel):
    """Configuration for emulator training and inference."""

    enabled: bool = Field(
        default=False,
        description="Enable emulator hot-swap for computations",
    )


class Configuration(BaseSettings):
    """Configuration for tissage-cosmique.

    Nested models may be consumed from environment variables named according to
    the pattern 'TISSAGE_COSMIQUE__NESTED__FIELD'.
    """

    model_config = SettingsConfigDict(
        env_prefix="TISSAGE_COSMIQUE__",
        env_nested_delimiter="__",
        nested_model_default_partial_update=True,
        case_sensitive=False,
        extra="ignore",
    )

    db: DatabaseConfiguration = DatabaseConfiguration()
    emulator: EmulatorConfiguration = EmulatorConfiguration()


_config: Configuration | None = None


def get_config() -> Configuration:
    """Get the configuration, creating it lazily on first access."""
    global _config
    if _config is None:
        _config = Configuration()
    return _config
