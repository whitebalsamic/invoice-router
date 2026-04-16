from typing import ClassVar, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..infrastructure.filesystem.paths import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TEMP_DIR,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
        populate_by_name=True,
        env_file=".env",
        env_file_encoding="utf-8",
    )

    ENV_ALIASES: ClassVar[dict[str, str]] = {
        "invoice_input_dir": "INVOICE_INPUT_DIR",
        "redis_url": "REDIS_URL",
        "database_url": "DATABASE_URL",
        "analysis_database_url": "ANALYSIS_DATABASE_URL",
        "benchmark_database_url": "BENCHMARK_DATABASE_URL",
        "dataset_root": "DATASET_ROOT",
        "temp_dir": "TEMP_DIR",
        "output_dir": "OUTPUT_DIR",
        "redis_max_connections": "REDIS_MAX_CONNECTIONS",
        "tessdata_prefix": "TESSDATA_PREFIX",
    }

    invoice_input_dir: Optional[str] = Field(None, alias="INVOICE_INPUT_DIR")
    redis_url: Optional[str] = Field(None, alias="REDIS_URL")
    database_url: Optional[str] = Field(None, alias="DATABASE_URL")
    analysis_database_url: Optional[str] = Field(None, alias="ANALYSIS_DATABASE_URL")
    benchmark_database_url: Optional[str] = Field(None, alias="BENCHMARK_DATABASE_URL")
    dataset_root: str = Field(str(DEFAULT_DATASET_ROOT), alias="DATASET_ROOT")
    temp_dir: str = Field(str(DEFAULT_TEMP_DIR), alias="TEMP_DIR")
    output_dir: str = Field(str(DEFAULT_OUTPUT_DIR), alias="OUTPUT_DIR")
    redis_max_connections: int = Field(50, alias="REDIS_MAX_CONNECTIONS")
    tessdata_prefix: Optional[str] = Field(None, alias="TESSDATA_PREFIX")

    @property
    def primary_database_target(self) -> str:
        self.require("database_url")
        assert self.database_url is not None
        return self.database_url

    @property
    def analysis_database_target(self) -> str:
        self.require("database_url")
        assert self.database_url is not None
        return self.analysis_database_url or self.database_url

    def missing_fields(self, *fields: str) -> list[str]:
        missing = []
        for field in fields:
            value = getattr(self, field)
            if value is None:
                missing.append(field)
                continue
            if isinstance(value, str) and not value.strip():
                missing.append(field)
        return missing

    def require(self, *fields: str) -> "Settings":
        missing = self.missing_fields(*fields)
        if missing:
            aliases = [self.ENV_ALIASES.get(field, field.upper()) for field in missing]
            raise ValueError(
                "Missing or empty required environment variables: " + ", ".join(aliases)
            )
        return self
