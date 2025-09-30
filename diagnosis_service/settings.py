import enum
from pathlib import Path
from tempfile import gettempdir
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

TEMP_DIR = Path(gettempdir())

class LogLevel(str, enum.Enum):
    """Possible log levels."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class Settings(BaseSettings):
    """
    Application settings.

    These parameters can be configured
    with environment variables.
    """

    host: str = "0.0.0.0"  # Listen on all interfaces for Cloud Run
    port: int = int("8080")  # Default port for Google Cloud Run, will be overridden by PORT env var if present
    # quantity of workers for uvicorn
    workers_count: int = 1
    # Enable uvicorn reloading
    reload: bool = False

    # Current environment
    environment: str = "dev"

    log_level: LogLevel = LogLevel.INFO

    # Grpc endpoint for opentelemetry.
    # E.G. http://localhost:4317
    opentelemetry_endpoint: Optional[str] = None

    hw_model_path: str = "diagnosis_service/models/inception_v3_digital_spiral_v1.onnx"
    vm_model_path: str = "diagnosis_service/models/model-vm_mdvr-kcl_lgbm_production.bin"

    google_api_key: str
    supabase_service_key: str
    supabase_url: str

    # CORS settings
    cors_origins: list[str] = ["*"]
    cors_methods: list[str] = ["*"]
    cors_headers: list[str] = ["*"]
    cors_credentials: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DIAGNOSIS_SERVICE_",
        env_file_encoding="utf-8",
    )


settings = Settings()
