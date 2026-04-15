from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    OUTPUT_DIR: Path = Path("outputs")
    FACES_DIR: Path = Path("outputs/faces")
    MATCHED_DIR: Path = Path("outputs/faces/matched")
    UNMATCHED_DIR: Path = Path("outputs/faces/unmatched")
    REFERENCE_DIR: Path = Path("outputs/faces/reference")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure output directories exist
for directory in [
    settings.OUTPUT_DIR,
    settings.FACES_DIR,
    settings.MATCHED_DIR,
    settings.UNMATCHED_DIR,
    settings.REFERENCE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
