"""Backend configuration resolved from environment variables and repo layout."""

from __future__ import annotations

from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SEISMIC_", extra="ignore")

    repo_root: Path = REPO_ROOT
    data_dir: Path = REPO_ROOT / "data"
    embeddings_dir: Path = REPO_ROOT / "embeddings"
    weights_dir: Path = REPO_ROOT / "weights"
    embedders_dir: Path = REPO_ROOT / "embedders"
    preprocessing_script: Path = REPO_ROOT / "preprocessing" / "preprocess.py"
    sgz_pool_size: int = 4

    @field_validator(
        "repo_root",
        "data_dir",
        "embeddings_dir",
        "weights_dir",
        "embedders_dir",
        "preprocessing_script",
        mode="after",
    )
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        return value.resolve()

    def weights_available(self) -> bool:
        return (self.weights_dir / "config.json").is_file() and (
            self.weights_dir / "model.safetensors"
        ).is_file()


settings = Settings()
