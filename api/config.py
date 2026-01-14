"""Configuration settings for the Briscola API."""

from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """API configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # API Settings
    api_title: str = "Briscola RL API"
    api_version: str = "1.0.0"
    debug: bool = False

    # CORS Settings
    cors_origins: List[str] = [
        "http://localhost:4200",
        "http://127.0.0.1:4200",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:80",
        "http://127.0.0.1:80",
    ]

    # Game Settings
    max_active_games: int = 100

    # Model Settings
    default_checkpoint_dir: str = "checkpoints"
    device: str = "auto"  # "cpu", "cuda", or "auto"

    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000

    def get_device(self) -> str:
        """Get the torch device to use."""
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


settings = Settings()
