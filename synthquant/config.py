"""Global configuration and settings using pydantic-settings."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """SynthQuant global settings.

    All settings can be overridden via environment variables prefixed with SYNTHQUANT_.

    Example:
        SYNTHQUANT_N_PATHS=50000 python my_script.py
    """

    model_config = SettingsConfigDict(
        env_prefix="SYNTHQUANT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Simulation defaults
    n_paths: int = Field(default=10_000, description="Default number of Monte Carlo paths")
    time_horizon: float = Field(default=1.0, description="Default time horizon in years")
    dt: float = Field(default=1 / 252, description="Default time step (daily)")
    random_seed: int | None = Field(default=None, description="Global random seed")

    # Data settings
    data_cache_dir: Path = Field(
        default=Path.home() / ".synthquant" / "cache",
        description="Local cache directory for market data",
    )
    default_symbols: list[str] = Field(
        default=["SPY", "QQQ", "GLD", "TLT"],
        description="Default symbols to use in examples",
    )

    # API keys (loaded from environment)
    polygon_api_key: str | None = Field(default=None, description="Polygon.io API key")
    alpha_vantage_api_key: str | None = Field(default=None, description="Alpha Vantage API key")

    # API server settings
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    api_workers: int = Field(default=1, description="Number of API workers")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )

    def configure_logging(self) -> None:
        """Configure the root logger based on settings."""
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get the global settings singleton.

    Returns:
        Settings: The global settings instance.
    """
    settings = Settings()
    settings.configure_logging()
    return settings
