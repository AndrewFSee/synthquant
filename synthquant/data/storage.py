"""Parquet-based market data storage."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["Storage", "ParquetStorage"]


class Storage(ABC):
    """Abstract base class for market data storage backends."""

    @abstractmethod
    def save(self, symbol: str, df: pd.DataFrame) -> None:
        """Persist a DataFrame for the given symbol.

        Args:
            symbol: Ticker symbol used as the storage key.
            df: OHLCV DataFrame to persist.
        """
        ...

    @abstractmethod
    def load(self, symbol: str) -> pd.DataFrame:
        """Load a previously stored DataFrame.

        Args:
            symbol: Ticker symbol to load.

        Returns:
            Stored OHLCV DataFrame.

        Raises:
            KeyError: If symbol is not found in storage.
        """
        ...

    @abstractmethod
    def list_symbols(self) -> list[str]:
        """Return the list of stored symbols.

        Returns:
            Sorted list of symbol strings.
        """
        ...


class ParquetStorage(Storage):
    """Local Parquet-based storage for OHLCV DataFrames.

    Files are stored as `<base_dir>/<symbol>.parquet`.

    Args:
        base_dir: Root directory for Parquet files. Created if it does not exist.
    """

    def __init__(self, base_dir: str | Path = "data/market") -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"ParquetStorage initialized at {self._base_dir}")

    def _path(self, symbol: str) -> Path:
        return self._base_dir / f"{symbol.upper()}.parquet"

    def save(self, symbol: str, df: pd.DataFrame) -> None:
        """Save DataFrame to Parquet.

        Args:
            symbol: Ticker symbol.
            df: OHLCV DataFrame with DatetimeIndex.
        """
        path = self._path(symbol)
        df.to_parquet(path, engine="pyarrow", compression="snappy")
        logger.info(f"Saved {symbol} to {path} ({len(df)} rows)")

    def load(self, symbol: str) -> pd.DataFrame:
        """Load DataFrame from Parquet.

        Args:
            symbol: Ticker symbol.

        Returns:
            Stored DataFrame.

        Raises:
            KeyError: If no Parquet file exists for the symbol.
        """
        path = self._path(symbol)
        if not path.exists():
            raise KeyError(f"No stored data for symbol '{symbol}' at {path}")
        df = pd.read_parquet(path, engine="pyarrow")
        logger.info(f"Loaded {symbol} from {path} ({len(df)} rows)")
        return df

    def list_symbols(self) -> list[str]:
        """List symbols available in storage.

        Returns:
            Sorted list of symbol strings (filename stems uppercased).
        """
        symbols = sorted(p.stem.upper() for p in self._base_dir.glob("*.parquet"))
        logger.debug(f"Found {len(symbols)} symbols in {self._base_dir}")
        return symbols

    def delete(self, symbol: str) -> None:
        """Delete stored data for a symbol.

        Args:
            symbol: Ticker symbol to delete.
        """
        path = self._path(symbol)
        if path.exists():
            path.unlink()
            logger.info(f"Deleted {path}")
        else:
            logger.warning(f"No file to delete for symbol '{symbol}'")
