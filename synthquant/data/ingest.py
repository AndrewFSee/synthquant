"""Market data ingestion from various sources."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date, datetime

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["DataSource", "YFinanceSource", "DataIngestor"]


class DataSource(ABC):
    """Abstract base class for market data sources.

    All data sources must return OHLCV DataFrames with a DatetimeIndex
    and columns: ['open', 'high', 'low', 'close', 'volume'].
    """

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        start: str | date | datetime,
        end: str | date | datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol.

        Args:
            symbol: Ticker symbol (e.g., 'SPY').
            start: Start date.
            end: End date.
            interval: Data frequency ('1d', '1h', etc.).

        Returns:
            DataFrame with DatetimeIndex and OHLCV columns.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the data source."""
        ...


class YFinanceSource(DataSource):
    """Yahoo Finance data source using yfinance.

    Args:
        auto_adjust: Whether to auto-adjust prices for splits/dividends.
    """

    def __init__(self, auto_adjust: bool = True) -> None:
        self._auto_adjust = auto_adjust

    @property
    def name(self) -> str:
        """Name of this data source."""
        return "yahoo_finance"

    def fetch(
        self,
        symbol: str,
        start: str | date | datetime,
        end: str | date | datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance.

        Args:
            symbol: Ticker symbol.
            start: Start date.
            end: End date.
            interval: Data frequency.

        Returns:
            DataFrame with normalized OHLCV columns.

        Raises:
            ImportError: If yfinance is not installed.
            ValueError: If data fetch fails or returns empty DataFrame.
        """
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError(
                "yfinance is required for YFinanceSource. "
                "Install with: pip install synthquant[data]"
            ) from e

        logger.info(f"Fetching {symbol} from Yahoo Finance ({start} to {end})")
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start,
            end=end,
            interval=interval,
            auto_adjust=self._auto_adjust,
        )

        if df.empty:
            raise ValueError(f"No data returned for symbol '{symbol}'")

        return self._normalize(df)

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame to standard OHLCV format."""
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.columns = df.columns.str.lower()
        cols = ["open", "high", "low", "close", "volume"]
        available = [c for c in cols if c in df.columns]
        return df[available]


class DataIngestor:
    """Orchestrates data ingestion from multiple sources.

    Args:
        sources: List of DataSource instances to use (in priority order).
    """

    def __init__(self, sources: list[DataSource] | None = None) -> None:
        self._sources = sources or [YFinanceSource()]

    def fetch(
        self,
        symbol: str,
        start: str | date | datetime,
        end: str | date | datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch data, trying each source in order.

        Args:
            symbol: Ticker symbol.
            start: Start date.
            end: End date.
            interval: Data frequency.

        Returns:
            Normalized OHLCV DataFrame.

        Raises:
            RuntimeError: If all sources fail.
        """
        errors: list[str] = []
        for source in self._sources:
            try:
                df = source.fetch(symbol, start, end, interval)
                logger.info(f"Successfully fetched {symbol} from {source.name}")
                return df
            except Exception as e:
                logger.warning(f"Source {source.name} failed for {symbol}: {e}")
                errors.append(f"{source.name}: {e}")

        raise RuntimeError(
            f"All data sources failed for {symbol}:\n" + "\n".join(errors)
        )

    def fetch_multiple(
        self,
        symbols: list[str],
        start: str | date | datetime,
        end: str | date | datetime,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols.

        Args:
            symbols: List of ticker symbols.
            start: Start date.
            end: End date.
            interval: Data frequency.

        Returns:
            Dict mapping symbol to OHLCV DataFrame.
        """
        results: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                results[symbol] = self.fetch(symbol, start, end, interval)
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
        return results
