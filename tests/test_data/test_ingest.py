"""Tests for data ingestion module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from synthquant.data.ingest import DataIngestor, DataSource, YFinanceSource


class ConcreteSource(DataSource):
    """Minimal concrete DataSource for testing."""

    @property
    def name(self) -> str:
        return "test_source"

    def fetch(self, symbol, start, end, interval="1d"):
        return pd.DataFrame()


def test_data_source_is_abstract() -> None:
    """DataSource cannot be instantiated directly (abstract class)."""
    with pytest.raises(TypeError):
        DataSource()  # type: ignore[abstract]


def test_concrete_source_can_be_instantiated() -> None:
    """A concrete DataSource subclass can be instantiated."""
    src = ConcreteSource()
    assert src.name == "test_source"


def test_yfinance_source_name() -> None:
    """YFinanceSource.name returns 'yahoo_finance'."""
    src = YFinanceSource()
    assert src.name == "yahoo_finance"


def test_data_ingestor_raises_when_all_sources_fail() -> None:
    """DataIngestor.fetch raises RuntimeError if every source fails."""

    class FailingSource(DataSource):
        @property
        def name(self) -> str:
            return "failing"

        def fetch(self, symbol, start, end, interval="1d"):
            raise ConnectionError("Network unavailable")

    ingestor = DataIngestor(sources=[FailingSource()])
    with pytest.raises(RuntimeError, match="All data sources failed"):
        ingestor.fetch("SPY", "2022-01-01", "2023-01-01")


def test_data_ingestor_fetch_multiple_handles_partial_failures() -> None:
    """fetch_multiple returns successful symbols even if some fail."""
    n = 10
    good_df = pd.DataFrame(
        {"close": [100.0] * n},
        index=pd.date_range("2022-01-01", periods=n, freq="B"),
    )

    class PartialSource(DataSource):
        @property
        def name(self) -> str:
            return "partial"

        def fetch(self, symbol, start, end, interval="1d"):
            if symbol == "GOOD":
                return good_df
            raise ValueError(f"No data for {symbol}")

    ingestor = DataIngestor(sources=[PartialSource()])
    results = ingestor.fetch_multiple(["GOOD", "BAD"], start="2022-01-01", end="2023-01-01")
    assert "GOOD" in results
    assert "BAD" not in results
    assert len(results["GOOD"]) == n


def test_data_ingestor_uses_first_successful_source() -> None:
    """DataIngestor tries sources in order and returns the first success."""
    calls: list[str] = []
    good_df = pd.DataFrame(
        {"close": [100.0]},
        index=pd.date_range("2022-01-01", periods=1, freq="B"),
    )

    class FirstSource(DataSource):
        @property
        def name(self) -> str:
            return "first"

        def fetch(self, symbol, start, end, interval="1d"):
            calls.append("first")
            raise RuntimeError("First failed")

    class SecondSource(DataSource):
        @property
        def name(self) -> str:
            return "second"

        def fetch(self, symbol, start, end, interval="1d"):
            calls.append("second")
            return good_df

    ingestor = DataIngestor(sources=[FirstSource(), SecondSource()])
    result = ingestor.fetch("SPY", "2022-01-01", "2022-01-02")
    assert calls == ["first", "second"]
    assert not result.empty
