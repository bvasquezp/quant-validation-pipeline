import pandas as pd
import pytest

from src.data.ingest import load_csv


def _write_ohlcv(path, dates):
    df = pd.DataFrame(
        {
            "datetime": dates,
            "open": [1.0] * len(dates),
            "high": [1.01] * len(dates),
            "low": [0.99] * len(dates),
            "close": [1.0] * len(dates),
            "volume": [100.0] * len(dates),
            "symbol": ["EURUSD"] * len(dates),
        }
    )
    df.to_csv(path, index=False)


def test_fx_loader_allows_weekend_gap(tmp_path):
    dates = list(pd.date_range("2024-01-05 23:00", periods=4, freq="15min"))
    dates.append(pd.Timestamp("2024-01-08 00:00"))
    path = tmp_path / "eurusd.csv"
    _write_ohlcv(path, dates)

    loaded = load_csv(path, frequency="15min", calendar="fx_24_5", symbol="EURUSD")

    assert len(loaded) == 5
    assert str(loaded.index.tz) == "UTC"


def test_fx_loader_allows_sunday_open_gap(tmp_path):
    path = tmp_path / "eurusd_sunday_open.csv"
    _write_ohlcv(
        path,
        [
            pd.Timestamp("2024-01-05 22:30"),
            pd.Timestamp("2024-01-05 22:45"),
            pd.Timestamp("2024-01-07 23:00"),
            pd.Timestamp("2024-01-07 23:15"),
        ],
    )

    loaded = load_csv(path, frequency="15min", calendar="fx_24_5", symbol="EURUSD")

    assert len(loaded) == 4


def test_fx_loader_rejects_internal_gap(tmp_path):
    dates = list(pd.date_range("2024-01-08 00:00", periods=2, freq="15min"))
    dates += [pd.Timestamp("2024-01-08 00:45"), pd.Timestamp("2024-01-08 01:00")]
    path = tmp_path / "eurusd.csv"
    _write_ohlcv(path, dates)

    with pytest.raises(ValueError, match="internal FX gap"):
        load_csv(path, frequency="15min", calendar="fx_24_5", symbol="EURUSD")


def test_fx_loader_allows_explicit_source_gap_without_filling(tmp_path):
    path = tmp_path / "eurusd_source_gap.csv"
    _write_ohlcv(
        path,
        [
            pd.Timestamp("2024-01-08 00:00"),
            pd.Timestamp("2024-01-08 00:15"),
            pd.Timestamp("2024-01-08 01:00"),
            pd.Timestamp("2024-01-08 01:15"),
        ],
    )

    loaded = load_csv(
        path,
        frequency="15min",
        calendar="fx_24_5",
        symbol="EURUSD",
        allowed_gap_dates=["2024-01-08"],
    )

    assert len(loaded) == 4
    assert loaded.index.to_series().diff().max() == pd.Timedelta("45min")
