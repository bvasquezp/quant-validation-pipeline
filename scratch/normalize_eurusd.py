"""Normalize the downloaded EUR/USD M15 source into the pipeline contract."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SOURCE = Path("data/raw/EURUSD_M15.csv")
OUTPUT = Path("data/raw/EURUSD_15min.csv")
NOTES = Path("config/eurusd_source.json")
START = pd.Timestamp("2018-01-01", tz="UTC")
END = pd.Timestamp("2024-02-09 23:59:59", tz="UTC")


def main() -> None:
    raw = pd.read_csv(SOURCE)
    datetime_index = pd.to_datetime(
        raw["Date"].astype(str) + " " + raw["Time"].astype(str),
        format="%Y.%m.%d %H:%M:%S",
        utc=True,
    )
    clean = pd.DataFrame(
        {
            "datetime": datetime_index,
            "open": raw["Open"],
            "high": raw["High"],
            "low": raw["Low"],
            "close": raw["Close"],
            "volume": raw["Tick volume"],
        }
    )
    clean = clean.loc[(clean.datetime >= START) & (clean.datetime <= END)]
    clean = clean.sort_values("datetime").drop_duplicates("datetime")
    if clean.empty:
        raise RuntimeError("No EUR/USD rows remain after the configured date filter")
    if clean[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise RuntimeError("Downloaded source contains NaN values")
    if (clean[["open", "high", "low", "close"]] <= 0).any().any():
        raise RuntimeError("Downloaded source contains non-positive prices")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(OUTPUT, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    NOTES.write_text(
        json.dumps(
            {
                "symbol": "EURUSD",
                "frequency": "15min",
                "calendar": "fx_24_5",
                "timezone": "UTC",
                "source_dataset": "Hugo Lemonnier / 15m Tick data EURUSD",
                "source_url": "https://www.kaggle.com/datasets/hugolemonnier/eur-15min",
                "upstream_claim": "Tick Data Suite export, BID-side OHLC plus tick volume",
                "license_note": "Kaggle page does not declare a clear license; keep provenance and verify permission before public/commercial use.",
                "start": str(clean.datetime.min()),
                "end": str(clean.datetime.max()),
                "rows": int(len(clean)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT} rows={len(clean)} start={clean.datetime.min()} end={clean.datetime.max()}")


if __name__ == "__main__":
    main()
