"""Validate and manifest the downloaded EURUSD source."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data.ingest import build_data_manifest, load_csv, save_data_manifest


def main() -> None:
    config = load_config("config/pipeline_config.yaml")
    path = Path(config.data.path)
    df = load_csv(
        path,
        frequency=config.data.frequency,
        calendar=config.data.calendar,
        timezone=config.data.timezone,
        symbol=config.data.symbol,
        allowed_gap_dates=config.data.allowed_gap_dates,
    )

    expected = pd.Timedelta(config.data.frequency)
    deltas = df.index.to_series().diff().dropna()
    gaps = deltas[deltas > expected]
    weekly = []
    source = []
    for current, delta in gaps.items():
        previous = df.index[df.index.get_loc(current) - 1]
        is_weekly = previous.weekday() == 4 and (
            current.weekday() == 0
            or (current.weekday() == 6 and current.hour >= 21)
        )
        item = {
            "previous": str(previous),
            "current": str(current),
            "delta": str(delta),
        }
        (weekly if is_weekly else source).append(item)

    manifest = build_data_manifest(
        path,
        df,
        symbol=config.data.symbol,
        frequency=config.data.frequency,
        calendar=config.data.calendar,
        timezone=config.data.timezone,
    )
    manifest["weekly_market_gaps"] = weekly
    manifest["allowlisted_source_gaps"] = source
    save_data_manifest(manifest, config.data.manifest_path)

    print(json.dumps({
        "path": str(path),
        "rows": len(df),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "weekly_market_gaps": len(weekly),
        "allowlisted_source_gaps": len(source),
        "manifest": config.data.manifest_path,
    }, indent=2))


if __name__ == "__main__":
    main()
