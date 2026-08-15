"""Compare downloaded EURUSD feeds without concatenating them."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ROOT / "data/raw/EURUSD_15min.csv",
    ROOT / "data/raw/EURUSD_M15_2023_2025.csv",
    ROOT / "data/raw/ohlc_15m.csv",
    ROOT / "data/raw/EURUSD_M15_200001030000_201912312245.csv",
]


def _find_column(df: pd.DataFrame, *names: str) -> str | None:
    by_lower = {
        str(col).strip().strip("<>").lower(): col for col in df.columns
    }
    for name in names:
        if name.lower() in by_lower:
            return by_lower[name.lower()]
    return None


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    dt_col = _find_column(df, "datetime", "timestamp", "date_time")
    if dt_col:
        timestamps = pd.to_datetime(df[dt_col], utc=True)
    else:
        date_col = _find_column(df, "date")
        time_col = _find_column(df, "time")
        if not date_col and time_col:
            timestamps = pd.to_datetime(df[time_col], utc=True)
        elif not date_col or not time_col:
            raise ValueError(
                f"Cannot find timestamp columns in {path.name}; columns={list(df.columns)}"
            )
        else:
            timestamps = pd.to_datetime(
                df[date_col].astype(str) + " " + df[time_col].astype(str),
                utc=True,
            )
    close_col = _find_column(df, "close")
    if close_col is None:
        raise ValueError(f"Cannot find close column in {path.name}")
    volume_col = _find_column(df, "volume", "tick_volume", "tickvol")
    result = pd.DataFrame(
        {"close": pd.to_numeric(df[close_col]).to_numpy()},
        index=timestamps,
    )
    if volume_col:
        result["volume"] = pd.to_numeric(
            df[volume_col], errors="coerce"
        ).to_numpy()
    return result[~result.index.duplicated(keep="first")].sort_index()


def main() -> None:
    loaded = {}
    for path in FILES:
        if not path.exists():
            continue
        frame = _load(path)
        loaded[path.name] = frame
        print(
            f"{path.name}: rows={len(frame)} start={frame.index.min()} "
            f"end={frame.index.max()} volume={'volume' in frame.columns} "
            f"sample={list(frame.index[:2])} close_non_na={frame['close'].notna().sum()} "
            f"close_head={list(frame['close'].head(2))}"
        )

    canonical = loaded.get("EURUSD_15min.csv")
    if canonical is None:
        raise FileNotFoundError("Canonical EURUSD_15min.csv is missing")
    print(
        "canonical_2023_sample=",
        list(canonical.loc["2023"].index[:3]),
        "canonical_2023_rows=",
        len(canonical.loc["2023"]),
    )

    for name, frame in loaded.items():
        if name == "EURUSD_15min.csv":
            continue
        best = None
        debug_shift = frame[["close"]].copy()
        debug_shift.index = debug_shift.index + pd.Timedelta(minutes=-480)
        debug_overlap = canonical[["close"]].join(
            debug_shift.rename(columns={"close": "close_other"}),
            how="inner",
        )
        print(f"{name}: debug_shift_minus_480_overlap={len(debug_overlap)}")
        for minutes in range(-12 * 60, 12 * 60 + 1, 15):
            shifted = frame[["close"]].copy()
            shifted.index = shifted.index + pd.Timedelta(minutes=minutes)
            overlap = canonical[["close"]].join(
                shifted.rename(columns={"close": "close_other"}),
                how="inner",
            ).dropna()
            if len(overlap) < 1000:
                continue
            returns = np.log(overlap).diff().dropna()
            corr = returns["close"].corr(returns["close_other"])
            candidate = (float(corr), minutes, len(overlap), overlap)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
        if best is None:
            print(f"{name}: no_shifted_overlap_over_1000_bars")
            continue
        corr, minutes, overlap_count, overlap = best
        median_abs_log_diff = np.median(
            np.abs(np.log(overlap["close"] / overlap["close_other"]))
        )
        print(
            f"{name}: best_shift_minutes={minutes} overlap={overlap_count} "
            f"return_corr={corr:.6f} median_abs_log_price_diff={median_abs_log_diff:.2e}"
        )


if __name__ == "__main__":
    main()
