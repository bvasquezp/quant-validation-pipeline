"""
Agente 1 -- Ingesta de datos point-in-time.

Carga datos crudos desde CSV o fuentes externas, asegurando que
no haya look-ahead bias. Los datos se guardan como Parquet.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.logger import get_logger

log = get_logger(__name__)


def build_data_manifest(
    path: str | Path,
    df: pd.DataFrame,
    *,
    symbol: str,
    frequency: str,
    calendar: str,
    timezone: str,
) -> Dict[str, Any]:
    """Construye un manifiesto reproducible del archivo validado."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected_delta = pd.Timedelta(frequency)
    deltas = df.index.to_series().diff().dropna()
    gaps = deltas[deltas > expected_delta]
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "symbol": symbol,
        "frequency": frequency,
        "calendar": calendar,
        "timezone": timezone,
        "rows": int(len(df)),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "columns": list(df.columns),
        "gap_count": int(len(gaps)),
        "max_gap": str(gaps.max()) if not gaps.empty else None,
    }


def save_data_manifest(manifest: Dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def load_csv(
    path: str | Path,
    datetime_col: str = "datetime",
    price_col: str = "close",
    volume_col: Optional[str] = "volume",
    frequency: Optional[str] = None,
    calendar: str = "24/7",
    timezone: str = "UTC",
    symbol: Optional[str] = None,
    allowed_gap_dates: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Carga un CSV con datos OHLCV y lo normaliza.

    Args:
        path: Ruta al archivo CSV.
        datetime_col: Nombre de la columna de timestamps.
        price_col: Nombre de la columna de precios de cierre.
        volume_col: Nombre de la columna de volumen (None si no existe).
        frequency: Frecuencia esperada (ej. "15min") para validacion.
        allowed_gap_dates: Fechas explicitas con huecos conocidos del feed.
            No rellena ni interpola barras; solo evita ocultar una incidencia
            ya registrada en la metadata de la fuente.

    Returns:
        DataFrame con indice DatetimeIndex, columnas normalizadas.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Data path must be a CSV file, not a directory: {path}")

    df = pd.read_csv(path, parse_dates=[datetime_col])
    if datetime_col not in df.columns:
        raise ValueError(f"Column {datetime_col!r} not found")
    if symbol and "symbol" in df.columns:
        symbols = df["symbol"].dropna().astype(str).unique()
        if len(symbols) != 1 or symbols[0].upper() != symbol.upper():
            raise ValueError(f"Expected one symbol {symbol!r}, found {symbols.tolist()}")

    timestamps = pd.DatetimeIndex(df[datetime_col])
    timestamps = (
        timestamps.tz_localize(timezone)
        if timestamps.tz is None
        else timestamps.tz_convert(timezone)
    )
    df[datetime_col] = timestamps
    df = df.sort_values(datetime_col).reset_index(drop=True).set_index(datetime_col)

    # Renombrar columnas si es necesario.
    expected_cols = ["open", "high", "low", "close", "volume"]
    for col in df.columns:
        if col.lower() in expected_cols and col != col.lower():
            df = df.rename(columns={col: col.lower()})

    required = {"open", "high", "low", "close"}
    if volume_col:
        required.add("volume")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.index.has_duplicates:
        raise ValueError("Duplicate timestamps are not allowed")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Timestamps must be monotonic increasing")
    if df[list(required)].isna().any().any():
        raise ValueError("Required OHLCV columns cannot contain NaN")
    if (df["open"] <= 0).any() or (df["close"] <= 0).any():
        raise ValueError("OHLC prices must be positive")
    if (df["high"] < df[["open", "close"]].max(axis=1)).any():
        raise ValueError("high must be >= open and close")
    if (df["low"] > df[["open", "close"]].min(axis=1)).any():
        raise ValueError("low must be <= open and close")

    if frequency and len(df) > 1:
        expected_delta = pd.Timedelta(frequency)
        deltas = df.index.to_series().diff().dropna()
        median_delta = deltas.median()
        if median_delta != expected_delta:
            raise ValueError(
                f"Frequency mismatch: configured={frequency!r}, "
                f"observed_median={median_delta}"
            )
        if (deltas < expected_delta).any():
            raise ValueError("Timestamps contain sub-frequency overlaps")
        if calendar not in {"24/7", "fx_24_5"}:
            raise ValueError(f"Unsupported market calendar: {calendar!r}")
        if calendar == "24/7" and (deltas > expected_delta).any():
            raise ValueError("Unexpected gaps for a 24/7 dataset")
        if calendar == "fx_24_5":
            allowed_dates = {
                str(pd.Timestamp(value).date()) for value in (allowed_gap_dates or [])
            }
            previous = df.index[:-1]
            current = df.index[1:]
            for prev_ts, curr_ts, delta in zip(previous, current, deltas):
                is_weekly_fx_gap = (
                    prev_ts.weekday() == 4
                    and (
                        curr_ts.weekday() == 0
                        or (curr_ts.weekday() == 6 and curr_ts.hour >= 21)
                    )
                )
                is_allowlisted_source_gap = (
                    str(prev_ts.date()) in allowed_dates
                    or str(curr_ts.date()) in allowed_dates
                )
                if delta > expected_delta and not (
                    is_weekly_fx_gap or is_allowlisted_source_gap
                ):
                    raise ValueError(f"Unexpected internal FX gap: {prev_ts} -> {curr_ts}")
                if delta > expected_delta and is_allowlisted_source_gap and not is_weekly_fx_gap:
                    log.warning(
                        "allowlisted_source_gap_no_interpolation",
                        previous=str(prev_ts),
                        current=str(curr_ts),
                        delta=str(delta),
                    )

    log.info(
        "data_loaded",
        path=str(path),
        n_rows=len(df),
        start=str(df.index.min()),
        end=str(df.index.max()),
    )

    return df


def compute_returns(
    prices: pd.Series,
    method: str = "log",
) -> pd.Series:
    """Calcula retornos a partir de precios.

    Args:
        prices: Serie de precios.
        method: "log" (log-retornos) o "simple" (retornos aritmeticos).

    Returns:
        Serie de retornos (sin el primer NaN).
    """
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    elif method == "simple":
        returns = prices.pct_change()
    else:
        raise ValueError(f"Unknown method: {method}")

    returns = returns.dropna()

    log.info(
        "returns_computed",
        method=method,
        n_obs=len(returns),
        mean=round(float(returns.mean()), 8),
        std=round(float(returns.std()), 8),
    )

    return returns


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Guarda un DataFrame como Parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow")
    log.info("parquet_saved", path=str(path), n_rows=len(df))
    return path


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Carga un DataFrame desde Parquet."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    df = pd.read_parquet(path, engine="pyarrow")
    log.info("parquet_loaded", path=str(path), n_rows=len(df))
    return df
