"""BTC/USD spot resolution pilot, development only.

Use a Kraken XBTUSD_15.csv (raw, no header). Timestamps are UTC bar starts.
Example: python -m scratch.btc_resolution_pilot XBTUSD_15.csv --holdout-start YYYY-MM-DD
Choose the holdout date before examining results and reserve six months.
This program reports exploratory results only and never opens the holdout for scoring.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DAY = pd.Timedelta(days=1)
FEE = 0.008
COLUMNS = ["open", "high", "low", "close", "volume", "trades"]


def load_development(path, cutoff):
    raw = pd.read_csv(path, header=None)
    if raw.shape[1] != 7:
        raise ValueError("Expected seven fields in Kraken OHLCVT format")
    timestamp = pd.to_datetime(raw.iloc[:, 0], unit="s", utc=True)
    data = raw.loc[timestamp < cutoff, raw.columns[1:]].copy()
    data.columns = COLUMNS
    data.index = pd.DatetimeIndex(timestamp[timestamp < cutoff])
    data = data.sort_index().astype(float)
    if data.empty or not np.isfinite(data.to_numpy()).all():
        raise ValueError("Empty or nonfinite development data")
    prices = data[["open", "high", "low", "close"]]
    if (prices <= 0).any().any() or (data[["volume", "trades"]] < 0).any().any():
        raise ValueError("Invalid prices or quantities")
    if (data.high < prices.max(axis=1)).any() or (data.low > prices.min(axis=1)).any():
        raise ValueError("Invalid high/low bounds")
    grid = pd.date_range(data.index[0], cutoff, freq="15min", inclusive="left")
    if data.index[0] != data.index[0].floor("15min") or not data.index.equals(grid):
        raise ValueError("Development bars must be unique and continuous on the UTC grid")
    return data


def sample_features(data, frequency, decision_times):
    close = np.log(data.close.resample(frequency, label="right", closed="left").last())
    bars_per_day = int(DAY / pd.Timedelta(frequency))
    features = {
        f"momentum_{days}d": close.diff(days * bars_per_day)
        for days in (1, 3, 7)
    }
    features["realized_volatility_3d"] = np.sqrt(
        close.diff().pow(2).rolling(3 * bars_per_day).sum()
    )
    return pd.DataFrame(features).reindex(decision_times)


def predict_development(data, cutoff):
    days = pd.date_range(data.index[0].ceil("D"), cutoff, freq="D", inclusive="left")
    # The 00:15 bar open proxies a fill strictly after the 00:00 decision.
    # Both variants share the same proxy and label horizon.
    fills = data.open.reindex(days + pd.Timedelta(minutes=15)).set_axis(days)
    targets = np.log(fills.shift(-1) / fills)
    feature_sets = {key: sample_features(data, key, days) for key in ("4h", "30min")}
    features = pd.concat(feature_sets, axis=1)
    usable = features.notna().all(axis=1) & targets.notna()
    features, targets = features.loc[usable], targets.loc[usable]
    if len(features) < 730 + 2 * 90 + 1:
        raise ValueError("Need 730 training days plus two complete 90-day development folds")
    positions = []
    tested_days = []
    for start in range(731, len(features) - 89, 90):
        test = features.iloc[start:start + 90]
        # Previous label ends after the first test decision, so purge that row.
        train = features.iloc[:start - 1]
        fold = pd.DataFrame(index=test.index)
        fold["cash"] = False
        fold["buy_hold"] = True
        fold["trend_7d"] = test["4h"]["momentum_7d"] > 0
        for frequency in feature_sets:
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(train[frequency], targets.loc[train.index])
            prediction = model.predict(test[frequency])
            fold[f"ridge_{frequency}"] = prediction > np.log((1 + FEE) / (1 - FEE))
        positions.append(fold)
        tested_days.extend(test.index)
    return pd.concat(positions), fills.loc[tested_days], targets.loc[tested_days]


def simulate(positions, fills, targets, friction_bps):
    # Fractional research proxy. Exchange minimums, ticks and funding are
    # deliberately outside this comparison and required for a paper gate.
    friction = friction_bps * 1e-4
    if not 0 <= friction < 1:
        raise ValueError("Invalid friction")
    cash, units = 30_000.0, 0.0
    curve = [cash]
    operations = 0
    for want_long, price, target in zip(positions, fills, targets, strict=True):
        if want_long and not units:
            units = cash / (price * (1 + friction) * (1 + FEE))
            cash = 0.0
            operations += 1
        elif not want_long and units:
            cash = units * price * (1 - friction) * (1 - FEE)
            units = 0.0
            operations += 1
        end_price = price * np.exp(target)
        curve.append(cash + units * end_price)
    if units:
        curve[-1] *= (1 - friction) * (1 - FEE)
        operations += 1
    equity = np.asarray(curve)
    return {
        "net_return": float(equity[-1] / equity[0] - 1),
        "max_drawdown": float((equity / np.maximum.accumulate(equity) - 1).min()),
        "operations": operations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--holdout-start", required=True)
    args = parser.parse_args()
    cutoff = pd.Timestamp(args.holdout_start, tz="UTC")
    if cutoff != cutoff.normalize():
        parser.error("Holdout start must be a UTC calendar date")
    data = load_development(args.csv, cutoff)
    positions, fills, targets = predict_development(data, cutoff)
    source_hash = hashlib.sha256(data.to_csv().encode("utf-8")).hexdigest()
    report = {
        "status": "DEVELOPMENT_ONLY",
        "development_canonical_sha256": source_hash,
        "holdout_start_utc": cutoff.isoformat(),
        "first_decision": positions.index[0].isoformat(),
        "last_decision": positions.index[-1].isoformat(),
        "observations": len(positions),
        "fee_each_side": FEE,
        "friction_bps_each_side": {
            str(bps): {
                name: simulate(positions[name], fills, targets, bps)
                for name in positions
            }
            for bps in (5, 10, 20)
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
