from __future__ import annotations

import pandas as pd

from src.config import annualization_factor_from_frequency, load_config
from src.orchestrator.pipeline import _generate_synthetic_data


def test_frequency_derives_annualization_factor():
    assert annualization_factor_from_frequency("15min", calendar="fx_24_5") == 24192.0
    assert annualization_factor_from_frequency("1h", calendar="fx_24_5") == 6048.0
    assert annualization_factor_from_frequency("15min") == 35040.0


def test_synthetic_data_uses_configured_frequency():
    config = load_config("config/pipeline_config.yaml")
    assert config.data.symbol == "EURUSD"
    assert config.data.calendar == "fx_24_5"
    assert config.execution.platform == "mt5"
    assert config.execution.dry_run is True
    assert config.execution.allow_live_orders is False
    assert config.execution.max_volume == 0.01
    data = _generate_synthetic_data(config, n_bars=5000, mu=0.0)
    assert not (
        (data.index.dayofweek == 5)
        | ((data.index.dayofweek == 6) & (data.index.hour < 23))
    ).any()
    for previous, current in zip(data.index[:-1], data.index[1:]):
        delta = current - previous
        assert delta == pd.Timedelta("15min") or (
            previous.weekday() == 4
            and (
                current.weekday() == 0
                or (current.weekday() == 6 and current.hour >= 23)
            )
        )
    assert str(data.index.tz) == "UTC"
