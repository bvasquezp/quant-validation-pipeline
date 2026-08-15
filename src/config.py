"""
Configuracion centralizada del pipeline.

Carga pipeline_config.yaml y lo valida con Pydantic.
Todos los parametros numericos deben venir de aqui, no hardcodeados.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Sub-modelos de configuracion
# ---------------------------------------------------------------------------

class ReproducibilityConfig(BaseModel):
    global_seed: int = 42


class DataConfig(BaseModel):
    source: str = "csv"
    path: str = "data/raw"
    symbol: str = "EURUSD"
    market: str = "forex"
    calendar: str = "fx_24_5"
    timezone: str = "UTC"
    manifest_path: str = "config/data_manifest.json"
    frequency: str = "15min"
    train_start: str = "2018-01-01"
    train_end: str = "2023-06-30"
    test_start: str = "2023-07-01"
    test_end: str = "2024-06-30"
    # Explicit source-quality exceptions. These dates are never interpolated;
    # they only document known missing-bar windows in the selected feed.
    allowed_gap_dates: List[str] = Field(default_factory=list)


class SplitsConfig(BaseModel):
    n_folds: int = 5
    purge_window: int = 10
    embargo_window: int = 5


class FractionalDiffConfig(BaseModel):
    d_range: Tuple[float, float] = (0.1, 0.9)
    d_step: float = 0.1
    threshold_adf_pvalue: float = 0.05
    weight_cutoff: float = 1e-5


class FeaturesConfig(BaseModel):
    fractional_diff: FractionalDiffConfig = FractionalDiffConfig()


class LGBMParams(BaseModel):
    n_estimators: int = 200
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_samples: int = 20


class BaselinesConfig(BaseModel):
    sma_short: int = 20
    sma_long: int = 50
    ridge_alpha: float = 1.0
    lgbm_params: LGBMParams = LGBMParams()


class TCNConfig(BaseModel):
    n_channels: List[int] = Field(default_factory=lambda: [64, 64, 64])
    kernel_size: int = 3
    dropout: float = 0.2


class AttentionConfig(BaseModel):
    n_heads: int = 4
    d_model: int = 64


class DeepConfig(BaseModel):
    tcn: TCNConfig = TCNConfig()
    attention: AttentionConfig = AttentionConfig()


class RLConfig(BaseModel):
    enabled: bool = False
    lookback: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    update_epochs: int = 4
    batch_size: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 10
    spread_pips: float = 1.0
    slippage_pips: float = 0.2
    pip_size: float = 0.0001
    turnover_penalty: float = 0.0
    drawdown_penalty: float = 0.0


class ModelsConfig(BaseModel):
    baselines: BaselinesConfig = BaselinesConfig()
    deep: DeepConfig = DeepConfig()
    rl: RLConfig = RLConfig()


class PSRDSRConfig(BaseModel):
    max_lag_autocorr: int = 5
    significance_level: float = 0.05


class BootstrapConfig(BaseModel):
    n_boot: int = 5000
    block_size: int = 50
    confidence_level: float = 0.95


class StatsConfig(BaseModel):
    psr_dsr: PSRDSRConfig = PSRDSRConfig()
    bootstrap: BootstrapConfig = BootstrapConfig()


class GateConfig(BaseModel):
    min_dsr: float = 0.95
    require_ablation_significance: bool = True
    max_pvalue_ablation: float = 0.05


class LedgerConfig(BaseModel):
    db_path: str = "config/trial_ledger.db"


class ExecutionConfig(BaseModel):
    """Broker adapter defaults; live sending must be explicitly enabled by code."""

    platform: str = "mt5"
    dry_run: bool = True
    allow_live_orders: bool = False
    magic: int = 260814
    deviation_points: int = 20
    max_volume: float = 0.01


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"


# ---------------------------------------------------------------------------
# Configuracion raiz
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    """Configuracion completa del pipeline. Ningun numero magico fuera de aqui."""

    config_version: str = "2"
    reproducibility: ReproducibilityConfig = ReproducibilityConfig()
    data: DataConfig = DataConfig()
    splits: SplitsConfig = SplitsConfig()
    features: FeaturesConfig = FeaturesConfig()
    models: ModelsConfig = ModelsConfig()
    stats: StatsConfig = StatsConfig()
    gate: GateConfig = GateConfig()
    ledger: LedgerConfig = LedgerConfig()
    execution: ExecutionConfig = ExecutionConfig()
    logging: LoggingConfig = LoggingConfig()


# ---------------------------------------------------------------------------
# Carga y seed
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path | None = None) -> PipelineConfig:
    """Carga configuracion desde YAML. Si no se pasa ruta, busca
    config/pipeline_config.yaml relativo al directorio de trabajo."""
    if config_path is None:
        config_path = Path("config/pipeline_config.yaml")
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path.resolve()}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    return PipelineConfig(**raw)


def annualization_factor_from_frequency(
    frequency: str,
    days_per_year: int = 365,
    calendar: str = "24/7",
    trading_days_per_year: int = 252,
) -> float:
    """Devuelve barras por año para una frecuencia de mercado.

    ``24/7`` usa días calendario; ``fx_24_5`` usa días operables FX.
    """
    try:
        seconds = pd.Timedelta(frequency).total_seconds()
    except ValueError as exc:
        raise ValueError(f"Invalid bar frequency: {frequency!r}") from exc

    if seconds <= 0 or days_per_year <= 0:
        raise ValueError("frequency and days_per_year must be positive")

    if calendar == "24/7":
        days = days_per_year
    elif calendar == "fx_24_5":
        if trading_days_per_year <= 0:
            raise ValueError("trading_days_per_year must be positive")
        days = trading_days_per_year
    else:
        raise ValueError(f"Unsupported market calendar: {calendar!r}")

    return days * 24.0 * 60.0 * 60.0 / seconds


def set_global_seed(seed: int) -> None:
    """Fija semilla en numpy, random y (si disponible) torch.
    Llamar UNA vez al inicio del pipeline, propagando desde el config."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
