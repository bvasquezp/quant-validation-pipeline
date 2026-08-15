"""
Script de auditoria quant avanzada:
  1. Audit de holding period y trades de SMA
  2. Audit de hiperparametros e IS vs OOS por fold de LightGBM
  3. Ejecucion forzada de Stage 2 (PyTorch TCN vs TCN+Attention)
  4. Placebo test multi-semilla (10 semillas con mu=0.0)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Root path
project_root = Path(".").resolve()
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.data.ingest import compute_returns
from src.data.pit_splits import combinatorial_purged_cv
from src.features.fractional_diff import build_feature_matrix
from src.models.baselines import lightgbm_classifier, sma_crossover
from src.models.deep_model import tcn_attention_ppo, tcn_supervised
from src.orchestrator.pipeline import _generate_synthetic_data, run_pipeline


def audit_sma_holding_period():
    print("=" * 60)
    print("1. AUDIT: SMA CROSSOVER HOLDING PERIOD & TRADES")
    print("=" * 60)
    config = load_config("config/pipeline_config.yaml")
    df = _generate_synthetic_data(config, mu=0.0)
    prices = df["close"]
    returns = compute_returns(prices, method="log")

    sma_s = prices.rolling(20).mean()
    sma_l = prices.rolling(50).mean()
    signal = np.where(sma_s > sma_l, 1.0, -1.0)
    
    # Cambios de posicion (trades)
    changes = np.diff(signal) != 0
    n_trades = int(np.sum(changes))
    avg_holding = len(signal) / max(n_trades, 1)

    print(f"Total barras: {len(signal)}")
    print(f"Numero de cambios de posicion (trades): {n_trades}")
    print(f"Holding period promedio: {avg_holding:.1f} barras")
    print()


def audit_lightgbm_folds():
    print("=" * 60)
    print("2. AUDIT: LIGHTGBM HYPERPARAMS & IS vs OOS BY FOLD")
    print("=" * 60)
    config = load_config("config/pipeline_config.yaml")
    df = _generate_synthetic_data(config, mu=0.0)
    prices = df["close"]
    returns = compute_returns(prices, method="log")

    features, _ = build_feature_matrix(df, price_col="close")
    target = returns.shift(-1).reindex(features.index)
    valid_mask = target.notna()
    features = features.loc[valid_mask]
    target = target.loc[valid_mask]

    folds = combinatorial_purged_cv(
        n_samples=len(features),
        n_folds=5,
        purge_window=10,
        embargo_window=5,
    )

    lgb_result = lightgbm_classifier(
        features, target, folds,
        lgbm_params=config.models.baselines.lgbm_params.model_dump(),
        ledger_path=None,
    )

    print("Hiperparametros LightGBM:")
    print(lgb_result["hyperparams"])
    print(f"Sharpe In-Sample (IS):   {lgb_result['sharpe_is']:.4f}")
    print(f"Sharpe Out-Sample (OOS): {lgb_result['sharpe_oos']:.4f}")
    print(f"Gap IS vs OOS:           {lgb_result['sharpe_is'] - lgb_result['sharpe_oos']:.4f}")
    print()


def audit_pytorch_stage2_distinctness():
    print("=" * 60)
    print("3. AUDIT: PYTORCH DEEP MODELS (TCN vs TCN+Attention) DISTINCTNESS")
    print("=" * 60)
    config = load_config("config/pipeline_config.yaml")
    df = _generate_synthetic_data(config, mu=0.0)
    prices = df["close"]
    returns = compute_returns(prices, method="log")

    features, _ = build_feature_matrix(df, price_col="close")
    target = returns.shift(-1).reindex(features.index)
    valid_mask = target.notna()
    features = features.loc[valid_mask]
    target = target.loc[valid_mask]

    folds = combinatorial_purged_cv(n_samples=len(features), n_folds=5)

    res_tcn = tcn_supervised(features, target, folds, ledger_path=None)
    res_att = tcn_attention_ppo(features, target, folds, ledger_path=None)

    ret_tcn = res_tcn["returns_oos"]
    ret_att = res_att["returns_oos"]

    # Correlacion de Pearson entre retornos OOS
    corr = np.corrcoef(ret_tcn, ret_att)[0, 1]
    diff_abs = np.mean(np.abs(ret_tcn - ret_att))

    print(f"Sharpe TCN Supervisado:       {res_tcn['sharpe_oos']:.4f}")
    print(f"Sharpe TCN+Attention+PPO:      {res_att['sharpe_oos']:.4f}")
    print(f"Correlacion entre retornos:   {corr:.4f}")
    print(f"Diferencia absoluta media:    {diff_abs:.6f}")
    print(f"¿Son modelos distintos?:       {corr < 0.99}")
    print()


def audit_multiseed_placebo():
    print("=" * 60)
    print("4. AUDIT: PLACEBO TEST MULTI-SEMILLA (10 Semillas, mu=0.0)")
    print("=" * 60)

    rows = []
    for seed in range(1, 11):
        # Reset DB
        db = Path("config/trial_ledger.db")
        if db.exists():
            db.unlink()

        cfg = load_config("config/pipeline_config.yaml")
        cfg.reproducibility.global_seed = seed
        
        # Ejecutar pipeline
        res = run_pipeline(config=cfg, mu_override=0.0)

        for model_name, dsr_data in res.dsr_results.items():
            rows.append({
                "Seed": seed,
                "Model": model_name,
                "SR_hat": round(dsr_data.get("sr_hat", 0), 4),
                "SR*": round(dsr_data.get("sr_star", 0), 4),
                "PSR": round(dsr_data.get("psr", 0), 4),
                "DSR": round(dsr_data.get("dsr", 0), 4),
                "V_eff": round(dsr_data.get("v_effective", 0), 2),
            })

    df_res = pd.DataFrame(rows)
    print(df_res.to_string(index=False))
    print()

    print("--- Resumen Estadistico de DSR por Modelo (10 Semillas) ---")
    summary = df_res.groupby("Model")["DSR"].agg(["mean", "std", "min", "max"])
    print(summary.to_string())


if __name__ == "__main__":
    audit_sma_holding_period()
    audit_lightgbm_folds()
    audit_pytorch_stage2_distinctness()
    audit_multiseed_placebo()
