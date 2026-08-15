"""
Script de auditoria de calibracion fina (50 semillas):
  1. Verificacion de factor de anualización A
  2. Medicion de skewness y kurtosis reales por modelo
  3. Ejecucion del placebo multi-semilla (50 semillas con mu=0.0)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(".").resolve()
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.orchestrator.pipeline import run_pipeline


def run_50_seed_audit():
    print("=" * 70)
    print("AUDITORIA DE CALIBRACION FINALE (50 SEMILLAS, mu=0.0)")
    print("=" * 70)

    rows = []
    for seed in range(1, 51):
        db = Path("config/trial_ledger.db")
        if db.exists():
            db.unlink()

        cfg = load_config("config/pipeline_config.yaml")
        cfg.reproducibility.global_seed = seed

        res = run_pipeline(config=cfg, mu_override=0.0)

        for model_name, dsr_data in res.dsr_results.items():
            rows.append({
                "Seed": seed,
                "Model": model_name,
                "SR_hat": dsr_data.get("sr_hat", 0),
                "SR_star": dsr_data.get("sr_star", 0),
                "PSR": dsr_data.get("psr", 0),
                "DSR": dsr_data.get("dsr", 0),
                "T_eff": dsr_data.get("t_effective", 0),
                "V_eff": dsr_data.get("v_effective", 0),
                "Skew": dsr_data.get("skewness", 0),
                "Kurt": dsr_data.get("kurtosis", 0),
            })

    df = pd.DataFrame(rows)

    print("\n--- RESUMEN ESTADISTICO (50 SEMILLAS) ---")
    summary = df.groupby("Model").agg(
        DSR_Mean=("DSR", "mean"),
        DSR_Std=("DSR", "std"),
        DSR_Min=("DSR", "min"),
        DSR_Max=("DSR", "max"),
        SR_hat_Mean=("SR_hat", "mean"),
        Skew_Mean=("Skew", "mean"),
        Kurt_Mean=("Kurt", "mean"),
    )
    print(summary.to_string())

    # Referencia teorica del usuario
    mc_ref = {
        "buy_hold": 0.5034,
        "sma_crossover": 0.3599,
        "ridge": 0.2722,
        "lightgbm": 0.2328,
    }

    print("\n--- COMPARACION VS REFERENCIA TEORICA MONTE CARLO ---")
    print(f"{'Model':<20} | {'DSR 50 Seed Mean':<18} | {'MC Ref':<10} | {'Diff':<8} | {'SE Mean (std/sqrt(50))':<20}")
    print("-" * 80)
    for model_name, ref in mc_ref.items():
        if model_name in summary.index:
            mean_val = summary.loc[model_name, "DSR_Mean"]
            std_val = summary.loc[model_name, "DSR_Std"]
            se_mean = std_val / np.sqrt(50)
            diff = mean_val - ref
            print(f"{model_name:<20} | {mean_val:<18.4f} | {ref:<10.4f} | {diff:<+8.4f} | {se_mean:<20.4f}")


if __name__ == "__main__":
    run_50_seed_audit()
