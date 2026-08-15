# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Ablation Report
#
# Este notebook ejecuta la escalera de ablacion y muestra los resultados
# de cada comparacion entre modelos consecutivos.
#
# Secuencia: Buy&Hold -> SMA -> Ridge -> LightGBM -> TCN -> TCN+Att+PPO

# %%
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Agregar el root del proyecto al path
project_root = Path("..").resolve()
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.orchestrator.pipeline import run_pipeline

# %% [markdown]
# ## Ejecutar pipeline

# %%
# Parametros (papermill puede inyectarlos aqui)
config_path = str(project_root / "config" / "pipeline_config.yaml")

# %%
config = load_config(config_path)
result = run_pipeline(config=config)

# %% [markdown]
# ## Resultados DSR por modelo

# %%
dsr_rows = []
for stage_name, dsr_data in result.dsr_results.items():
    dsr_rows.append({
        "Model": stage_name,
        "SR_hat": round(dsr_data.get("sr_hat", 0), 4),
        "SR*": round(dsr_data.get("sr_star", 0), 6),
        "PSR": round(dsr_data.get("psr", 0), 4),
        "DSR": round(dsr_data.get("dsr", 0), 4),
        "T_eff": round(dsr_data.get("t_effective", 0), 0),
        "V_nom": dsr_data.get("v_nominal", 0),
        "V_eff": round(dsr_data.get("v_effective", 0), 1),
        "Skew": round(dsr_data.get("skewness", 0), 4),
        "Kurt": round(dsr_data.get("kurtosis", 0), 4),
    })

df_dsr = pd.DataFrame(dsr_rows)
print(df_dsr.to_string(index=False))

# %% [markdown]
# ## Resultados de Ablacion (bootstrap)

# %%
abl_rows = []
for comp_name, comp_data in result.ablation_results.items():
    if comp_name == "recommended_stage":
        continue
    if not isinstance(comp_data, dict):
        continue
    abl_rows.append({
        "Comparison": comp_name,
        "Decision": comp_data.get("decision", "N/A"),
        "SR_A": round(comp_data.get("sharpe_a", 0), 4) if "sharpe_a" in comp_data else "N/A",
        "SR_B": round(comp_data.get("sharpe_b", 0), 4) if "sharpe_b" in comp_data else "N/A",
        "Diff": round(comp_data.get("observed_diff", 0), 4) if "observed_diff" in comp_data else "N/A",
        "p-value": round(comp_data.get("p_value", 1), 4) if "p_value" in comp_data else "N/A",
        "CI_95": f"[{comp_data.get('ci_lower', 0):.4f}, {comp_data.get('ci_upper', 0):.4f}]"
            if "ci_lower" in comp_data else "N/A",
    })

df_abl = pd.DataFrame(abl_rows)
print(df_abl.to_string(index=False))

# %% [markdown]
# ## Modelo recomendado

# %%
print(f"Modelo recomendado por ablacion: {result.recommended_stage}")
print(f"Razon: la complejidad adicional mas alla de este modelo")
print(f"no se justifica estadisticamente (p >= 0.05).")
