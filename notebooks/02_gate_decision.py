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
# # Gate Decision Report
#
# Este notebook compila todos los resultados del pipeline y documenta
# la decision GO/NO-GO con la justificacion completa.
#
# **El reporte final documenta:**
# - T usado (nominal y efectivo)
# - V nominal y V efectivo
# - Correlacion promedio entre trials
# - Decision y razon

# %%
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path("..").resolve()
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.orchestrator.pipeline import run_pipeline

# %% [markdown]
# ## Ejecutar pipeline completo

# %%
config_path = str(project_root / "config" / "pipeline_config.yaml")

# %%
config = load_config(config_path)
result = run_pipeline(config=config)

# %% [markdown]
# ## Decision GO/NO-GO

# %%
print("=" * 60)
print(f"  DECISION:  {result.gate_decision}")
print("=" * 60)
print()
print(f"Modelo recomendado: {result.recommended_stage}")
print(f"Tiempo de ejecucion: {result.execution_time_seconds:.1f}s")
print()
print("Razones:")
for i, reason in enumerate(result.gate_reasons, 1):
    print(f"  {i}. {reason}")

# %% [markdown]
# ## Detalle completo (JSON)

# %%
print(result.to_json())

# %% [markdown]
# ## Resumen estadistico

# %%
if result.recommended_stage in result.dsr_results:
    dsr = result.dsr_results[result.recommended_stage]
    print(f"Modelo: {result.recommended_stage}")
    print(f"  SR observado:   {dsr.get('sr_hat', 'N/A')}")
    print(f"  SR* (nula):     {dsr.get('sr_star', 'N/A')}")
    print(f"  DSR:            {dsr.get('dsr', 'N/A')}")
    print(f"  T nominal:      {dsr.get('t_nominal', 'N/A')}")
    print(f"  T efectivo:     {dsr.get('t_effective', 'N/A')}")
    print(f"  V nominal:      {dsr.get('v_nominal', 'N/A')}")
    print(f"  V efectivo:     {dsr.get('v_effective', 'N/A')}")
    print(f"  Skewness:       {dsr.get('skewness', 'N/A')}")
    print(f"  Kurtosis:       {dsr.get('kurtosis', 'N/A')}")
