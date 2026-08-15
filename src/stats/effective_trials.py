"""
Correccion de V (numero de trials) por correlacion entre trials.

Si los V trials estan correlacionados (ej. 50 variantes de LightGBM
que comparten casi los mismos hiperparametros), V "efectivo" es menor
que V nominal. Sin esta correccion, penalizas de mas.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional

from src.logger import get_logger

log = get_logger(__name__)


def effective_trials(
    V_nominal: int,
    avg_pairwise_corr: float = 0.0,
) -> float:
    """Calcula V efectivo ajustando por correlacion entre trials.

    V_eff = V / (1 + (V-1) * rho_promedio)

    Cuando rho = 0 (trials independientes), V_eff = V.
    Cuando rho = 1 (trials identicos), V_eff = 1.

    Args:
        V_nominal: Numero total de trials registrados.
        avg_pairwise_corr: Correlacion promedio entre pares de trials.
            Debe estar en [0, 1]. Valores negativos se clampean a 0.

    Returns:
        V efectivo (>= 1.0).
    """
    if V_nominal <= 1:
        return float(max(V_nominal, 1))

    rho = max(0.0, min(1.0, avg_pairwise_corr))

    denominator = 1.0 + (V_nominal - 1) * rho
    v_eff = V_nominal / denominator

    # V efectivo nunca menor a 1
    v_eff = max(v_eff, 1.0)

    if rho > 0.01:
        log.info(
            "effective_trials_adjusted",
            V_nominal=V_nominal,
            avg_pairwise_corr=round(rho, 4),
            V_effective=round(v_eff, 2),
        )

    return v_eff


def estimate_pairwise_correlation(
    returns_matrix: np.ndarray,
) -> float:
    """Estima la correlacion promedio entre pares de series de retornos
    de distintos trials.

    Args:
        returns_matrix: Matriz (n_trials, n_obs) de retornos OOS.
            Cada fila es la serie de retornos de un trial.

    Returns:
        Correlacion promedio entre pares (excluye la diagonal).
    """
    if returns_matrix.ndim != 2 or returns_matrix.shape[0] < 2:
        return 0.0

    n_trials = returns_matrix.shape[0]

    # Correlacion de Pearson entre todas las series
    corr_matrix = np.corrcoef(returns_matrix)

    # Mascara para la triangular superior (sin diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    # Filtrar NaN (trials con varianza cero)
    valid = corr_matrix[mask]
    valid = valid[~np.isnan(valid)]

    if len(valid) == 0:
        return 0.0

    avg_corr = float(np.mean(valid))

    log.info(
        "pairwise_correlation_estimated",
        n_trials=n_trials,
        n_pairs=len(valid),
        avg_correlation=round(avg_corr, 4),
    )

    return avg_corr
