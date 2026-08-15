"""
Agente 5 -- Nucleo estadistico: PSR y DSR.

T y V son cosas distintas y entran en formulas distintas:
  - T = observaciones de retornos (determina la varianza del Sharpe estimado)
  - V = configuraciones probadas (determina la barra de comparacion SR*)

Implementa las correcciones por:
  - Autocorrelacion serial (Newey-West para T efectivo)
  - Skewness y kurtosis de retornos (formula de Lo)
  - Correlacion entre trials (V efectivo)
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy import stats as sp_stats

from src.logger import get_logger
from src.stats.effective_trials import effective_trials

log = get_logger(__name__)

# Constante de Euler-Mascheroni
EULER_MASCHERONI = 0.5772156649015329


# ---------------------------------------------------------------------------
# T efectivo -- correccion por autocorrelacion serial (Newey-West)
# ---------------------------------------------------------------------------

def effective_sample_size(returns: np.ndarray, max_lag: Optional[int] = None) -> float:
    """Ajusta T hacia abajo si hay autocorrelacion serial.

    Usa el estimador Newey-West para el tamaño de muestra efectivo:
        T_eff = T / (1 + 2 * sum_{k=1}^{max_lag} (1 - k/(max_lag+1)) * rho_k)

    Si max_lag no se especifica, calcula dinámicamente el lag basado en la
    decadencia de la autocorrelacion serial (hasta min(T/4, 200)).
    """
    T = len(returns)
    if T < 5:
        return float(T)

    mean_r = np.mean(returns)
    demeaned = returns - mean_r
    var_r = np.sum(demeaned ** 2) / T

    if var_r < 1e-15:
        return float(T)

    if max_lag is None:
        limit = min(T // 4, 100)
        autocorrs = []
        for k in range(1, limit + 1):
            autocov_k = np.sum(demeaned[k:] * demeaned[:-k]) / T
            autocorrs.append(autocov_k / var_r)
        
        threshold = 1.96 / np.sqrt(T)
        dynamic_lag = 5
        for k, ac in enumerate(autocorrs, 1):
            if abs(ac) > threshold:
                dynamic_lag = k
        max_lag = min(max(dynamic_lag * 2, 5), limit)

    rho_sum = 0.0
    for k in range(1, max_lag + 1):
        if k >= T:
            break
        autocov_k = np.sum(demeaned[k:] * demeaned[:-k]) / T
        rho_k = autocov_k / var_r
        weight = 1.0 - k / (max_lag + 1)
        rho_sum += weight * rho_k

    denominator = 1.0 + 2.0 * rho_sum
    if denominator <= 0:
        return float(T)

    t_eff = T / denominator
    return max(t_eff, 2.0)


# ---------------------------------------------------------------------------
# Varianza del Sharpe Ratio estimado (formula de Lo, 2002)
# ---------------------------------------------------------------------------

def sharpe_variance(
    sr_hat: float,
    T: float,
    skewness: float,
    kurtosis: float,
) -> float:
    """Var(SR_hat) segun Lo (2002), ajustada por skew y kurtosis.

    Var(SR) = (1/(T-1)) * (1 - gamma3*SR + ((gamma4 - 1)/4)*SR^2)

    donde gamma3 = skewness, gamma4 = excess kurtosis + 3 (kurtosis total).
    """
    if T <= 1:
        return float("inf")

    gamma4 = kurtosis + 3.0

    var = (1.0 / (T - 1)) * (
        1.0
        - skewness * sr_hat
        + ((gamma4 - 1.0) / 4.0) * sr_hat ** 2
    )
    return max(var, 1e-15)


# ---------------------------------------------------------------------------
# SR* -- Sharpe esperado bajo la nula (maximo de V trials nulos)
# ---------------------------------------------------------------------------

def expected_max_sharpe(
    V: float,
    sigma_sr: float,
) -> float:
    """SR* esperado bajo H0: todos los V trials tienen SR real = 0.

    SR* = sigma_SR * [(1 - gamma) * Phi^{-1}(1 - 1/V)
                       + gamma * Phi^{-1}(1 - 1/(V*e))]

    donde gamma = Euler-Mascheroni, Phi^{-1} = inversa normal estandar.

    Si V = 1, SR* = 0 (un solo trial no tiene sesgo de seleccion).
    """
    if V <= 1.0:
        return 0.0

    gamma = EULER_MASCHERONI

    p1 = max(1.0 - 1.0 / V, 1e-10)
    p2 = max(1.0 - 1.0 / (V * np.e), 1e-10)
    p1 = min(p1, 1.0 - 1e-10)
    p2 = min(p2, 1.0 - 1e-10)

    sr_star = sigma_sr * (
        (1.0 - gamma) * sp_stats.norm.ppf(p1)
        + gamma * sp_stats.norm.ppf(p2)
    )
    return sr_star


# ---------------------------------------------------------------------------
# PSR -- Probabilistic Sharpe Ratio
# ---------------------------------------------------------------------------

def probabilistic_sharpe_ratio(
    sr_hat: float,
    sr_benchmark: float,
    T: float,
    skewness: float,
    kurtosis: float,
) -> float:
    """PSR = Prob(SR real > sr_benchmark | datos).

    PSR = Phi( (SR_hat - SR_benchmark) / sqrt(Var(SR)) )
    """
    var_sr = sharpe_variance(sr_hat, T, skewness, kurtosis)
    std_sr = np.sqrt(var_sr)

    if std_sr < 1e-15:
        return 1.0 if sr_hat > sr_benchmark else 0.0

    z = (sr_hat - sr_benchmark) / std_sr
    return float(sp_stats.norm.cdf(z))


# ---------------------------------------------------------------------------
# DSR -- Deflated Sharpe Ratio (la pieza central)
# ---------------------------------------------------------------------------

def deflated_sharpe_ratio(
    sr_hat: float,
    returns: np.ndarray,
    n_trials: int,
    avg_pairwise_corr: float = 0.0,
    max_lag_autocorr: Optional[int] = None,
    annualized_input: bool = True,
    annualization_factor: Optional[float] = None,
) -> Dict[str, float]:
    """Calcula el Deflated Sharpe Ratio.

    IMPORTANTE: Convierte sr_hat (anualizado) a Sharpe por periodo
    para mantener la consistencia de escala con sigma_SR = 1/sqrt(T-1).
    """
    returns = np.asarray(returns, dtype=np.float64)
    T_nominal = len(returns)

    if annualized_input:
        if annualization_factor is None or annualization_factor <= 0:
            raise ValueError(
                "annualization_factor is required and must be positive "
                "when annualized_input=True"
            )
    elif annualization_factor is None:
        annualization_factor = 1.0

    if T_nominal < 3:
        log.warning("dsr_insufficient_data", T=T_nominal)
        return {
            "psr": float("nan"),
            "dsr": float("nan"),
            "sr_star": float("nan"),
            "sr_hat": sr_hat,
            "t_nominal": T_nominal,
            "t_effective": float(T_nominal),
            "v_nominal": n_trials,
            "v_effective": float(n_trials),
            "skewness": float("nan"),
            "kurtosis": float("nan"),
            "sigma_sr": float("nan"),
        }

    # Convertir Sharpe observado a la escala del periodo de los datos
    if annualized_input and annualization_factor > 0:
        sr_hat_per_period = sr_hat / np.sqrt(annualization_factor)
    else:
        sr_hat_per_period = sr_hat

    # 1. T efectivo
    t_eff = effective_sample_size(returns, max_lag=max_lag_autocorr)

    # 2. Momentos de la distribucion de retornos
    skew = float(sp_stats.skew(returns, bias=False))
    kurt = float(sp_stats.kurtosis(returns, bias=False))

    # 3. V efectivo
    v_eff = effective_trials(n_trials, avg_pairwise_corr)

    # 4. Sigma_SR (desviacion estandar del SR por periodo bajo la nula)
    sigma_sr = np.sqrt(sharpe_variance(0.0, t_eff, skew, kurt))

    # 5. SR* por periodo esperado por azar
    sr_star_period = expected_max_sharpe(v_eff, sigma_sr)

    # 6. PSR sin ajuste de seleccion (benchmark = 0)
    psr = probabilistic_sharpe_ratio(sr_hat_per_period, 0.0, t_eff, skew, kurt)

    # 7. DSR -- PSR con benchmark = SR* (ajustado por seleccion)
    dsr = probabilistic_sharpe_ratio(sr_hat_per_period, sr_star_period, t_eff, skew, kurt)

    # Re-anualizar SR* para presentacion
    sr_star_annual = sr_star_period * np.sqrt(annualization_factor) if annualized_input else sr_star_period

    result = {
        "psr": psr,
        "dsr": dsr,
        "sr_star": sr_star_annual,
        "sr_hat": sr_hat,
        "t_nominal": T_nominal,
        "t_effective": t_eff,
        "v_nominal": n_trials,
        "v_effective": v_eff,
        "skewness": skew,
        "kurtosis": kurt,
        "sigma_sr": sigma_sr,
    }

    log.info("dsr_computed", **{k: round(v, 6) if isinstance(v, float) else v
                                for k, v in result.items()})
    return result
