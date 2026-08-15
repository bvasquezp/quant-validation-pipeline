"""
Agente 2 -- Diferenciacion fraccional (Fixed-Width Window FFD).

Transforma series de precios no estacionarias en series estacionarias
preservando la mayor cantidad de memoria posible (a diferencia de
la diferenciacion entera d=1 que destruye la memoria).

La d optima es la minima d que logra estacionariedad (ADF p < umbral).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from src.logger import get_logger

log = get_logger(__name__)


@dataclass
class FoldFeatureSet:
    """Features/targets for one fold with point-in-time ``d`` selection."""

    fold_id: int
    train_features: pd.DataFrame
    train_target: pd.Series
    test_features: pd.DataFrame
    test_target: pd.Series
    d_used: float
    ffd_lookback: int
    d_selection_method: str = "rule_based_adf"


def _ffd_weights(d: float, threshold: float = 1e-5) -> np.ndarray:
    """Calcula los pesos para la diferenciacion fraccional (FFD).

    Los pesos son los coeficientes del operador (1-B)^d expandido
    como serie binomial, truncados cuando caen debajo del umbral.

    w_k = (-1)^k * C(d, k) = prod_{i=0}^{k-1} (d - i) / (i + 1) * (-1)^k
    """
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
        if k > 10000:  # safeguard
            break

    return np.array(weights[::-1])  # revertir para convolucion


def fractional_diff(
    series: pd.Series,
    d: float,
    weight_cutoff: float = 1e-5,
) -> pd.Series:
    """Aplica diferenciacion fraccional de orden d a una serie.

    Args:
        series: Serie de precios o valores.
        d: Orden de diferenciacion (0 < d < 1 tipicamente).
        weight_cutoff: Umbral para truncar pesos de la expansion.

    Returns:
        Serie diferenciada fraccionalmente (mas corta que la original
        por el ancho de la ventana de pesos).
    """
    weights = _ffd_weights(d, threshold=weight_cutoff)
    width = len(weights)

    if width > len(series):
        log.warning(
            "ffd_window_exceeds_series",
            d=d,
            weight_window=width,
            series_length=len(series),
        )
        return pd.Series(dtype=float)

    # Convolucion con los pesos FFD
    values = series.values
    result = np.full(len(values), np.nan)

    for i in range(width - 1, len(values)):
        window = values[i - width + 1: i + 1]
        result[i] = np.dot(weights, window)

    output = pd.Series(
        result[width - 1:],
        index=series.index[width - 1:],
        name=f"{series.name}_ffd_d{d:.2f}" if series.name else f"ffd_d{d:.2f}",
    )

    return output


def ffd_lookback(d: float, weight_cutoff: float = 1e-5) -> int:
    """Numero de barras previas que usa una feature FFD."""
    return max(len(_ffd_weights(d, threshold=weight_cutoff)) - 1, 0)


def adf_test(series: pd.Series, max_lag: Optional[int] = None) -> Dict:
    """Ejecuta el test Augmented Dickey-Fuller para estacionariedad.

    Returns:
        Dict con: statistic, pvalue, n_lags, n_obs, critical_values, stationary
    """
    clean = series.dropna()
    if len(clean) < 20:
        return {
            "statistic": float("nan"),
            "pvalue": 1.0,
            "n_lags": 0,
            "n_obs": len(clean),
            "critical_values": {},
            "stationary": False,
        }

    result = adfuller(clean.values, maxlag=max_lag, autolag="AIC")

    return {
        "statistic": float(result[0]),
        "pvalue": float(result[1]),
        "n_lags": int(result[2]),
        "n_obs": int(result[3]),
        "critical_values": {k: float(v) for k, v in result[4].items()},
        "stationary": result[1] < 0.05,
    }


def find_optimal_d(
    series: pd.Series,
    d_range: Tuple[float, float] = (0.1, 0.9),
    d_step: float = 0.1,
    threshold_adf_pvalue: float = 0.05,
    weight_cutoff: float = 1e-5,
) -> Dict:
    """Encuentra la d minima que logra estacionariedad.

    Itera d desde d_range[0] hasta d_range[1] en pasos de d_step.
    Para cada d, aplica FFD y ejecuta el test ADF.
    La d optima es la menor d con p-valor < umbral.

    Args:
        series: Serie de precios original.
        d_range: (d_min, d_max) rango de busqueda.
        d_step: Paso de busqueda.
        threshold_adf_pvalue: p-valor maximo para aceptar estacionariedad.
        weight_cutoff: Umbral para truncar pesos FFD.

    Returns:
        Dict con: optimal_d, adf_results (dict por d), series_transformed
    """
    d_values = np.arange(d_range[0], d_range[1] + d_step / 2, d_step)
    adf_results = {}
    optimal_d = None
    optimal_series = None

    for d in d_values:
        d = round(d, 4)
        transformed = fractional_diff(series, d, weight_cutoff=weight_cutoff)

        if len(transformed) < 20:
            log.warning("ffd_too_short", d=d, length=len(transformed))
            continue

        adf = adf_test(transformed)
        adf_results[d] = adf

        log.info(
            "ffd_adf_test",
            d=d,
            adf_pvalue=round(adf["pvalue"], 6),
            stationary=adf["stationary"],
            n_obs=adf["n_obs"],
        )

        if adf["stationary"] and optimal_d is None:
            optimal_d = d
            optimal_series = transformed

    if optimal_d is None:
        log.warning(
            "ffd_no_stationary_d_found",
            d_range=d_range,
            d_step=d_step,
        )
        # Fallback: usar d=1 (diferenciacion entera)
        optimal_d = 1.0
        optimal_series = series.diff().dropna()

    return {
        "optimal_d": optimal_d,
        "adf_results": adf_results,
        "series_transformed": optimal_series,
    }


def build_feature_matrix(
    df: pd.DataFrame,
    price_col: str = "close",
    d: Optional[float] = None,
    d_range: Tuple[float, float] = (0.1, 0.9),
    d_step: float = 0.1,
    threshold_adf_pvalue: float = 0.05,
    weight_cutoff: float = 1e-5,
) -> Tuple[pd.DataFrame, float]:
    """Construye la matriz de features con la serie diferenciada
    fraccionalmente y features adicionales.

    Si d no se especifica, busca la d optima automaticamente.

    Returns:
        (features_df, d_used)
    """
    prices = df[price_col]

    if d is not None:
        transformed = fractional_diff(prices, d, weight_cutoff)
        d_used = d
    else:
        result = find_optimal_d(
            prices,
            d_range=d_range,
            d_step=d_step,
            threshold_adf_pvalue=threshold_adf_pvalue,
            weight_cutoff=weight_cutoff,
        )
        transformed = result["series_transformed"]
        d_used = result["optimal_d"]

    # Construir features
    features = pd.DataFrame(index=transformed.index)
    features["ffd_price"] = transformed

    # Retornos log
    if price_col in df.columns:
        log_ret = np.log(df[price_col] / df[price_col].shift(1))
        features["log_return"] = log_ret.reindex(features.index)

    # Volatilidad rolling (z-scored)
    if "log_return" in features.columns:
        vol = features["log_return"].rolling(20).std()
        vol_mean = vol.expanding().mean()
        vol_std = vol.expanding().std()
        features["vol_zscore"] = (vol - vol_mean) / vol_std.replace(0, np.nan)

    # Volumen z-scored (si disponible)
    if "volume" in df.columns:
        vol_data = df["volume"].reindex(features.index)
        vol_mean = vol_data.expanding().mean()
        vol_std = vol_data.expanding().std()
        features["volume_zscore"] = (
            (vol_data - vol_mean) / vol_std.replace(0, np.nan)
        )

    features = features.dropna()

    log.info(
        "feature_matrix_built",
        n_rows=len(features),
        n_features=len(features.columns),
        d_used=d_used,
        features=list(features.columns),
    )

    return features, d_used


def build_fold_feature_sets(
    df: pd.DataFrame,
    target: pd.Series,
    folds: List["CVFold"],
    price_col: str = "close",
    fixed_d: Optional[float] = None,
    d_range: Tuple[float, float] = (0.1, 0.9),
    d_step: float = 0.1,
    threshold_adf_pvalue: float = 0.05,
    weight_cutoff: float = 1e-5,
) -> List[FoldFeatureSet]:
    """Construye datos por fold sin seleccionar ``d`` sobre el fold de test.

    ``folds`` indexa la serie ``target``. La selección ADF se ejecuta solo
    sobre las observaciones de train del fold; después se transforma la serie
    completa con el ``d`` elegido, usando únicamente valores pasados para cada
    timestamp. Las matrices finales se vuelven a recortar por índice de fold.
    """
    prices = df[price_col]
    sample_index = target.index
    feature_cache: Dict[float, pd.DataFrame] = {}
    fold_sets: List[FoldFeatureSet] = []

    for fold in folds:
        train_pos = fold.train_indices[fold.train_indices < len(sample_index)]
        test_pos = fold.test_indices[fold.test_indices < len(sample_index)]
        # Para el test de un fold, una regla PIT no puede mirar train posterior
        # al inicio del test aunque CPCV lo permita para el ajuste del modelo.
        pre_test_train_pos = train_pos[train_pos < test_pos.min()]
        selection_pos = pre_test_train_pos if len(pre_test_train_pos) >= 20 else train_pos
        train_times = sample_index[selection_pos]

        if fixed_d is None:
            # Los valores usados para decidir d pertenecen exclusivamente a train.
            train_prices = prices.reindex(train_times).dropna()
            selection = find_optimal_d(
                train_prices,
                d_range=d_range,
                d_step=d_step,
                threshold_adf_pvalue=threshold_adf_pvalue,
                weight_cutoff=weight_cutoff,
            )
            d_used = float(selection["optimal_d"])
            selection_method = "rule_based_adf"
        else:
            d_used = float(fixed_d)
            selection_method = "config_fixed"

        if d_used not in feature_cache:
            feature_cache[d_used], _ = build_feature_matrix(
                df,
                price_col=price_col,
                d=d_used,
                weight_cutoff=weight_cutoff,
            )
        features = feature_cache[d_used]

        def select_positions(positions: np.ndarray) -> Tuple[pd.DataFrame, pd.Series]:
            timestamps = sample_index[positions]
            x = features.reindex(timestamps)
            y = target.reindex(timestamps)
            valid = x.notna().all(axis=1) & y.notna()
            return x.loc[valid], y.loc[valid]

        # No se entrena con observaciones posteriores al test. Esto evita que
        # una ventana FFD larga cruce el bloque de test; el purge sigue
        # cubriendo el horizonte de la etiqueta, no el lookback de una feature
        # que ya fue construida de forma causal.
        train_features, train_target = select_positions(pre_test_train_pos)
        test_features, test_target = select_positions(test_pos)

        fold_sets.append(
            FoldFeatureSet(
                fold_id=fold.fold_id,
                train_features=train_features,
                train_target=train_target,
                test_features=test_features,
                test_target=test_target,
                d_used=d_used,
                ffd_lookback=ffd_lookback(d_used, weight_cutoff),
                d_selection_method=selection_method,
            )
        )

        log.info(
            "fold_features_built",
            fold_id=fold.fold_id,
            d_used=d_used,
            ffd_lookback=ffd_lookback(d_used, weight_cutoff),
            train_size=len(train_features),
            test_size=len(test_features),
        )

    return fold_sets
