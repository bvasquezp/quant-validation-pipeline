"""
Agente 3 -- Modelos baseline.

Buy&Hold, SMA cross, Ridge y LightGBM.
Cada modelo entrenado/evaluado registra un trial en el Ledger.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.data.pit_splits import CVFold
from src.features.fractional_diff import FoldFeatureSet
from src.ledger.trial_ledger import TrialRecord, log_trial
from src.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _sharpe_ratio(returns: np.ndarray, annualization: float = 1.0) -> float:
    """Sharpe ratio anualizado."""
    if len(returns) < 2:
        return 0.0
    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1)
    if std_r < 1e-15:
        return 0.0
    return float(mean_r / std_r * np.sqrt(annualization))


def _ledger_context(
    metadata: Optional[Dict[str, Any]],
    annualization: float,
) -> Dict[str, Any]:
    metadata = metadata or {}
    d_values = metadata.get("d_used_by_fold", [])
    unique_d = {float(d) for d in d_values}
    return {
        "d_used": next(iter(unique_d)) if len(unique_d) == 1 else None,
        "d_selection_method": metadata.get("d_selection_method", "") if d_values else "",
        "annualization_factor": annualization,
        "frequency": metadata.get("frequency", ""),
        "config_version": metadata.get("config_version", ""),
    }


# ---------------------------------------------------------------------------
# Baseline 0: Buy & Hold
# ---------------------------------------------------------------------------

def buy_and_hold(
    returns: pd.Series,
    ledger_path: Optional[str] = None,
    annualization: float = 1.0,
    ledger_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Baseline 0: mantener posicion larga todo el periodo.

    No hay entrenamiento -- los retornos son simplemente los retornos
    del activo subyacente.
    """
    ret = returns.values
    sr = _sharpe_ratio(ret, annualization)

    result = {
        "model_name": "buy_and_hold",
        "returns_oos": ret,
        "oos_timestamps": returns.index.to_numpy(),
        "sharpe_oos": sr,
        "sharpe_is": sr,  # igual porque no hay in-sample
        "n_observations": len(ret),
        "hyperparams": {},
    }

    if ledger_path:
        record = TrialRecord(
            agent="baseline",
            model_name="buy_and_hold",
            config_hash=TrialRecord.hash_config({}),
            hyperparams=json.dumps({}),
            sharpe_is=sr,
            sharpe_oos=sr,
            n_observations=len(ret),
            **_ledger_context(ledger_metadata, annualization),
        )
        log_trial(record, ledger_path)
        result["trial_id"] = record.trial_id

    log.info("baseline_buy_hold", sharpe=round(sr, 4), n_obs=len(ret))
    return result


# ---------------------------------------------------------------------------
# Baseline 1: SMA Crossover
# ---------------------------------------------------------------------------

def sma_crossover(
    prices: pd.Series,
    returns: pd.Series,
    sma_short: int = 20,
    sma_long: int = 50,
    ledger_path: Optional[str] = None,
    annualization: float = 1.0,
    ledger_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Baseline 1: cruce de medias moviles simples.

    Señal: long si SMA_short > SMA_long, flat (o short) de lo contrario.
    Los retornos de la estrategia = señal * retornos del activo.
    """
    sma_s = prices.rolling(sma_short).mean()
    sma_l = prices.rolling(sma_long).mean()

    # Señal: +1 cuando SMA corta > SMA larga, -1 de lo contrario
    signal = np.where(sma_s > sma_l, 1.0, -1.0)
    signal = pd.Series(signal, index=prices.index)

    # Retornos de la estrategia (señal del periodo anterior * retorno actual)
    strategy_returns = signal.shift(1) * returns
    strategy_returns = strategy_returns.dropna()
    ret = strategy_returns.values

    sr = _sharpe_ratio(ret, annualization)

    hyperparams = {"sma_short": sma_short, "sma_long": sma_long}
    result = {
        "model_name": "sma_crossover",
        "returns_oos": ret,
        "oos_timestamps": strategy_returns.index.to_numpy(),
        "sharpe_oos": sr,
        "sharpe_is": sr,
        "n_observations": len(ret),
        "hyperparams": hyperparams,
    }

    if ledger_path:
        record = TrialRecord(
            agent="baseline",
            model_name="sma_crossover",
            config_hash=TrialRecord.hash_config(hyperparams),
            hyperparams=json.dumps(hyperparams),
            sharpe_is=sr,
            sharpe_oos=sr,
            n_observations=len(ret),
            **_ledger_context(ledger_metadata, annualization),
        )
        log_trial(record, ledger_path)
        result["trial_id"] = record.trial_id

    log.info("baseline_sma_crossover", sharpe=round(sr, 4), n_obs=len(ret))
    return result


# ---------------------------------------------------------------------------
# Baseline 2: Ridge Regression
# ---------------------------------------------------------------------------

def ridge_regression(
    features: pd.DataFrame,
    target: pd.Series,
    folds: List[CVFold],
    alpha: float = 1.0,
    ledger_path: Optional[str] = None,
    annualization: float = 1.0,
    fold_feature_sets: Optional[List[FoldFeatureSet]] = None,
    ledger_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Baseline 2: Ridge regression con CPCV.

    Usa las features como entrada y los retornos futuros como target.
    La prediccion se convierte en señal: long si prediccion > 0, short si < 0.
    """
    feature_values = features.values
    target_values = target.values
    scaler = StandardScaler()

    all_oos_returns = []
    all_is_returns = []
    all_oos_timestamps = []

    if fold_feature_sets is None:
        fold_arrays = []
        for fold in folds:
            train_mask = fold.train_indices[fold.train_indices < len(feature_values)]
            test_mask = fold.test_indices[fold.test_indices < len(feature_values)]
            fold_arrays.append(
                (
                    feature_values[train_mask],
                    target_values[train_mask],
                    feature_values[test_mask],
                    target_values[test_mask],
                    features.index[test_mask].to_numpy(),
                )
            )
    else:
        fold_arrays = [
            (
                item.train_features.values,
                item.train_target.values,
                item.test_features.values,
                item.test_target.values,
                item.test_features.index.to_numpy(),
            )
            for item in fold_feature_sets
        ]

    for X_train_raw, y_train, X_test_raw, y_test, oos_index in fold_arrays:
        if len(X_train_raw) < 5 or len(X_test_raw) < 1:
            continue

        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        # Predicciones -> señal
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

        signal_train = np.sign(pred_train)
        signal_test = np.sign(pred_test)

        is_returns = signal_train * y_train
        oos_returns = signal_test * y_test

        all_is_returns.extend(is_returns)
        all_oos_returns.extend(oos_returns)
        all_oos_timestamps.extend(oos_index)

    ret_is = np.array(all_is_returns)
    ret_oos = np.array(all_oos_returns)
    sr_is = _sharpe_ratio(ret_is, annualization)
    sr_oos = _sharpe_ratio(ret_oos, annualization)

    hyperparams = {"alpha": alpha, "n_folds": len(folds)}
    if ledger_metadata:
        hyperparams["feature_metadata"] = ledger_metadata
    result = {
        "model_name": "ridge_regression",
        "returns_oos": ret_oos,
        "oos_timestamps": np.asarray(all_oos_timestamps),
        "returns_is": ret_is,
        "sharpe_oos": sr_oos,
        "sharpe_is": sr_is,
        "n_observations": len(ret_oos),
        "hyperparams": hyperparams,
    }

    if ledger_path:
        record = TrialRecord(
            agent="baseline",
            model_name="ridge_regression",
            config_hash=TrialRecord.hash_config(hyperparams),
            hyperparams=json.dumps(hyperparams),
            sharpe_is=sr_is,
            sharpe_oos=sr_oos,
            n_observations=len(ret_oos),
            **_ledger_context(ledger_metadata, annualization),
        )
        log_trial(record, ledger_path)
        result["trial_id"] = record.trial_id

    log.info(
        "baseline_ridge",
        sharpe_is=round(sr_is, 4),
        sharpe_oos=round(sr_oos, 4),
        n_obs=len(ret_oos),
    )
    return result


# ---------------------------------------------------------------------------
# Baseline 3: LightGBM
# ---------------------------------------------------------------------------

def lightgbm_classifier(
    features: pd.DataFrame,
    target: pd.Series,
    folds: List[CVFold],
    lgbm_params: Optional[Dict] = None,
    ledger_path: Optional[str] = None,
    annualization: float = 1.0,
    fold_feature_sets: Optional[List[FoldFeatureSet]] = None,
    ledger_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Baseline 3: LightGBM con CPCV.

    Clasifica la direccion del retorno (up/down) y genera señales.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        log.error("lightgbm_not_installed")
        raise

    default_params = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "objective": "binary",
        "verbose": -1,
        "n_jobs": 1,
    }
    if lgbm_params:
        default_params.update(lgbm_params)

    feature_values = features.values
    target_values = target.values

    # Binarizar target: 1 si retorno > 0, 0 si <= 0
    target_binary = (target_values > 0).astype(int)

    all_oos_returns = []
    all_is_returns = []
    all_oos_timestamps = []

    if fold_feature_sets is None:
        fold_arrays = []
        for fold in folds:
            train_mask = fold.train_indices[fold.train_indices < len(feature_values)]
            test_mask = fold.test_indices[fold.test_indices < len(feature_values)]
            fold_arrays.append(
                (
                    feature_values[train_mask],
                    target_values[train_mask],
                    feature_values[test_mask],
                    target_values[test_mask],
                    features.index[test_mask].to_numpy(),
                )
            )
    else:
        fold_arrays = [
            (
                item.train_features.values,
                item.train_target.values,
                item.test_features.values,
                item.test_target.values,
                item.test_features.index.to_numpy(),
            )
            for item in fold_feature_sets
        ]

    for X_train, y_train_ret, X_test, y_test_ret, oos_index in fold_arrays:
        if len(X_train) < 20 or len(X_test) < 1:
            continue

        y_train = (y_train_ret > 0).astype(int)

        model = lgb.LGBMClassifier(**default_params)
        model.fit(X_train, y_train)

        # Probabilidades -> señal
        prob_train = model.predict_proba(X_train)[:, 1]
        prob_test = model.predict_proba(X_test)[:, 1]

        # Señal: (prob - 0.5) * 2 para escalar a [-1, 1]
        signal_test = (prob_test - 0.5) * 2

        oos_returns = signal_test * y_test_ret
        all_oos_returns.extend(oos_returns)
        all_oos_timestamps.extend(oos_index)

        # In-sample
        signal_train = (prob_train - 0.5) * 2
        is_returns = signal_train * y_train_ret
        all_is_returns.extend(is_returns)

    ret_is = np.array(all_is_returns)
    ret_oos = np.array(all_oos_returns)
    sr_is = _sharpe_ratio(ret_is, annualization)
    sr_oos = _sharpe_ratio(ret_oos, annualization)

    # Limpiar params para serializar
    serializable_params = {
        k: v for k, v in default_params.items()
        if k != "verbose"
    }
    if ledger_metadata:
        serializable_params["feature_metadata"] = ledger_metadata
    result = {
        "model_name": "lightgbm",
        "returns_oos": ret_oos,
        "oos_timestamps": np.asarray(all_oos_timestamps),
        "returns_is": ret_is,
        "sharpe_oos": sr_oos,
        "sharpe_is": sr_is,
        "n_observations": len(ret_oos),
        "hyperparams": serializable_params,
    }

    if ledger_path:
        record = TrialRecord(
            agent="baseline",
            model_name="lightgbm",
            config_hash=TrialRecord.hash_config(serializable_params),
            hyperparams=json.dumps(serializable_params, default=str),
            sharpe_is=sr_is,
            sharpe_oos=sr_oos,
            n_observations=len(ret_oos),
            **_ledger_context(ledger_metadata, annualization),
        )
        log_trial(record, ledger_path)
        result["trial_id"] = record.trial_id

    log.info(
        "baseline_lightgbm",
        sharpe_is=round(sr_is, 4),
        sharpe_oos=round(sr_oos, 4),
        n_obs=len(ret_oos),
    )
    return result
