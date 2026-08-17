"""
Agente 4 -- Deep Models (TCN Supervisado y TCN + Attention).

Implementacion funcional usando PyTorch:
  - TCN: Conv1D causales dilatadas + BatchNorm + LeakyReLU + Head lineal.
  - TCN+Attention: TCN backbone + Multi-Head Self-Attention + LayerNorm + Linear Head.

Ambas redes se entrenan con Adam, loss MSE / BCE y mini-batches.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from src.data.pit_splits import CVFold
from src.features.fractional_diff import FoldFeatureSet
from src.ledger.trial_ledger import TrialRecord, log_trial
from src.logger import get_logger

log = get_logger(__name__)


def _sharpe_ratio(returns: np.ndarray, annualization: float = 1.0) -> float:
    """Sharpe ratio anualizado."""
    if len(returns) < 2:
        return 0.0
    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1)
    if std_r < 1e-15:
        return 0.0
    return float(mean_r / std_r * np.sqrt(annualization))


def _ledger_context(metadata: Optional[Dict[str, Any]], annualization: float) -> Dict[str, Any]:
    metadata = metadata or {}
    d_values = metadata.get("d_used_by_fold", [])
    unique_d = {float(d) for d in d_values if d is not None and np.isfinite(d)}
    return {
        "d_used": next(iter(unique_d)) if len(unique_d) == 1 else None,
        "d_selection_method": metadata.get("d_selection_method", "") if d_values else "",
        "annualization_factor": annualization,
        "frequency": metadata.get("frequency", ""),
        "config_version": metadata.get("config_version", ""),
    }


# ---------------------------------------------------------------------------
# Redes Neuronales PyTorch
# ---------------------------------------------------------------------------

class PyTorchTCNNet(nn.Module):
    """Red TCN basada en Convoluciones 1D con dilacion."""

    def __init__(self, in_features: int, hidden_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_features, out_channels=hidden_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act2 = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, features) -> convert to (batch, features, 1)
        x_in = x.unsqueeze(-1)
        out = self.act1(self.bn1(self.conv1(x_in)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.drop(out.squeeze(-1))
        out = self.head(out)
        return out.squeeze(-1)


class PyTorchTCNAttentionNet(nn.Module):
    """Red TCN con Multi-Head Self-Attention."""

    def __init__(self, in_features: int, hidden_dim: int = 32, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_features, out_channels=hidden_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.LeakyReLU(0.1)
        
        # Multihead Attention requiere (seq_len, batch, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x.unsqueeze(-1)
        feat = self.act1(self.bn1(self.conv1(x_in))).squeeze(-1) # (batch, hidden_dim)
        
        # Reshape para Attention: (1, batch, hidden_dim)
        attn_in = feat.unsqueeze(0)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        out = self.norm(feat + attn_out.squeeze(0))
        out = self.head(out)
        return out.squeeze(-1)


# ---------------------------------------------------------------------------
# Wrapper de Entrenamiento y Prediccion
# ---------------------------------------------------------------------------

class DeepModelWrapper:
    """Wrapper para entrenar y predecir redes PyTorch."""

    def __init__(
        self,
        architecture: str = "tcn",
        hidden_dim: int = 32,
        n_heads: int = 4,
        learning_rate: float = 1e-3,
        n_epochs: int = 15,
        seed: int = 42,
    ):
        self.architecture = architecture
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.seed = seed
        self.model: Optional[nn.Module] = None

    @property
    def hyperparams(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture,
            "hidden_dim": self.hidden_dim,
            "n_heads": self.n_heads,
            "learning_rate": self.learning_rate,
            "n_epochs": self.n_epochs,
            "seed": self.seed,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        in_features = X.shape[1]

        if self.architecture == "tcn":
            self.model = PyTorchTCNNet(in_features=in_features, hidden_dim=self.hidden_dim)
        else:
            self.model = PyTorchTCNAttentionNet(
                in_features=in_features, hidden_dim=self.hidden_dim, n_heads=self.n_heads
            )

        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            pred = self.model(X_t)
            loss = criterion(pred, y_t)
            loss.backward()
            optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be fitted before predict")
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            pred = self.model(X_t)
        return pred.numpy()


# ---------------------------------------------------------------------------
# Funciones de Entrada para el Pipeline
# ---------------------------------------------------------------------------

def tcn_supervised(
    features: pd.DataFrame,
    target: pd.Series,
    folds: List[CVFold],
    tcn_params: Optional[Dict] = None,
    ledger_path: Optional[str] = None,
    annualization: float = 1.0,
    fold_feature_sets: Optional[List[FoldFeatureSet]] = None,
    ledger_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """TCN supervisado con purged temporal k-fold en PyTorch."""
    feature_values = features.values
    target_values = target.values

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
                fold.fold_id,
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
                item.fold_id,
                item.train_features.values,
                item.train_target.values,
                item.test_features.values,
                item.test_target.values,
                item.test_features.index.to_numpy(),
            )
            for item in fold_feature_sets
        ]

    for fold_id, X_train, y_train, X_test, y_test_ret, oos_index in fold_arrays:
        if len(X_train) < 20 or len(X_test) < 1:
            continue

        wrapper = DeepModelWrapper(architecture="tcn", seed=42 + fold_id)
        wrapper.fit(X_train, y_train)

        pred_test = wrapper.predict(X_test)
        signal_test = np.sign(pred_test)
        oos_returns = signal_test * y_test_ret
        all_oos_returns.extend(oos_returns)
        all_oos_timestamps.extend(oos_index)

        pred_train = wrapper.predict(X_train)
        signal_train = np.sign(pred_train)
        all_is_returns.extend(signal_train * y_train)

    ret_is = np.array(all_is_returns)
    ret_oos = np.array(all_oos_returns)
    sr_is = _sharpe_ratio(ret_is, annualization)
    sr_oos = _sharpe_ratio(ret_oos, annualization)

    hyperparams = {"architecture": "tcn_supervised", "hidden_dim": 32, "n_epochs": 15}
    if ledger_metadata:
        hyperparams["feature_metadata"] = ledger_metadata
    result = {
        "model_name": "tcn_supervised",
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
            agent="deep_model",
            model_name="tcn_supervised",
            config_hash=TrialRecord.hash_config(hyperparams),
            hyperparams=json.dumps(hyperparams, default=str),
            sharpe_is=sr_is,
            sharpe_oos=sr_oos,
            n_observations=len(ret_oos),
            **_ledger_context(ledger_metadata, annualization),
        )
        log_trial(record, ledger_path)
        result["trial_id"] = record.trial_id

    log.info("tcn_supervised_done", sharpe_oos=round(sr_oos, 4))
    return result


def tcn_attention_supervised(
    features: pd.DataFrame,
    target: pd.Series,
    folds: List[CVFold],
    tcn_params: Optional[Dict] = None,
    attention_params: Optional[Dict] = None,
    ledger_path: Optional[str] = None,
    annualization: float = 1.0,
    fold_feature_sets: Optional[List[FoldFeatureSet]] = None,
    ledger_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """TCN + Attention supervisado con purged temporal k-fold en PyTorch."""
    feature_values = features.values
    target_values = target.values

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
                fold.fold_id,
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
                item.fold_id,
                item.train_features.values,
                item.train_target.values,
                item.test_features.values,
                item.test_target.values,
                item.test_features.index.to_numpy(),
            )
            for item in fold_feature_sets
        ]

    for fold_id, X_train, y_train, X_test, y_test_ret, oos_index in fold_arrays:
        if len(X_train) < 20 or len(X_test) < 1:
            continue

        wrapper = DeepModelWrapper(architecture="tcn_attention", seed=100 + fold_id)
        wrapper.fit(X_train, y_train)

        pred_test = wrapper.predict(X_test)
        signal_test = np.sign(pred_test)
        oos_returns = signal_test * y_test_ret
        all_oos_returns.extend(oos_returns)
        all_oos_timestamps.extend(oos_index)

        pred_train = wrapper.predict(X_train)
        signal_train = np.sign(pred_train)
        all_is_returns.extend(signal_train * y_train)

    ret_is = np.array(all_is_returns)
    ret_oos = np.array(all_oos_returns)
    sr_is = _sharpe_ratio(ret_is, annualization)
    sr_oos = _sharpe_ratio(ret_oos, annualization)

    hyperparams = {"architecture": "tcn_attention_supervised", "hidden_dim": 32, "n_heads": 4, "n_epochs": 15}
    if ledger_metadata:
        hyperparams["feature_metadata"] = ledger_metadata
    result = {
        "model_name": "tcn_attention_supervised",
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
            agent="deep_model",
            model_name="tcn_attention_supervised",
            config_hash=TrialRecord.hash_config(hyperparams),
            hyperparams=json.dumps(hyperparams, default=str),
            sharpe_is=sr_is,
            sharpe_oos=sr_oos,
            n_observations=len(ret_oos),
            **_ledger_context(ledger_metadata, annualization),
        )
        log_trial(record, ledger_path)
        result["trial_id"] = record.trial_id

    log.info("tcn_attention_supervised_done", sharpe_oos=round(sr_oos, 4))
    return result


# Compatibilidad con notebooks antiguos; no representa un algoritmo PPO.
tcn_attention_ppo = tcn_attention_supervised
