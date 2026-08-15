"""
Agente 0 -- Orchestrator.

Coordina el DAG completo del pipeline de validacion.
Mantiene el Trial Ledger, aplica reintentos y genera el log de ejecucion.

Flujo:
  1. Data Agent      -> genera folds CPCV
  2. Feature Agent   -> genera matriz de features
  3. Baseline Agent  -> Baseline 0, 1, 2, 3 -> cada uno logea trial
  4. Deep/RL Agents -> TCN supervisado, Attention supervisado, PPO real
  5. Statistical Validation Agent -> DSR/PSR para CADA etapa
  6. Ablation Agent  -> bootstrap secuencial
  7. Reporting Agent -> decision GO/NO-GO
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import (
    PipelineConfig,
    annualization_factor_from_frequency,
    load_config,
    set_global_seed,
)
from src.data.ingest import build_data_manifest, compute_returns, load_csv, save_data_manifest
from src.data.pit_splits import combinatorial_purged_cv, verify_no_overlap
from src.features.fractional_diff import build_fold_feature_sets
from src.ledger.trial_ledger import (
    TrialRecord,
    count_trials,
    get_all_trials,
    get_sharpe_oos_series,
    log_trial,
)
from src.logger import get_logger, setup_logging
from src.models.baselines import (
    buy_and_hold,
    lightgbm_classifier,
    ridge_regression,
    sma_crossover,
)
from src.models.deep_model import tcn_attention_supervised, tcn_supervised
from src.models.rl import ppo_trading
from src.stats.bootstrap import ablation_ladder
from src.stats.effective_trials import estimate_pairwise_correlation
from src.stats.psr_dsr import deflated_sharpe_ratio

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Resultado del pipeline
# ---------------------------------------------------------------------------

class PipelineResult:
    """Resultado completo del pipeline."""

    def __init__(self):
        self.stage_results: Dict[str, Dict] = {}
        self.dsr_results: Dict[str, Dict] = {}
        self.ablation_results: Dict[str, Dict] = {}
        self.gate_decision: str = "NO-GO"
        self.gate_reasons: List[str] = []
        self.recommended_stage: str = ""
        self.execution_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "gate_decision": self.gate_decision,
            "gate_reasons": self.gate_reasons,
            "recommended_stage": self.recommended_stage,
            "execution_time_seconds": self.execution_time_seconds,
            "dsr_results": {},
            "ablation_results": {},
        }
        for name, dsr in self.dsr_results.items():
            result["dsr_results"][name] = {
                k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in dsr.items()
            }
        for name, abl in self.ablation_results.items():
            if isinstance(abl, dict):
                result["ablation_results"][name] = {
                    k: (round(v, 6) if isinstance(v, (int, float)) else v)
                    for k, v in abl.items()
                }
            else:
                result["ablation_results"][name] = abl
        return result

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_pipeline(
    config: Optional[PipelineConfig] = None,
    config_path: Optional[str] = None,
    data_path: Optional[str] = None,
    mu_override: Optional[float] = None,
    force_stage2: bool = False,
) -> PipelineResult:
    """Ejecuta el pipeline completo de validacion.

    Args:
        config: Configuracion pre-cargada (tiene prioridad).
        config_path: Ruta al YAML de configuracion.
        data_path: Ruta al CSV de datos (override del config).
        mu_override: Drift para datos sinteticos (usar 0.0 para placebo test).
        force_stage2: Ejecuta TCNs aunque la ablacion Stage 0 -> 1 se detenga.

    Returns:
        PipelineResult con todos los resultados y la decision GO/NO-GO.
    """
    start_time = time.time()
    result = PipelineResult()

    # --- Configuracion ---
    if config is None:
        config = load_config(config_path)

    setup_logging(config.logging.level, config.logging.format)
    set_global_seed(config.reproducibility.global_seed)
    annualization_factor = annualization_factor_from_frequency(
        config.data.frequency,
        calendar=config.data.calendar,
    )
    ledger_metadata = {
        "frequency": config.data.frequency,
        "annualization_factor": annualization_factor,
        "config_version": config.config_version,
    }
    base_ledger_metadata = dict(ledger_metadata)

    ledger_path = config.ledger.db_path
    log.info("pipeline_started", config=config.model_dump())

    # --- Agente 1: Data ---
    log.info("stage_data_start")
    actual_data_path = data_path or config.data.path

    using_synthetic = False
    try:
        df = load_csv(
            actual_data_path,
            frequency=config.data.frequency,
            calendar=config.data.calendar,
            timezone=config.data.timezone,
            symbol=config.data.symbol,
            allowed_gap_dates=config.data.allowed_gap_dates,
        )
    except FileNotFoundError:
        using_synthetic = True
        log.warning(
            "data_file_not_found_generating_synthetic",
            path=actual_data_path,
        )
        df = _generate_synthetic_data(config, mu=mu_override if mu_override is not None else 0.05)

    if not using_synthetic:
        manifest = build_data_manifest(
            actual_data_path,
            df,
            symbol=config.data.symbol,
            frequency=config.data.frequency,
            calendar=config.data.calendar,
            timezone=config.data.timezone,
        )
        save_data_manifest(manifest, config.data.manifest_path)
        df, reserved_oos = _split_real_data(df, config)
        log.info("final_oos_reserved", **reserved_oos)

    prices = df["close"]
    returns = compute_returns(prices, method="log")

    # Target: retorno futuro de una barra, definido antes de construir features.
    target = returns.shift(-1).dropna()

    # Generar folds CPCV sobre el universo de targets.
    folds = combinatorial_purged_cv(
        n_samples=len(target),
        n_folds=config.splits.n_folds,
        purge_window=config.splits.purge_window,
        embargo_window=config.splits.embargo_window,
    )

    if not verify_no_overlap(folds):
        raise RuntimeError(
            "CRITICAL: Overlap detected in CPCV folds. "
            "All results would be invalid."
        )

    log.info("stage_data_done", n_folds=len(folds), n_returns=len(returns))

    # --- Agente 2: Features ---
    log.info("stage_features_start")
    feat_cfg = config.features.fractional_diff

    fold_feature_sets = _build_fold_features(df, target, folds, feat_cfg)
    d_values = [item.d_used for item in fold_feature_sets]
    lookbacks = [item.ffd_lookback for item in fold_feature_sets]
    required_purge = max(lookbacks, default=config.splits.purge_window)
    if required_purge > config.splits.purge_window and not using_synthetic:
        folds = combinatorial_purged_cv(
            n_samples=len(target),
            n_folds=config.splits.n_folds,
            purge_window=required_purge,
            embargo_window=config.splits.embargo_window,
        )
        if not verify_no_overlap(folds):
            raise RuntimeError("Dynamic FFD purge produced overlapping folds")
        fold_feature_sets = _build_fold_features(df, target, folds, feat_cfg)
        d_values = [item.d_used for item in fold_feature_sets]
        lookbacks = [item.ffd_lookback for item in fold_feature_sets]
        if max(lookbacks, default=0) > required_purge:
            raise RuntimeError("FFD lookback exceeds the effective purge window")
        ledger_metadata.update(
            {
                "d_used_by_fold": d_values,
                "ffd_lookback_by_fold": lookbacks,
            }
        )
    ledger_metadata.update(
        {
            "d_used_by_fold": d_values,
            "ffd_lookback_by_fold": lookbacks,
            "d_selection_method": "rule_based_adf",
        }
    )
    if lookbacks and max(lookbacks) > max(
        config.splits.purge_window,
        config.splits.embargo_window,
    ):
        log.info(
            "ffd_lookback_causality_verified_without_overlap",
            max_ffd_lookback=max(lookbacks),
            purge_window=config.splits.purge_window,
            embargo_window=config.splits.embargo_window,
            train_policy="pre_test_only" if using_synthetic else "dynamic_purge",
        )

    non_empty = next(
        (
            item.train_features
            for item in fold_feature_sets
            if not item.train_features.empty
        ),
        pd.DataFrame(),
    )
    features = non_empty

    log.info(
        "stage_features_done",
        d_used_by_fold=d_values,
        ffd_lookback_by_fold=lookbacks,
        n_features=features.shape[1] if not features.empty else 0,
        n_samples=len(target),
    )

    # --- Stage 0: Naive / Rule-Based Models ---
    log.info("stage_0_naive_models_start")
    bh_result = buy_and_hold(
        returns,
        ledger_path=ledger_path,
        annualization=annualization_factor,
        ledger_metadata=base_ledger_metadata,
    )
    result.stage_results["buy_hold"] = bh_result

    sma_result = sma_crossover(
        prices, returns,
        sma_short=config.models.baselines.sma_short,
        sma_long=config.models.baselines.sma_long,
        ledger_path=ledger_path,
        annualization=annualization_factor,
        ledger_metadata=base_ledger_metadata,
    )
    result.stage_results["sma_crossover"] = sma_result

    # Campeon Etapa 0 (mayor Sharpe OOS)
    stage0_candidates = [bh_result, sma_result]
    stage0_champion = max(stage0_candidates, key=lambda x: x.get("sharpe_oos", -999))
    log.info("stage_0_champion_selected", champion=stage0_champion["model_name"], sharpe=round(stage0_champion["sharpe_oos"], 4))

    # --- Stage 1: Linear / ML Baseline Models ---
    log.info("stage_1_ml_baselines_start")
    ridge_result = ridge_regression(
        features, target, folds,
        alpha=config.models.baselines.ridge_alpha,
        ledger_path=ledger_path,
        annualization=annualization_factor,
        fold_feature_sets=fold_feature_sets,
        ledger_metadata=ledger_metadata,
    )
    result.stage_results["ridge"] = ridge_result

    lgbm_params = config.models.baselines.lgbm_params.model_dump()
    lgbm_result = lightgbm_classifier(
        features, target, folds,
        lgbm_params=lgbm_params,
        ledger_path=ledger_path,
        annualization=annualization_factor,
        fold_feature_sets=fold_feature_sets,
        ledger_metadata=ledger_metadata,
    )
    result.stage_results["lightgbm"] = lgbm_result

    # Campeon Etapa 1
    stage1_candidates = [ridge_result, lgbm_result]
    stage1_champion = max(stage1_candidates, key=lambda x: x.get("sharpe_oos", -999))
    log.info("stage_1_champion_selected", champion=stage1_champion["model_name"], sharpe=round(stage1_champion["sharpe_oos"], 4))

    # --- Ablacion Stage 0 vs Stage 1 ---
    stage0_label = f"Stage 0 ({stage0_champion['model_name']})"
    stage1_label = f"Stage 1 ({stage1_champion['model_name']})"
    champions_by_stage = {
        stage0_label: stage0_champion["returns_oos"],
        stage1_label: stage1_champion["returns_oos"],
    }
    candidates_by_stage = {
        stage0_label: {
            bh_result["model_name"]: _candidate_payload(bh_result),
            sma_result["model_name"]: _candidate_payload(sma_result),
        },
        stage1_label: {
            ridge_result["model_name"]: _candidate_payload(ridge_result),
            lgbm_result["model_name"]: _candidate_payload(lgbm_result),
        },
    }

    ablation_0_1 = ablation_ladder(
        returns_by_stage=champions_by_stage,
        candidate_returns_by_stage=candidates_by_stage,
        n_boot=config.stats.bootstrap.n_boot,
        block_size=config.stats.bootstrap.block_size,
        significance_level=config.stats.psr_dsr.significance_level,
        seed=config.reproducibility.global_seed,
        annualization=annualization_factor,
    )
    result.ablation_results.update(ablation_0_1)

    # Verificar si se justifica avanzar a Stage 2 (Deep Models)
    advance_to_stage2 = force_stage2 or (
        ablation_0_1.get("recommended_stage") == list(champions_by_stage.keys())[1]
    )

    if advance_to_stage2:
        log.info("stage_gate_passed_advancing_to_deep_models")
        tcn_params = config.models.deep.tcn.model_dump()
        tcn_result = tcn_supervised(
            features, target, folds,
            tcn_params=tcn_params,
            ledger_path=ledger_path,
            annualization=annualization_factor,
            fold_feature_sets=fold_feature_sets,
            ledger_metadata=ledger_metadata,
        )
        result.stage_results["tcn_supervised"] = tcn_result

        att_params = config.models.deep.attention.model_dump()
        tcn_att_result = tcn_attention_supervised(
            features, target, folds,
            tcn_params=tcn_params,
            attention_params=att_params,
            ledger_path=ledger_path,
            annualization=annualization_factor,
            fold_feature_sets=fold_feature_sets,
            ledger_metadata=ledger_metadata,
        )
        result.stage_results["tcn_attention_supervised"] = tcn_att_result

        stage2_candidates = [tcn_result, tcn_att_result]
        if config.models.rl.enabled:
            ppo_result = ppo_trading(
                fold_feature_sets,
                config.models.rl,
                seed=config.reproducibility.global_seed,
                ledger_path=ledger_path,
                ledger_metadata=ledger_metadata,
                annualization=annualization_factor,
            )
            result.stage_results["ppo_trading"] = ppo_result
            stage2_candidates.append(ppo_result)

        stage2_champion = max(stage2_candidates, key=lambda x: x.get("sharpe_oos", -999))
        log.info("stage_2_champion_selected", champion=stage2_champion["model_name"], sharpe=round(stage2_champion["sharpe_oos"], 4))

        # Ablacion Stage 1 vs Stage 2
        stage1_label = f"Stage 1 ({stage1_champion['model_name']})"
        stage2_label = f"Stage 2 ({stage2_champion['model_name']})"
        champions_by_stage_full = {
            stage1_label: stage1_champion["returns_oos"],
            stage2_label: stage2_champion["returns_oos"],
        }
        stage2_candidate_map = {
            item["model_name"]: _candidate_payload(item)
            for item in stage2_candidates
        }
        candidates_by_stage_full = {
            stage1_label: {
                ridge_result["model_name"]: _candidate_payload(ridge_result),
                lgbm_result["model_name"]: _candidate_payload(lgbm_result),
            },
            stage2_label: stage2_candidate_map,
        }
        ablation_1_2 = ablation_ladder(
            returns_by_stage=champions_by_stage_full,
            candidate_returns_by_stage=candidates_by_stage_full,
            n_boot=config.stats.bootstrap.n_boot,
            block_size=config.stats.bootstrap.block_size,
            significance_level=config.stats.psr_dsr.significance_level,
            seed=config.reproducibility.global_seed,
            annualization=annualization_factor,
        )
        result.ablation_results.update(ablation_1_2)
        final_recommended = ablation_1_2.get("recommended_stage")
    else:
        log.info("stage_gate_stopped_at_stage_1", reason="Stage 1 did not significantly outperform Stage 0")
        final_recommended = list(champions_by_stage.keys())[0]

    result.recommended_stage = final_recommended

    # --- Agente 5: Statistical Validation (DSR/PSR) ---
    log.info("stage_statistical_validation_start")

    all_trials = get_all_trials(ledger_path)
    accumulated_returns = []

    for stage_name, stage_data in result.stage_results.items():
        ret_oos = stage_data.get("returns_oos", np.array([]))
        sr_oos = stage_data.get("sharpe_oos", 0.0)

        if len(ret_oos) < 3:
            continue

        accumulated_returns.append(ret_oos)

        trial_id = stage_data.get("trial_id", None)
        cutoff = None
        if trial_id:
            matching = [t for t in all_trials if t.trial_id == trial_id]
            if matching:
                cutoff = matching[0].timestamp

        v_nominal = count_trials(ledger_path, cutoff_timestamp=cutoff)
        v_nominal = max(v_nominal, 1)

        avg_corr = 0.0
        if len(accumulated_returns) >= 2:
            min_len = min(len(r) for r in accumulated_returns)
            if min_len > 10:
                ret_matrix = np.vstack([r[:min_len] for r in accumulated_returns])
                avg_corr = estimate_pairwise_correlation(ret_matrix)

        dsr_result = deflated_sharpe_ratio(
            sr_hat=sr_oos,
            returns=ret_oos,
            n_trials=v_nominal,
            avg_pairwise_corr=avg_corr,
            max_lag_autocorr=config.stats.psr_dsr.max_lag_autocorr,
            annualization_factor=annualization_factor,
        )
        result.dsr_results[stage_name] = dsr_result

    log.info("stage_statistical_validation_done")

    # --- Agente 7: Gate Decision ---
    log.info("stage_gate_start")
    gate_result = _gate_decision(result, config)
    result.gate_decision = gate_result["decision"]
    result.gate_reasons = gate_result["reasons"]
    result.execution_time_seconds = time.time() - start_time

    log.info(
        "pipeline_completed",
        decision=result.gate_decision,
        recommended_stage=result.recommended_stage,
        execution_time=round(result.execution_time_seconds, 2),
    )

    return result


def _gate_decision(
    result: PipelineResult,
    config: PipelineConfig,
) -> Dict[str, Any]:
    reasons = []
    decision = "GO"

    recommended = result.recommended_stage
    if not recommended or recommended == "unknown":
        return {"decision": "NO-GO", "reasons": ["No recommended stage"]}

    # ``recommended_stage`` incluye la etiqueta humana de la etapa
    # (``Stage 0 (buy_hold)``), mientras ``dsr_results`` usa la clave del
    # modelo (``buy_hold``). Resolver ambas formas evita que el gate trate
    # un DSR válido como ausente.
    dsr_data = result.dsr_results.get(recommended, {})
    if not dsr_data and "(" in recommended and recommended.endswith(")"):
        model_name = recommended.rsplit("(", 1)[1][:-1]
        dsr_data = result.dsr_results.get(model_name, {})
        if not dsr_data:
            stage_key = next(
                (
                    key
                    for key, stage_data in result.stage_results.items()
                    if stage_data.get("model_name") == model_name
                ),
                None,
            )
            if stage_key:
                dsr_data = result.dsr_results.get(stage_key, {})
    dsr_val = dsr_data.get("dsr", 0.0)

    if isinstance(dsr_val, float) and not np.isnan(dsr_val):
        if dsr_val >= config.gate.min_dsr:
            reasons.append(
                f"DSR of {recommended} = {dsr_val:.4f} >= {config.gate.min_dsr} [PASS]"
            )
        else:
            decision = "NO-GO"
            reasons.append(
                f"DSR of {recommended} = {dsr_val:.4f} < {config.gate.min_dsr} [FAIL]"
            )
    else:
        decision = "NO-GO"
        reasons.append(f"DSR not available for {recommended} [FAIL]")

    if config.gate.require_ablation_significance:
        for comp_name, comp_data in result.ablation_results.items():
            if comp_name == "recommended_stage":
                continue
            if not isinstance(comp_data, dict):
                continue
            if comp_data.get("decision") == "SKIPPED":
                continue

            p_val = comp_data.get("p_value", 1.0)
            comp_decision = comp_data.get("decision", "UNKNOWN")

            if comp_decision == "ADVANCE" and p_val < config.gate.max_pvalue_ablation:
                reasons.append(f"Ablation {comp_name}: p={p_val:.4f} [PASS]")
            elif comp_decision == "STOP":
                reasons.append(
                    f"Ablation {comp_name}: STOPPED (p={p_val:.4f}) [INFO]"
                )

    if dsr_data:
        reasons.append(
            f"T_effective={dsr_data.get('t_effective', 'N/A')}, "
            f"V_nominal={dsr_data.get('v_nominal', 'N/A')}, "
            f"V_effective={dsr_data.get('v_effective', 'N/A')}, "
            f"SR*={dsr_data.get('sr_star', 'N/A')}"
        )

    return {"decision": decision, "reasons": reasons}


def _split_real_data(
    df: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, Dict[str, str]]:
    """Reserva el periodo final OOS y devuelve solo el periodo de desarrollo."""
    start = pd.Timestamp(config.data.train_start, tz=config.data.timezone)
    train_end = pd.Timestamp(config.data.train_end, tz=config.data.timezone)
    test_start = pd.Timestamp(config.data.test_start, tz=config.data.timezone)
    test_end = pd.Timestamp(config.data.test_end, tz=config.data.timezone)
    if not (start <= train_end < test_start <= test_end):
        raise ValueError("Temporal boundaries must satisfy train < test and be ordered")

    train = df.loc[start:train_end]
    test = df.loc[test_start:test_end]
    if train.empty or test.empty:
        raise ValueError("Configured train/test windows are not covered by the dataset")
    return train, {
        "train_start": str(train.index.min()),
        "train_end": str(train.index.max()),
        "test_start": str(test.index.min()),
        "test_end": str(test.index.max()),
        "test_rows": str(len(test)),
    }


def _build_fold_features(df, target, folds, feat_cfg):
    return build_fold_feature_sets(
        df,
        target,
        folds,
        price_col="close",
        d_range=tuple(feat_cfg.d_range),
        d_step=feat_cfg.d_step,
        threshold_adf_pvalue=feat_cfg.threshold_adf_pvalue,
        weight_cutoff=feat_cfg.weight_cutoff,
    )


def _candidate_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "returns": result.get("returns_oos", np.array([])),
        "timestamps": result.get("oos_timestamps"),
    }


def _generate_synthetic_data(
    config: PipelineConfig,
    n_bars: int = 5000,
    mu: float = 0.05,
) -> pd.DataFrame:
    """Genera datos OHLCV sinteticos para testing.

    Args:
        config: PipelineConfig.
        n_bars: Numero de barras.
        mu: Drift anual (usar 0.0 para placebo test sin sesgo).
    """
    rng = np.random.default_rng(config.reproducibility.global_seed)

    annualization_factor = annualization_factor_from_frequency(
        config.data.frequency,
        calendar=config.data.calendar,
    )
    dt = 1 / annualization_factor
    sigma = 0.20

    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.normal(size=n_bars)
    prices = 100 * np.exp(np.cumsum(log_returns))

    noise = rng.uniform(0.001, 0.005, size=n_bars)
    candidate_count = n_bars
    if config.data.calendar == "fx_24_5":
        # Generate weekday bars plus the Sunday-evening reopen; the gap after
        # Friday is part of the FX data contract
        # part of the FX data contract and must also exist in the placebo.
        # Real UTC feeds commonly reopen on Sunday evening.
        candidate_count = int(np.ceil(n_bars * 7 / 5)) + 10
    dates = pd.date_range(
        "2018-01-01",
        periods=candidate_count,
        freq=config.data.frequency,
        tz=config.data.timezone,
    )
    if config.data.calendar == "fx_24_5":
        dates = dates[
            (dates.dayofweek < 5)
            | ((dates.dayofweek == 6) & (dates.hour >= 23))
        ][:n_bars]
        if len(dates) != n_bars:
            raise RuntimeError("Could not generate enough weekday FX bars")

    df = pd.DataFrame({
        "open": prices * (1 - noise),
        "high": prices * (1 + noise * 2),
        "low": prices * (1 - noise * 2),
        "close": prices,
        "volume": rng.integers(1000, 100000, size=n_bars).astype(float),
    }, index=dates)
    df.index.name = "datetime"

    log.info("synthetic_data_generated", n_bars=n_bars, mu=mu)
    return df
