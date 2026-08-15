"""Auditoría reproducible de la ronda de remediación.

Uso:
    python scratch/remediation_audit.py --seeds 30
    python scratch/remediation_audit.py --seeds 30 --stage2

No usa el ledger persistente: cada semilla trabaja con resultados en memoria.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import annualization_factor_from_frequency, load_config, set_global_seed
from src.data.ingest import compute_returns
from src.data.pit_splits import combinatorial_purged_cv
from src.features.fractional_diff import build_fold_feature_sets
from src.models.baselines import (
    buy_and_hold,
    lightgbm_classifier,
    ridge_regression,
    sma_crossover,
)
from src.models.deep_model import tcn_attention_ppo, tcn_supervised
from src.orchestrator.pipeline import _generate_synthetic_data
from src.orchestrator.pipeline import run_pipeline
from src.stats.effective_trials import estimate_pairwise_correlation
from src.stats.psr_dsr import deflated_sharpe_ratio


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a binomial rate."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("successes/trials must define a valid binomial sample")
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return center - half, center + half


def run(seeds: int, include_stage2: bool, fixed_d: float | None = None) -> None:
    logging.disable(logging.CRITICAL)
    rows = []
    deep_rows = []

    for seed in range(1, seeds + 1):
        cfg = load_config(PROJECT_ROOT / "config" / "pipeline_config.yaml")
        cfg.reproducibility.global_seed = seed
        set_global_seed(seed)
        annualization = annualization_factor_from_frequency(
            cfg.data.frequency,
            calendar=cfg.data.calendar,
        )

        df = _generate_synthetic_data(cfg, mu=0.0)
        prices = df["close"]
        returns = compute_returns(prices, method="log")
        target = returns.shift(-1).dropna()
        folds = combinatorial_purged_cv(
            len(target),
            n_folds=cfg.splits.n_folds,
            purge_window=cfg.splits.purge_window,
            embargo_window=cfg.splits.embargo_window,
        )
        fold_sets = build_fold_feature_sets(
            df,
            target,
            folds,
            fixed_d=fixed_d,
            d_range=tuple(cfg.features.fractional_diff.d_range),
            d_step=cfg.features.fractional_diff.d_step,
            threshold_adf_pvalue=cfg.features.fractional_diff.threshold_adf_pvalue,
            weight_cutoff=cfg.features.fractional_diff.weight_cutoff,
        )
        metadata = {
            "d_used_by_fold": [item.d_used for item in fold_sets],
            "ffd_lookback_by_fold": [item.ffd_lookback for item in fold_sets],
            "d_selection_method": (
                "config_fixed" if fixed_d is not None else "rule_based_adf"
            ),
            "frequency": cfg.data.frequency,
            "config_version": cfg.config_version,
        }
        features = next(
            (item.train_features for item in fold_sets if not item.train_features.empty),
            pd.DataFrame(),
        )

        stages = [
            buy_and_hold(returns, annualization=annualization),
            sma_crossover(
                prices,
                returns,
                cfg.models.baselines.sma_short,
                cfg.models.baselines.sma_long,
                annualization=annualization,
            ),
            ridge_regression(
                features,
                target,
                folds,
                alpha=cfg.models.baselines.ridge_alpha,
                annualization=annualization,
                fold_feature_sets=fold_sets,
                ledger_metadata=metadata,
            ),
            lightgbm_classifier(
                features,
                target,
                folds,
                lgbm_params=cfg.models.baselines.lgbm_params.model_dump(),
                annualization=annualization,
                fold_feature_sets=fold_sets,
                ledger_metadata=metadata,
            ),
        ]

        accumulated = []
        for trial, stage in enumerate(stages, 1):
            ret = np.asarray(stage["returns_oos"])
            accumulated.append(ret)
            min_len = min(len(x) for x in accumulated)
            corr = (
                estimate_pairwise_correlation(
                    np.vstack([x[:min_len] for x in accumulated])
                )
                if trial > 1 and min_len > 10
                else 0.0
            )
            dsr = deflated_sharpe_ratio(
                sr_hat=stage["sharpe_oos"],
                returns=ret,
                n_trials=trial,
                avg_pairwise_corr=corr,
                max_lag_autocorr=cfg.stats.psr_dsr.max_lag_autocorr,
                annualization_factor=annualization,
            )
            rows.append(
                {
                    "seed": seed,
                    "model": ["buy_hold", "sma_crossover", "ridge", "lightgbm"][trial - 1],
                    "dsr": dsr["dsr"],
                    "t_nominal": dsr["t_nominal"],
                    "t_effective": dsr["t_effective"],
                    "n_observations": len(ret),
                    "v_nominal": dsr["v_nominal"],
                    "skewness": dsr["skewness"],
                    "kurtosis": dsr["kurtosis"],
                    "d_used": metadata["d_used_by_fold"],
                    "ffd_lookback": metadata["ffd_lookback_by_fold"],
                }
            )

        if include_stage2:
            tcn = tcn_supervised(
                features,
                target,
                folds,
                annualization=annualization,
                fold_feature_sets=fold_sets,
            )
            attention = tcn_attention_ppo(
                features,
                target,
                folds,
                annualization=annualization,
                fold_feature_sets=fold_sets,
            )
            if len(tcn["returns_oos"]) and len(attention["returns_oos"]):
                n = min(len(tcn["returns_oos"]), len(attention["returns_oos"]))
                corr = float(
                    np.corrcoef(tcn["returns_oos"][:n], attention["returns_oos"][:n])[0, 1]
                )
            else:
                corr = float("nan")
            deep_rows.append(
                {
                    "seed": seed,
                    "tcn_sharpe": tcn["sharpe_oos"],
                    "attention_sharpe": attention["sharpe_oos"],
                    "return_correlation": corr,
                    "mean_abs_difference": float(
                        np.mean(
                            np.abs(
                                tcn["returns_oos"][:n]
                                - attention["returns_oos"][:n]
                            )
                        )
                    )
                    if n
                    else float("nan"),
                }
            )

    result = pd.DataFrame(rows)
    print("MODEL|N|DSR_MEAN|DSR_STD|SE_MEAN|T_NOMINAL_MEDIAN|T_EFFECTIVE_MEDIAN|N_OBS_MEDIAN|SKEW_MEAN|KURT_MEAN")
    for model, group in result.groupby("model", sort=False):
        print(
            f"{model}|{len(group)}|{group.dsr.mean():.6f}|"
            f"{group.dsr.std(ddof=1):.6f}|"
            f"{group.dsr.std(ddof=1) / np.sqrt(len(group)):.6f}|"
            f"{group['t_nominal'].median():.1f}|"
            f"{group['t_effective'].median():.1f}|"
            f"{group['n_observations'].median():.1f}|"
            f"{group['skewness'].mean():+.6f}|{group['kurtosis'].mean():+.6f}"
        )
    if deep_rows:
        deep = pd.DataFrame(deep_rows)
        print("STAGE2|N|CORR_MEAN|CORR_MIN|CORR_MAX|IDENTICAL_COUNT")
        print(
            f"stage2|{len(deep)}|{deep.return_correlation.mean():.6f}|"
            f"{deep.return_correlation.min():.6f}|"
            f"{deep.return_correlation.max():.6f}|"
            f"{int((deep.return_correlation >= 0.999999).sum())}"
            )


def run_gate(seeds: int, mu: float, details: bool = False) -> dict[str, float | int]:
    """Re-ejecuta el DAG completo con ledger aislado por semilla."""
    logging.disable(logging.CRITICAL)
    logging.getLogger().setLevel(logging.CRITICAL)
    decisions = []
    recommended = []

    with tempfile.TemporaryDirectory(prefix="dsr_gate_audit_") as temp_dir:
        for seed in range(1, seeds + 1):
            cfg = load_config(PROJECT_ROOT / "config" / "pipeline_config.yaml")
            cfg.reproducibility.global_seed = seed
            cfg.ledger.db_path = str(Path(temp_dir) / f"seed_{seed}.db")
            cfg.logging.level = "CRITICAL"
            result = run_pipeline(config=cfg, mu_override=mu)
            decisions.append(result.gate_decision)
            recommended.append(result.recommended_stage)
            if seed % 25 == 0:
                print(f"PROGRESS|seed={seed}|n={len(decisions)}", flush=True)
            if details:
                dsr_key = next(
                    (
                        key
                        for key, stage_data in result.stage_results.items()
                        if result.recommended_stage.endswith(
                            f"({stage_data.get('model_name')})"
                        )
                    ),
                    None,
                )
                dsr = result.dsr_results.get(dsr_key, {}).get("dsr")
                p_values = [
                    f"{key}:{value.get('p_value')}"
                    for key, value in result.ablation_results.items()
                    if isinstance(value, dict) and "p_value" in value
                ]
                print(
                    f"GATE_DETAIL|seed={seed}|decision={result.gate_decision}|"
                    f"recommended={result.recommended_stage}|dsr={dsr}|"
                    f"p_values={','.join(p_values)}"
                )

    counts = pd.Series(decisions).value_counts().to_dict()
    print("GATE|N|GO|NO_GO|UNIQUE_RECOMMENDATIONS")
    print(
        f"gate|{len(decisions)}|{counts.get('GO', 0)}|"
        f"{counts.get('NO-GO', 0)}|{len(set(recommended))}"
    )
    lo, hi = wilson_interval(counts.get("GO", 0), len(decisions))
    print(f"FPR|{counts.get('GO', 0) / len(decisions):.6f}|{lo:.6f}|{hi:.6f}")
    print("RECOMMENDED|" + "|".join(f"{k}:{v}" for k, v in sorted(pd.Series(recommended).value_counts().items())))
    return {
        "n": len(decisions),
        "go": counts.get("GO", 0),
        "no_go": counts.get("NO-GO", 0),
        "fpr": counts.get("GO", 0) / len(decisions),
        "ci_low": lo,
        "ci_high": hi,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--stage2", action="store_true")
    parser.add_argument("--fixed-d", type=float, default=None)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    if args.gate:
        run_gate(args.seeds, args.mu, args.details)
    else:
        run(args.seeds, args.stage2, args.fixed_d)
