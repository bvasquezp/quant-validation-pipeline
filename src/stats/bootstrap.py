"""
Agente 6 -- Bootstrap de diferencia de Sharpe entre modelos.

Usa BLOCK bootstrap (no iid) porque los retornos financieros no son
intercambiables independientemente -- hay autocorrelacion serial.

Criterio de aceptacion: la complejidad adicional solo se conserva si
p_value < 0.05 Y el DSR de la etapa mas compleja es mayor.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.logger import get_logger

log = get_logger(__name__)


def _sharpe_ratio(returns: np.ndarray, annualization: float = 1.0) -> float:
    """Sharpe ratio con factor de anualización explícito."""
    if len(returns) < 2:
        return 0.0
    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1)
    if std_r < 1e-15:
        return 0.0
    return float(mean_r / std_r * np.sqrt(annualization))


def _block_bootstrap_indices(
    n: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera indices para un block bootstrap circular.

    Los bloques se seleccionan con reemplazo y se concatenan
    hasta cubrir n observaciones.
    """
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=n_blocks)

    indices = []
    for start in starts:
        block = np.arange(start, start + block_size) % n
        indices.append(block)

    indices = np.concatenate(indices)[:n]
    return indices


def bootstrap_sharpe_diff(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    n_boot: int = 5000,
    block_size: int = 50,
    confidence_level: float = 0.95,
    seed: Optional[int] = None,
    annualization: float = 1.0,
) -> Dict[str, float]:
    """Compara Sharpe de b vs a usando block bootstrap.

    H0: SR(b) - SR(a) <= 0  (la complejidad adicional de b no ayuda).
    H1: SR(b) - SR(a) > 0   (b es genuinamente mejor).

    Args:
        returns_a: Retornos OOS del modelo mas simple (baseline).
        returns_b: Retornos OOS del modelo mas complejo.
        n_boot: Numero de replicas bootstrap.
        block_size: Tamaño del bloque para block bootstrap.
        confidence_level: Nivel de confianza para el intervalo.
        seed: Semilla para reproducibilidad.

    Returns:
        Dict con claves: observed_diff, diff_mean, ci_lower, ci_upper,
                         p_value, significant, sharpe_a, sharpe_b
    """
    returns_a = np.asarray(returns_a, dtype=np.float64)
    returns_b = np.asarray(returns_b, dtype=np.float64)

    rng = np.random.default_rng(seed)

    # Sharpe observado
    sr_a = _sharpe_ratio(returns_a, annualization)
    sr_b = _sharpe_ratio(returns_b, annualization)
    observed_diff = sr_b - sr_a

    # Las dos series deben tener la misma longitud para bootstrap pareado
    n = min(len(returns_a), len(returns_b))
    ret_a = returns_a[:n]
    ret_b = returns_b[:n]

    # Asegurar que block_size no excede n
    block_size = min(block_size, max(n // 3, 1))

    # Bootstrap
    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = _block_bootstrap_indices(n, block_size, rng)
        sr_a_boot = _sharpe_ratio(ret_a[idx], annualization)
        sr_b_boot = _sharpe_ratio(ret_b[idx], annualization)
        boot_diffs[i] = sr_b_boot - sr_a_boot

    # Estadisticas
    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))
    diff_mean = float(np.mean(boot_diffs))

    # p-valor one-sided bajo H0: centrar la distribución bootstrap en cero.
    centered_diffs = boot_diffs - diff_mean
    p_value = float((1 + np.sum(centered_diffs >= observed_diff)) / (n_boot + 1))
    significant = p_value < (1.0 - confidence_level)

    result = {
        "observed_diff": observed_diff,
        "diff_mean": diff_mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "significant": float(significant),
        "sharpe_a": sr_a,
        "sharpe_b": sr_b,
    }

    log.info(
        "bootstrap_sharpe_diff_computed",
        model_a_sharpe=round(sr_a, 4),
        model_b_sharpe=round(sr_b, 4),
        observed_diff=round(observed_diff, 4),
        p_value=round(p_value, 4),
        significant=significant,
    )

    return result


def ablation_ladder(
    returns_by_stage: Dict[str, np.ndarray],
    candidate_returns_by_stage: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    n_boot: int = 5000,
    block_size: int = 50,
    significance_level: float = 0.05,
    seed: Optional[int] = None,
    annualization: float = 1.0,
) -> Dict[str, Dict]:
    """Ejecuta la secuencia de ablacion completa.

    Compara pares consecutivos:
        stage_0 vs stage_1, stage_1 vs stage_2, ...

    y determina en que punto la complejidad adicional deja de
    justificarse estadisticamente.

    Args:
        returns_by_stage: Dict ordenado {nombre_etapa: retornos_oos}.
            Ej: {"buy_hold": [...], "sma": [...], "ridge": [...], "lgbm": [...]}
        n_boot: Replicas bootstrap.
        block_size: Tamaño de bloque.
        significance_level: Umbral de significancia.
        seed: Semilla.

    Returns:
        Dict con resultados por par + resumen ("recommended_stage").
    """
    stage_names = list(returns_by_stage.keys())
    if len(stage_names) < 2:
        return {"recommended_stage": stage_names[0] if stage_names else "none"}

    if candidate_returns_by_stage is None:
        candidate_returns_by_stage = {
            name: {name: np.asarray(returns_by_stage[name], dtype=np.float64)}
            for name in stage_names
        }

    rng = np.random.default_rng(seed)
    results = {}
    recommended_stage = stage_names[0]

    for i in range(len(stage_names) - 1):
        name_a = stage_names[i]
        name_b = stage_names[i + 1]
        comparison_key = f"{name_a}_vs_{name_b}"

        # Semilla derivada para reproducibilidad por comparacion
        comp_seed = rng.integers(0, 2**31)

        result = selection_aware_sharpe_diff(
            candidate_returns_by_stage[name_a],
            candidate_returns_by_stage[name_b],
            n_boot=n_boot,
            block_size=block_size,
            confidence_level=1.0 - significance_level,
            seed=comp_seed,
            annualization=annualization,
        )

        # Solo avanza si la mejora es significativa
        if result["p_value"] < significance_level and result["observed_diff"] > 0:
            recommended_stage = name_b
            result["decision"] = "ADVANCE"
        else:
            result["decision"] = "STOP"

        results[comparison_key] = result

        if result["decision"] == "STOP":
            log.info(
                "ablation_stopped",
                stopped_at=name_a,
                next_stage_rejected=name_b,
                p_value=result["p_value"],
            )
            # Marcar el resto como no evaluados
            for j in range(i + 2, len(stage_names)):
                skip_key = f"{stage_names[j-1]}_vs_{stage_names[j]}"
                results[skip_key] = {"decision": "SKIPPED"}
            break

    results["recommended_stage"] = recommended_stage
    return results


def selection_aware_sharpe_diff(
    candidates_a: Dict[str, Any],
    candidates_b: Dict[str, Any],
    n_boot: int = 5000,
    block_size: int = 50,
    confidence_level: float = 0.95,
    seed: Optional[int] = None,
    annualization: float = 1.0,
) -> Dict[str, float | str]:
    """Compara campeones después de repetir la selección bajo H0.

    Cada candidato se centra a media cero, se remuestrean bloques conjuntos
    y se vuelve a seleccionar el máximo de Sharpe dentro de cada etapa.
    """
    if not candidates_a or not candidates_b:
        raise ValueError("Both stages need at least one candidate")

    aligned = _align_candidates(candidates_a, candidates_b)
    candidates_a = aligned[0]
    candidates_b = aligned[1]
    all_candidates = list(candidates_a.items()) + list(candidates_b.items())
    n = min(len(values) for _, values in all_candidates)
    if n < max(3, block_size):
        block_size = max(1, min(block_size, n // 3))
    if n < 3:
        raise ValueError("At least three aligned observations are required")

    matrix_a = np.column_stack([np.asarray(values[:n], dtype=float) for values in candidates_a.values()])
    matrix_b = np.column_stack([np.asarray(values[:n], dtype=float) for values in candidates_b.values()])
    sr_a = np.array([_sharpe_ratio(matrix_a[:, i], annualization) for i in range(matrix_a.shape[1])])
    sr_b = np.array([_sharpe_ratio(matrix_b[:, i], annualization) for i in range(matrix_b.shape[1])])
    champ_a = int(np.argmax(sr_a))
    champ_b = int(np.argmax(sr_b))
    observed_diff = float(sr_b[champ_b] - sr_a[champ_a])

    null_matrix = np.column_stack(
        [matrix_a - matrix_a.mean(axis=0), matrix_b - matrix_b.mean(axis=0)]
    )
    rng = np.random.default_rng(seed)
    null_diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = _block_bootstrap_indices(n, block_size, rng)
        boot = null_matrix[idx]
        boot_a = boot[:, : matrix_a.shape[1]]
        boot_b = boot[:, matrix_a.shape[1] :]
        boot_sr_a = max(_sharpe_ratio(boot_a[:, j], annualization) for j in range(boot_a.shape[1]))
        boot_sr_b = max(_sharpe_ratio(boot_b[:, j], annualization) for j in range(boot_b.shape[1]))
        null_diffs[i] = boot_sr_b - boot_sr_a

    p_value = float((1 + np.sum(null_diffs >= observed_diff)) / (n_boot + 1))
    alpha = 1.0 - confidence_level
    return {
        "observed_diff": observed_diff,
        "diff_mean_null": float(null_diffs.mean()),
        "null_p95": float(np.percentile(null_diffs, 95)),
        "ci_lower": float(np.percentile(null_diffs, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(null_diffs, 100 * (1 - alpha / 2))),
        "p_value": p_value,
        "significant": float(p_value < (1.0 - confidence_level)),
        "sharpe_a": float(sr_a[champ_a]),
        "sharpe_b": float(sr_b[champ_b]),
        "champion_a": list(candidates_a)[champ_a],
        "champion_b": list(candidates_b)[champ_b],
        "selection_aware": True,
    }


def _align_candidates(*groups: Dict[str, Any]) -> tuple[Dict[str, np.ndarray], ...]:
    """Alinea candidatos por timestamp cuando el modelo los proporciona."""
    parsed = []
    timestamp_keys = []
    for group in groups:
        parsed_group = {}
        for name, value in group.items():
            if isinstance(value, dict):
                returns = np.asarray(value["returns"], dtype=np.float64)
                timestamps = value.get("timestamps")
            else:
                returns = np.asarray(value, dtype=np.float64)
                timestamps = None
            parsed_group[name] = (returns, timestamps)
            if timestamps is not None:
                timestamp_keys.append([str(item) for item in timestamps])
        parsed.append(parsed_group)

    if not timestamp_keys:
        return tuple(
            {
                name: values[0][: min(len(item[0]) for item in group.values())]
                for name, values in group.items()
            }
            for group in parsed
        )

    common = list(timestamp_keys[0])
    for keys in timestamp_keys[1:]:
        key_set = set(keys)
        common = [key for key in common if key in key_set]
    if len(common) < 3:
        raise ValueError("Candidate returns have fewer than three aligned timestamps")

    aligned_groups = []
    for group in parsed:
        aligned_group = {}
        for name, (returns, timestamps) in group.items():
            if timestamps is None:
                raise ValueError("Either all candidates provide timestamps or none do")
            positions = {str(timestamp): i for i, timestamp in enumerate(timestamps)}
            aligned_group[name] = np.asarray([returns[positions[key]] for key in common])
        aligned_groups.append(aligned_group)
    return tuple(aligned_groups)
