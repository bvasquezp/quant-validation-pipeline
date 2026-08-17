"""
Agente 1 -- Purged/embargoed temporal k-fold validation.

El purging y embargo son obligatorios: sin ellos, la informacion del
futuro se filtra al conjunto de entrenamiento a traves de la
autocorrelacion serial de los retornos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.logger import get_logger

log = get_logger(__name__)


@dataclass
class CVFold:
    """Un fold de cross-validation con indices purged y embargoed."""
    fold_id: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    purge_indices: np.ndarray
    embargo_indices: np.ndarray


def purged_train_test_split(
    n_samples: int,
    train_range: Tuple[int, int],
    test_range: Tuple[int, int],
    purge_window: int = 10,
    embargo_window: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Genera indices de train excluyendo la zona de purge y embargo
    alrededor del test set.

    Purge: elimina las observaciones de train que estan dentro de
    `purge_window` barras ANTES del inicio del test set.

    Embargo: elimina las observaciones de train que estan dentro de
    `embargo_window` barras DESPUES del final del test set.

    Esto previene la filtracion de informacion del futuro.

    Args:
        n_samples: Total de observaciones.
        train_range: (start, end) indices del rango de train candidato.
        test_range: (start, end) indices del rango de test.
        purge_window: Barras de purging antes del test.
        embargo_window: Barras de embargo despues del test.

    Returns:
        (train_indices, test_indices) con purge/embargo aplicados.
    """
    test_start, test_end = test_range

    # Zona de purge: barras antes del test
    purge_start = max(0, test_start - purge_window)
    purge_end = test_start

    # Zona de embargo: barras despues del test
    embargo_start = test_end + 1
    embargo_end = min(n_samples, test_end + 1 + embargo_window)

    # Indices excluidos
    excluded = set(range(purge_start, purge_end))
    excluded.update(range(embargo_start, embargo_end))
    excluded.update(range(test_start, test_end + 1))

    # Train: todo el rango candidato menos las exclusiones
    all_train = set(range(train_range[0], train_range[1] + 1))
    train_indices = np.array(sorted(all_train - excluded))
    test_indices = np.arange(test_start, test_end + 1)

    return train_indices, test_indices


def purged_kfold_cv(
    n_samples: int,
    n_folds: int = 5,
    purge_window: int = 10,
    embargo_window: int = 5,
) -> List[CVFold]:
    """Genera folds temporales purged/embargoed.

    Divide los datos en n_folds segmentos contiguos. Para cada fold,
    un segmento es test y el resto es train (con purge/embargo aplicados
    en los bordes entre train y test).

    Esto no construye las combinaciones de un CPCV completo; es un k-fold
    temporal con proteccion contra filtracion.

    Args:
        n_samples: Total de observaciones.
        n_folds: Numero de folds.
        purge_window: Barras de purging.
        embargo_window: Barras de embargo.

    Returns:
        Lista de CVFold con indices de train, test, purge y embargo.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if n_samples < n_folds * (purge_window + embargo_window + 2):
        raise ValueError(
            f"Not enough samples ({n_samples}) for {n_folds} folds "
            f"with purge={purge_window}, embargo={embargo_window}"
        )

    # Dividir en segmentos contiguos
    fold_boundaries = np.linspace(0, n_samples, n_folds + 1, dtype=int)
    folds = []

    for fold_id in range(n_folds):
        test_start = fold_boundaries[fold_id]
        test_end = fold_boundaries[fold_id + 1] - 1

        # El rango de train candidato es todo lo que no es test
        # Aplicamos purge/embargo para cada lado del test
        all_indices = set(range(n_samples))
        test_set = set(range(test_start, test_end + 1))

        # Purge antes del test
        purge_start = max(0, test_start - purge_window)
        purge_set = set(range(purge_start, test_start))

        # Embargo despues del test
        embargo_end = min(n_samples, test_end + 1 + embargo_window)
        embargo_set = set(range(test_end + 1, embargo_end))

        train_set = all_indices - test_set - purge_set - embargo_set
        train_indices = np.array(sorted(train_set))
        test_indices = np.arange(test_start, test_end + 1)
        purge_indices = np.array(sorted(purge_set))
        embargo_indices = np.array(sorted(embargo_set))

        fold = CVFold(
            fold_id=fold_id,
            train_indices=train_indices,
            test_indices=test_indices,
            purge_indices=purge_indices,
            embargo_indices=embargo_indices,
        )
        folds.append(fold)

        log.info(
            "purged_kfold_created",
            fold_id=fold_id,
            train_size=len(train_indices),
            test_size=len(test_indices),
            purge_size=len(purge_indices),
            embargo_size=len(embargo_indices),
        )

    return folds


def combinatorial_purged_cv(*args, **kwargs) -> List[CVFold]:
    """Compatibilidad: el nombre historico era incorrecto; usar purged_kfold_cv."""
    return purged_kfold_cv(*args, **kwargs)


def verify_no_overlap(folds: List[CVFold]) -> bool:
    """Verifica que no haya overlap entre train y test+purge+embargo
    en ningun fold.

    Esta verificacion es un invariante critico: si falla, hay filtracion
    de informacion del futuro y todos los resultados son invalidos.
    """
    all_ok = True
    for fold in folds:
        train_set = set(fold.train_indices)
        test_set = set(fold.test_indices)
        purge_set = set(fold.purge_indices)
        embargo_set = set(fold.embargo_indices)

        # Train no debe solapar con test
        overlap_test = train_set & test_set
        if overlap_test:
            log.error(
                "overlap_train_test",
                fold_id=fold.fold_id,
                overlap_size=len(overlap_test),
            )
            all_ok = False

        # Train no debe solapar con purge
        overlap_purge = train_set & purge_set
        if overlap_purge:
            log.error(
                "overlap_train_purge",
                fold_id=fold.fold_id,
                overlap_size=len(overlap_purge),
            )
            all_ok = False

        # Train no debe solapar con embargo
        overlap_embargo = train_set & embargo_set
        if overlap_embargo:
            log.error(
                "overlap_train_embargo",
                fold_id=fold.fold_id,
                overlap_size=len(overlap_embargo),
            )
            all_ok = False

    if all_ok:
        log.info("no_overlap_verified", n_folds=len(folds))
    return all_ok
