"""
Tests para purging/embargo y folds temporales.

El invariante critico: no debe haber overlap entre train y
test/purge/embargo en ningun fold.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.pit_splits import (
    CVFold,
    purged_kfold_cv,
    purged_train_test_split,
    verify_no_overlap,
)


class TestPurgedTrainTestSplit:
    """Tests para el split con purge/embargo."""

    def test_basic_split(self):
        """Verifica que el split basico funciona."""
        train_idx, test_idx = purged_train_test_split(
            n_samples=100,
            train_range=(0, 99),
            test_range=(40, 59),
            purge_window=5,
            embargo_window=3,
        )
        # Test indices deben ser 40-59
        np.testing.assert_array_equal(test_idx, np.arange(40, 60))

        # Train no debe contener indices del test
        assert len(set(train_idx) & set(test_idx)) == 0

        # Train no debe contener indices de la zona de purge (35-39)
        purge_zone = set(range(35, 40))
        assert len(set(train_idx) & purge_zone) == 0

        # Train no debe contener indices de la zona de embargo (60-62)
        embargo_zone = set(range(60, 63))
        assert len(set(train_idx) & embargo_zone) == 0

    def test_no_overlap_at_boundaries(self):
        """Verifica que no hay overlap en los bordes."""
        train_idx, test_idx = purged_train_test_split(
            n_samples=50,
            train_range=(0, 49),
            test_range=(0, 9),
            purge_window=5,
            embargo_window=5,
        )
        # No overlap
        assert len(set(train_idx) & set(test_idx)) == 0

    def test_large_purge_embargo(self):
        """Con ventanas grandes, el train se reduce significativamente."""
        train_idx, test_idx = purged_train_test_split(
            n_samples=100,
            train_range=(0, 99),
            test_range=(40, 59),
            purge_window=20,
            embargo_window=10,
        )
        # Deben haberse excluido mas indices
        assert len(train_idx) <= 100 - 20 - 20 - 10


class TestPurgedKFold:
    """Tests para purged/embargoed temporal k-fold."""

    def test_basic_purged_kfold(self):
        """Verifica el numero configurado de folds temporales."""
        folds = purged_kfold_cv(
            n_samples=1000, n_folds=5, purge_window=10, embargo_window=5,
        )
        assert len(folds) == 5

    def test_no_overlap_any_fold(self):
        """INVARIANTE CRITICO: no hay overlap en ningun fold."""
        folds = purged_kfold_cv(
            n_samples=1000, n_folds=5, purge_window=10, embargo_window=5,
        )
        assert verify_no_overlap(folds) is True

    def test_no_overlap_extreme_windows(self):
        """Incluso con ventanas grandes, no hay overlap."""
        folds = purged_kfold_cv(
            n_samples=5000, n_folds=5, purge_window=50, embargo_window=30,
        )
        assert verify_no_overlap(folds) is True

    def test_all_samples_accounted(self):
        """Cada indice del dataset debe aparecer en train o test de algun fold."""
        n = 500
        folds = purged_kfold_cv(
            n_samples=n, n_folds=5, purge_window=5, embargo_window=3,
        )
        # Todos los indices deben aparecer como test en exactamente un fold
        all_test = np.concatenate([f.test_indices for f in folds])
        assert len(set(all_test)) == n

    def test_train_sizes_reasonable(self):
        """El tamaño de train debe ser menor que n_samples pero razonable."""
        folds = purged_kfold_cv(
            n_samples=1000, n_folds=5, purge_window=10, embargo_window=5,
        )
        for fold in folds:
            # Al menos la mitad de los datos en train
            assert len(fold.train_indices) > 400
            # Menos que todo el dataset
            assert len(fold.train_indices) < 1000

    def test_insufficient_samples_raises(self):
        """Debe lanzar error si no hay suficientes muestras."""
        with pytest.raises(ValueError, match="Not enough samples"):
            purged_kfold_cv(
                n_samples=20, n_folds=5, purge_window=10, embargo_window=5,
            )

    def test_fold_ids_sequential(self):
        """Los IDs de fold deben ser secuenciales."""
        folds = purged_kfold_cv(n_samples=1000, n_folds=5)
        for i, fold in enumerate(folds):
            assert fold.fold_id == i
