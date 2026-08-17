"""
Tests para diferenciacion fraccional.

Verifica que la FFD pasa ADF con p < 0.05 para una serie
de precios tipica (random walk).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.fractional_diff import (
    _ffd_weights,
    adf_test,
    build_fold_feature_sets,
    fractional_diff,
    find_optimal_d,
)
from src.data.pit_splits import CVFold


class TestFFDWeights:
    """Tests para los pesos de diferenciacion fraccional."""

    def test_d_zero_gives_unit_weight(self):
        """d=0 -> no diferenciacion, peso = [1]."""
        w = _ffd_weights(0.0, threshold=1e-5)
        assert len(w) == 1
        assert abs(w[0] - 1.0) < 1e-10

    def test_d_one_gives_diff_weights(self):
        """d=1 -> diferenciacion entera, pesos = [-1, 1]."""
        w = _ffd_weights(1.0, threshold=1e-5)
        np.testing.assert_allclose(w, [-1.0, 1.0], atol=1e-5)

    def test_weights_decay(self):
        """Los pesos deben decaer en magnitud."""
        w = _ffd_weights(0.5, threshold=1e-8)
        magnitudes = np.abs(w)
        # Los ultimos pesos (indices mas bajos) deben ser mas pequeños
        assert magnitudes[-1] >= magnitudes[0]

    def test_positive_d_range(self):
        """Para 0 < d < 1, todos los pesos deben ser finitos."""
        for d in [0.1, 0.3, 0.5, 0.7, 0.9]:
            w = _ffd_weights(d, threshold=1e-5)
            assert all(np.isfinite(w))
            assert len(w) >= 2


class TestFractionalDiff:
    """Tests para la diferenciacion fraccional."""

    def test_output_shorter_than_input(self):
        """La salida es mas corta que la entrada por el ancho de la ventana."""
        rng = np.random.default_rng(42)
        series = pd.Series(np.cumsum(rng.normal(0, 1, 2000)))
        result = fractional_diff(series, d=0.5, weight_cutoff=1e-4)
        assert len(result) < len(series)
        assert len(result) > 0

    def test_d_one_equals_diff(self):
        """d=1 debe dar resultado similar a .diff()."""
        rng = np.random.default_rng(42)
        prices = pd.Series(np.cumsum(rng.normal(0, 1, 100)) + 100)
        ffd = fractional_diff(prices, d=1.0, weight_cutoff=1e-5)
        simple_diff = prices.diff().dropna()
        # Deben ser aproximadamente iguales
        common_idx = ffd.index.intersection(simple_diff.index)
        if len(common_idx) > 0:
            np.testing.assert_allclose(
                ffd.loc[common_idx].values,
                simple_diff.loc[common_idx].values,
                atol=1e-8,
            )


class TestADFTest:
    """Tests para el test ADF."""

    def test_stationary_series(self):
        """Una serie de ruido blanco debe ser estacionaria."""
        rng = np.random.default_rng(42)
        noise = pd.Series(rng.normal(0, 1, 1000))
        result = adf_test(noise)
        assert bool(result["stationary"]) is True
        assert result["pvalue"] < 0.05

    def test_random_walk_not_stationary(self):
        """Un random walk no debe ser estacionario."""
        rng = np.random.default_rng(42)
        rw = pd.Series(np.cumsum(rng.normal(0, 1, 1000)))
        result = adf_test(rw)
        assert bool(result["stationary"]) is False
        assert result["pvalue"] > 0.05

    def test_short_series(self):
        """Una serie muy corta devuelve no estacionaria."""
        short = pd.Series([1.0, 2.0, 3.0])
        result = adf_test(short)
        assert result["stationary"] is False


class TestFindOptimalD:
    """Tests para la busqueda de d optima."""

    def test_finds_d_for_random_walk(self):
        """Debe encontrar una d < 1 que haga estacionario un random walk."""
        rng = np.random.default_rng(42)
        prices = pd.Series(np.cumsum(rng.normal(0, 1, 2000)) + 100)

        result = find_optimal_d(
            prices,
            d_range=(0.1, 0.9),
            d_step=0.1,
            threshold_adf_pvalue=0.05,
        )

        assert result["optimal_d"] <= 1.0
        assert result["optimal_d"] > 0.0
        assert result["series_transformed"] is not None
        assert len(result["adf_results"]) > 0

    def test_adf_passes_at_optimal_d(self):
        """La serie transformada con d optima debe pasar ADF."""
        rng = np.random.default_rng(42)
        prices = pd.Series(np.cumsum(rng.normal(0, 1, 2000)) + 100)

        result = find_optimal_d(prices)
        transformed = result["series_transformed"]

        adf = adf_test(transformed)
        # La serie con d optima debe ser estacionaria
        assert adf["pvalue"] < 0.05, (
            f"FFD with d={result['optimal_d']} did not pass ADF: "
            f"p={adf['pvalue']:.4f}"
        )


def test_first_fold_never_selects_d_from_future_train():
    dates = pd.date_range("2024-01-01", periods=120, freq="15min", tz="UTC")
    prices = pd.Series(
        100.0 + np.cumsum(np.random.default_rng(7).normal(size=len(dates))),
        index=dates,
        name="close",
    )
    df = pd.DataFrame({"close": prices})
    target = prices.pct_change().dropna()
    fold = CVFold(
        fold_id=0,
        train_indices=np.arange(20, len(target)),
        test_indices=np.arange(0, 20),
        purge_indices=np.array([], dtype=int),
        embargo_indices=np.array([], dtype=int),
    )

    item = build_fold_feature_sets(df, target, [fold])[0]

    assert item.d_used is None
    assert item.d_selection_method == "not_evaluable_no_pre_test_history"
    assert item.train_features.empty
    assert item.test_features.empty
