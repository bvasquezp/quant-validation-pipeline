"""
Tests para el nucleo estadistico: PSR/DSR.

El test critico: test_dsr_at_null_is_half.
Si SR_hat == SR* (el Sharpe iguala exactamente lo esperado por azar
bajo V trials), DSR debe converger a ~0.5.

Sin este test, no hay forma de saber si la implementacion del paper
esta bien traducida a codigo.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sp_stats

from src.stats.psr_dsr import (
    deflated_sharpe_ratio,
    effective_sample_size,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_variance,
)
from src.stats.effective_trials import effective_trials, estimate_pairwise_correlation


# ---------------------------------------------------------------------------
# Tests de effective_sample_size (Newey-West)
# ---------------------------------------------------------------------------

class TestEffectiveSampleSize:
    """Tests para la correccion de T por autocorrelacion."""

    def test_iid_returns_t_approx_nominal(self):
        """Con retornos iid, T efectivo debe ser cercano a T nominal."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, size=1000)
        t_eff = effective_sample_size(returns, max_lag=5)
        # Deberia ser cercano a 1000, con tolerancia amplia por ruido
        assert 800 < t_eff <= 1000

    def test_autocorrelated_returns_reduce_t(self):
        """Con retornos autocorrelacionados, T efectivo debe ser < T nominal."""
        rng = np.random.default_rng(42)
        n = 1000
        innovations = rng.normal(0, 0.02, size=n)
        # AR(1) con rho = 0.5
        returns = np.zeros(n)
        returns[0] = innovations[0]
        for i in range(1, n):
            returns[i] = 0.5 * returns[i-1] + innovations[i]

        t_eff = effective_sample_size(returns, max_lag=5)
        assert t_eff < n * 0.8  # al menos 20% de reduccion

    def test_few_observations(self):
        """Con pocas observaciones, devuelve T nominal."""
        returns = np.array([0.01, -0.01, 0.02])
        t_eff = effective_sample_size(returns, max_lag=5)
        assert t_eff == 3.0

    def test_constant_returns(self):
        """Retornos constantes (varianza cero) devuelve T nominal."""
        returns = np.ones(100) * 0.001
        t_eff = effective_sample_size(returns, max_lag=5)
        assert t_eff == 100.0


# ---------------------------------------------------------------------------
# Tests de sharpe_variance
# ---------------------------------------------------------------------------

class TestSharpeVariance:
    """Tests para la varianza de Lo (2002)."""

    def test_gaussian_returns(self):
        """Con retornos gaussianos (skew=0, kurt=0), Var(SR) = 1/(T-1)."""
        var = sharpe_variance(sr_hat=0.0, T=100.0, skewness=0.0, kurtosis=0.0)
        assert abs(var - 1.0 / 99.0) < 1e-10

    def test_positive_skew_reduces_variance(self):
        """Skew positivo con SR positivo reduce la varianza."""
        var_no_skew = sharpe_variance(sr_hat=1.0, T=100, skewness=0.0, kurtosis=0.0)
        var_pos_skew = sharpe_variance(sr_hat=1.0, T=100, skewness=0.5, kurtosis=0.0)
        assert var_pos_skew < var_no_skew

    def test_fat_tails_increase_variance(self):
        """Exceso de kurtosis (colas pesadas) aumenta la varianza."""
        var_normal = sharpe_variance(sr_hat=1.0, T=100, skewness=0.0, kurtosis=0.0)
        var_fat = sharpe_variance(sr_hat=1.0, T=100, skewness=0.0, kurtosis=6.0)
        assert var_fat > var_normal


# ---------------------------------------------------------------------------
# Tests de expected_max_sharpe (SR*)
# ---------------------------------------------------------------------------

class TestExpectedMaxSharpe:
    """Tests para SR* bajo la nula."""

    def test_single_trial_sr_star_is_zero(self):
        """Con V=1, SR* = 0 (no hay sesgo de seleccion)."""
        sr_star = expected_max_sharpe(V=1, sigma_sr=0.1)
        assert sr_star == 0.0

    def test_more_trials_higher_sr_star(self):
        """Mas trials -> SR* mas alto (mas probable encontrar uno bueno por azar)."""
        sr_star_10 = expected_max_sharpe(V=10, sigma_sr=0.1)
        sr_star_100 = expected_max_sharpe(V=100, sigma_sr=0.1)
        assert sr_star_100 > sr_star_10 > 0

    def test_sr_star_scales_with_sigma(self):
        """SR* escala linealmente con sigma_SR."""
        sr_star_1 = expected_max_sharpe(V=50, sigma_sr=0.1)
        sr_star_2 = expected_max_sharpe(V=50, sigma_sr=0.2)
        assert abs(sr_star_2 / sr_star_1 - 2.0) < 0.01


# ---------------------------------------------------------------------------
# Tests de PSR
# ---------------------------------------------------------------------------

class TestPSR:
    """Tests para el Probabilistic Sharpe Ratio."""

    def test_psr_zero_benchmark_positive_sharpe(self):
        """PSR > 0.5 si SR_hat > benchmark con datos suficientes."""
        psr = probabilistic_sharpe_ratio(
            sr_hat=0.5, sr_benchmark=0.0, T=500, skewness=0.0, kurtosis=0.0,
        )
        assert psr > 0.5

    def test_psr_at_benchmark_is_half(self):
        """PSR = 0.5 cuando SR_hat == benchmark."""
        psr = probabilistic_sharpe_ratio(
            sr_hat=1.0, sr_benchmark=1.0, T=500, skewness=0.0, kurtosis=0.0,
        )
        assert abs(psr - 0.5) < 1e-10

    def test_psr_negative_sharpe_below_half(self):
        """PSR < 0.5 si SR_hat < benchmark."""
        psr = probabilistic_sharpe_ratio(
            sr_hat=-0.5, sr_benchmark=0.0, T=500, skewness=0.0, kurtosis=0.0,
        )
        assert psr < 0.5


# ---------------------------------------------------------------------------
# TEST CRITICO: DSR en el caso nulo
# ---------------------------------------------------------------------------

class TestDSR:
    """Tests para el Deflated Sharpe Ratio."""

    def test_dsr_reference_external(self):
        """Caso de referencia calculado de forma INDEPENDIENTE
        con T, V, rho, skew, kurtosis fijos.
        """
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, size=1000)
        result = deflated_sharpe_ratio(
            sr_hat=0.056133,
            returns=returns,
            n_trials=10,
            avg_pairwise_corr=0.3,
            max_lag_autocorr=0,
            annualized_input=False,
        )
        assert abs(result["v_effective"] - 2.702703) < 1e-4
        assert abs(result["sr_star"] - 0.024490) < 1e-3
        assert abs(result["dsr"] - 0.841191) < 1e-3

    def test_dsr_decreases_with_more_trials(self):
        """Mas trials (V mayor) -> DSR menor (mas penalizacion)."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, size=1000)
        sr_hat = np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252)

        dsr_v5 = deflated_sharpe_ratio(
            sr_hat=sr_hat, returns=returns, n_trials=5,
            annualization_factor=252.0,
        )
        dsr_v100 = deflated_sharpe_ratio(
            sr_hat=sr_hat, returns=returns, n_trials=100,
            annualization_factor=252.0,
        )

        assert dsr_v100["dsr"] < dsr_v5["dsr"]

    def test_dsr_higher_with_correlated_trials(self):
        """Trials correlacionados reducen V efectivo, aumentando DSR
        (menos penalizacion porque no son independientes)."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, size=1000)
        sr_hat = np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252)

        dsr_uncorr = deflated_sharpe_ratio(
            sr_hat=sr_hat, returns=returns, n_trials=50,
            avg_pairwise_corr=0.0,
            annualization_factor=252.0,
        )
        dsr_corr = deflated_sharpe_ratio(
            sr_hat=sr_hat, returns=returns, n_trials=50,
            avg_pairwise_corr=0.5,
            annualization_factor=252.0,
        )

        assert dsr_corr["dsr"] > dsr_uncorr["dsr"]

    def test_dsr_output_keys(self):
        """Verifica que el output tiene todas las claves esperadas."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, size=500)
        sr_hat = 1.0

        result = deflated_sharpe_ratio(
            sr_hat=sr_hat, returns=returns, n_trials=10,
            annualization_factor=252.0,
        )

        expected_keys = {
            "psr", "dsr", "sr_star", "sr_hat",
            "t_nominal", "t_effective", "v_nominal", "v_effective",
            "skewness", "kurtosis", "sigma_sr",
        }
        assert set(result.keys()) == expected_keys

    def test_annualized_and_periodic_inputs_are_equivalent(self):
        """La rama anualizada debe ser solo una conversión de unidades."""
        rng = np.random.default_rng(123)
        returns = rng.normal(0.0, 0.02, size=1000)
        annualization = 35040.0  # 15-minute bars, 24/7 calendar
        sr_period = 0.04

        annualized = deflated_sharpe_ratio(
            sr_hat=sr_period * np.sqrt(annualization),
            returns=returns,
            n_trials=3,
            annualized_input=True,
            annualization_factor=annualization,
        )
        periodic = deflated_sharpe_ratio(
            sr_hat=sr_period,
            returns=returns,
            n_trials=3,
            annualized_input=False,
        )

        for key in ("psr", "dsr", "t_effective", "v_effective"):
            assert abs(annualized[key] - periodic[key]) < 1e-12
        assert abs(annualized["sr_star"] / np.sqrt(annualization) - periodic["sr_star"]) < 1e-12


# ---------------------------------------------------------------------------
# Tests de effective_trials
# ---------------------------------------------------------------------------

class TestEffectiveTrials:
    """Tests para la correccion de V."""

    def test_independent_trials(self):
        """Con rho=0, V_eff = V_nominal."""
        v_eff = effective_trials(50, 0.0)
        assert abs(v_eff - 50.0) < 1e-10

    def test_perfectly_correlated(self):
        """Con rho=1, V_eff = 1."""
        v_eff = effective_trials(50, 1.0)
        assert abs(v_eff - 1.0) < 1e-10

    def test_partial_correlation(self):
        """Con rho=0.5, V_eff < V_nominal pero > 1."""
        v_eff = effective_trials(50, 0.5)
        assert 1.0 < v_eff < 50.0

    def test_single_trial(self):
        """Un solo trial siempre da V_eff = 1."""
        assert effective_trials(1, 0.0) == 1.0
        assert effective_trials(1, 0.5) == 1.0

    def test_negative_correlation_clamped(self):
        """Correlacion negativa se clampea a 0."""
        v_eff = effective_trials(50, -0.3)
        assert abs(v_eff - 50.0) < 1e-10


class TestEstimatePairwiseCorrelation:
    """Tests para la estimacion de correlacion entre trials."""

    def test_identical_returns(self):
        """Series identicas deben tener correlacion ~1."""
        returns = np.random.default_rng(42).normal(0, 0.02, size=100)
        matrix = np.vstack([returns, returns, returns])
        corr = estimate_pairwise_correlation(matrix)
        assert abs(corr - 1.0) < 1e-10

    def test_independent_returns(self):
        """Series independientes deben tener correlacion ~0."""
        rng = np.random.default_rng(42)
        matrix = rng.normal(0, 0.02, size=(10, 1000))
        corr = estimate_pairwise_correlation(matrix)
        assert abs(corr) < 0.15  # tolerancia amplia por muestreo

    def test_single_trial_returns_zero(self):
        """Con una sola serie, devuelve 0."""
        matrix = np.random.default_rng(42).normal(0, 0.02, size=(1, 100))
        corr = estimate_pairwise_correlation(matrix)
        assert corr == 0.0
