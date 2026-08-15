import numpy as np

from src.stats.bootstrap import selection_aware_sharpe_diff


def test_selection_aware_bootstrap_reselects_each_stage():
    rng = np.random.default_rng(7)
    stage_a = {"a0": rng.normal(0, 0.01, 120), "a1": rng.normal(0, 0.01, 120)}
    stage_b = {"b0": rng.normal(0, 0.01, 120), "b1": rng.normal(0, 0.01, 120)}

    result = selection_aware_sharpe_diff(
        stage_a,
        stage_b,
        n_boot=100,
        block_size=20,
        annualization=1.0,
        seed=11,
    )

    assert result["selection_aware"] is True
    assert result["champion_a"] in stage_a
    assert result["champion_b"] in stage_b
    assert 0.0 < result["p_value"] <= 1.0
