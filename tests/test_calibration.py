import os

import pytest

from scratch.remediation_audit import run_gate, wilson_interval


def test_wilson_interval_for_reported_null_rate():
    low, high = wilson_interval(20, 300)
    assert low == pytest.approx(0.043567, abs=1e-6)
    assert high == pytest.approx(0.100724, abs=1e-6)


@pytest.mark.calibration
@pytest.mark.skipif(
    os.getenv("RUN_LONG_CALIBRATION") != "1",
    reason="set RUN_LONG_CALIBRATION=1 for the 300-seed gate canary",
)
def test_gate_null_calibration_300():
    result = run_gate(300, 0.0)
    assert result["n"] == 300
    assert result["ci_low"] <= 0.05 <= result["ci_high"]


@pytest.mark.calibration
@pytest.mark.skipif(
    os.getenv("RUN_LONG_CALIBRATION") != "1",
    reason="set RUN_LONG_CALIBRATION=1 for the positive control canary",
)
def test_gate_positive_control_30():
    result = run_gate(30, 5.0)
    assert result["n"] == 30
    assert result["go"] >= 27
