import numpy as np

from src.config import load_config
from src.orchestrator.pipeline import PipelineResult, _gate_decision


def test_gate_resolves_human_stage_label_to_model_dsr():
    result = PipelineResult()
    result.recommended_stage = "Stage 0 (buy_and_hold)"
    result.data_mode = "synthetic"
    result.final_oos_status = "not_applicable_synthetic"
    result.stage_results["buy_hold"] = {"model_name": "buy_and_hold"}
    result.dsr_results["buy_hold"] = {"dsr": np.float64(0.99)}

    config = load_config("config/pipeline_config.yaml")
    decision = _gate_decision(result, config)

    assert decision["decision"] == "GO"


def test_real_gate_requires_final_oos_evidence():
    result = PipelineResult()
    result.recommended_stage = "Stage 0 (buy_and_hold)"
    result.data_mode = "real"
    result.final_oos_status = "evaluated"
    result.stage_results["buy_hold"] = {"model_name": "buy_and_hold"}
    result.dsr_results["buy_hold"] = {"dsr": 0.99}

    config = load_config("config/pipeline_config.yaml")
    without_oos = _gate_decision(result, config)
    assert without_oos["decision"] == "NO-GO"

    result.final_oos_results["buy_hold"] = {
        "dsr": 0.99,
        "n_observations": 100,
    }
    with_oos = _gate_decision(result, config)
    assert with_oos["decision"] == "GO"
