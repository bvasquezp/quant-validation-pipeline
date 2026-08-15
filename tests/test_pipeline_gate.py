import numpy as np

from src.config import load_config
from src.orchestrator.pipeline import PipelineResult, _gate_decision


def test_gate_resolves_human_stage_label_to_model_dsr():
    result = PipelineResult()
    result.recommended_stage = "Stage 0 (buy_and_hold)"
    result.stage_results["buy_hold"] = {"model_name": "buy_and_hold"}
    result.dsr_results["buy_hold"] = {"dsr": np.float64(0.99)}

    config = load_config("config/pipeline_config.yaml")
    decision = _gate_decision(result, config)

    assert decision["decision"] == "GO"
