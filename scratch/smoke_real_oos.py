"""Small real-data smoke for the reserved-OOS path."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data.ingest import load_csv
from src.orchestrator.pipeline import run_pipeline


def main() -> None:
    config = load_config("config/pipeline_config.yaml")
    source = load_csv(
        config.data.path,
        frequency=config.data.frequency,
        calendar=config.data.calendar,
        timezone=config.data.timezone,
        symbol=config.data.symbol,
        allowed_gap_dates=config.data.allowed_gap_dates,
    ).iloc[:16000]
    split = int(len(source) * 0.75)

    with tempfile.TemporaryDirectory(prefix="real_oos_smoke_") as directory:
        directory = Path(directory)
        csv_path = directory / "EURUSD_15min_subset.csv"
        source.to_csv(csv_path, index=True)

        config.data.train_start = str(source.index[0])
        config.data.train_end = str(source.index[split - 1])
        config.data.test_start = str(source.index[split])
        config.data.test_end = str(source.index[-1])
        config.features.fractional_diff.d_range = (0.3, 0.3)
        config.stats.bootstrap.n_boot = 100
        config.ledger.db_path = str(directory / "ledger.db")
        config.data.manifest_path = str(directory / "manifest.json")
        config.logging.level = "CRITICAL"

        result = run_pipeline(config=config, data_path=str(csv_path))
        print(json.dumps({
            "data_mode": result.data_mode,
            "final_oos_status": result.final_oos_status,
            "final_oos_models": sorted(result.final_oos_results),
            "recommended_stage": result.recommended_stage,
            "gate_decision": result.gate_decision,
            "fold_diagnostics": result.fold_diagnostics,
        }, indent=2, default=str))


if __name__ == "__main__":
    main()
