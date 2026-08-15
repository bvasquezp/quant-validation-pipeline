"""
Punto de entrada para ejecutar el pipeline desde la linea de comandos.

Uso:
    python -m src [--config CONFIG_PATH] [--data DATA_PATH]
"""

from __future__ import annotations

import argparse
import json
import sys

from src.config import load_config
from src.orchestrator.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de Validacion Estadistica con DSR"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/pipeline_config.yaml",
        help="Ruta al archivo de configuracion YAML",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Ruta al CSV de datos (override del config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta para guardar el resultado JSON",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    result = run_pipeline(config=config, data_path=args.data)

    output_json = result.to_json()
    print(output_json)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"\nResultado guardado en: {args.output}", file=sys.stderr)

    # Exit code basado en la decision
    sys.exit(0 if result.gate_decision == "GO" else 1)


if __name__ == "__main__":
    main()
