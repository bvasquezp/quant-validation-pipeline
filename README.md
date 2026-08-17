# Quant Validation Pipeline

Pipeline de investigación para validar estrategias cuantitativas con datos
point-in-time, purged/embargoed temporal k-fold, diferenciación fraccional, DSR/PSR,
bootstrap selection-aware, modelos baseline, deep learning y PPO.

> Estado: investigación/paper trading. No envía órdenes por defecto y no es
> asesoría financiera.

## Qué valida

- Prevención de target leakage y look-ahead bias.
- Selección temporal de `d` fraccional dentro de cada fold.
- Purge/embargo acorde al lookback efectivo de FFD.
- Sharpe probabilístico y Deflated Sharpe Ratio con anualización configurable.
- Gate de ablación con bootstrap por bloques.
- Ledger reproducible de trials y metadatos de configuración.
- Adaptador MT5 en `dry_run`; las órdenes live requieren confirmaciones
  explícitas y credenciales inyectadas desde el entorno.

## Instalación

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Python requerido: 3.10 o superior. PyTorch se instala como dependencia porque
el pipeline importa los modelos deep/RL al arrancar.

## Tests

```bash
pytest -q
```

Los canarios estadísticos largos están marcados aparte:

```bash
RUN_LONG_CALIBRATION=1 pytest -m calibration
```

## Datos

El pipeline usa un único feed canónico configurado en
`config/pipeline_config.yaml`. El CSV no se versiona en GitHub: debe
descargarse localmente, normalizarse y validarse.

Fuente seleccionada: [EUR/USD M15 de Hugo Lemonnier](https://www.kaggle.com/datasets/hugolemonnier/eur-15min).

```bash
python scratch/normalize_eurusd.py
python scratch/validate_eurusd_source.py
python scratch/compare_eurusd_sources.py
```

El normalizador genera `data/raw/EURUSD_15min.csv`. La validación conserva los
huecos del feed y no interpola precios. La procedencia queda en
`config/eurusd_source.json`; el manifiesto local se genera en
`config/data_manifest.json` y está excluido del repositorio porque contiene
rutas locales y un hash específico del archivo descargado.

No se concatenan feeds: las fuentes secundarias tienen convenciones de sesión,
zona horaria y volumen distintas. Se usan únicamente como auditoría de
solapamiento. Consulta el detalle en
`reports/eurusd_source_evaluation.md`.

## Ejecutar el pipeline

Con el CSV validado:

```bash
python -m src --config config/pipeline_config.yaml --output outputs/result.json
```

El proceso reserva y evalúa el tramo OOS definido en la configuración antes de
permitir un `GO`. Un `NO-GO` devuelve código de salida distinto de cero; eso es
intencional.

## Estructura

```text
src/          código del pipeline
tests/        pruebas unitarias y canarios
config/       configuración y procedencia
scratch/      normalización y auditorías reproducibles
reports/      evidencia de validación
docs/         protocolo de ejecución/paper trading
data/raw/     solo README; datasets locales ignorados por Git
```

## Seguridad de ejecución

La configuración predeterminada mantiene `dry_run: true` y
`allow_live_orders: false`. Nunca se deben subir tokens, credenciales MT5,
`.env`, bases de datos locales ni datasets cuya licencia no permita su
redistribución. Antes de usar dinero real, seguir `docs/mt5_canary.md` y
validar paper trading, costes, spread, slippage, límites de pérdida y circuit
breaker.
