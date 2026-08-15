# Reporte de cierre — Remediación DSR

Fecha de ejecución: 2026-08-14  
Datos de auditoría: OHLCV sintético FX 24/5, configuración `15min`; calibración
nula de 300 semillas y control positivo de 30 semillas.

## Resultado ejecutivo

La remediación estadística principal quedó implementada: Ridge queda dentro de
la referencia MC, la anualización es explícita para FX 24/5, los candidatos se
alinean por timestamp y el gate re-selecciona campeones dentro de cada réplica
nula. La calibración de 300 semillas produjo `20/300 GO` (FPR puntual `6.67%`,
IC95% Wilson `[4.36%, 10.07%]`), compatible con el 5% nominal pero no idéntica.
El control positivo produjo `30/30 GO`.

Esto no valida una estrategia financiera real: todavía hace falta verificar
costes, slippage, spread, liquidez y un OOS con datos reales.

## Hallazgos y evidencia

| # | Hallazgo | Estado | Evidencia |
|---|---|---|---|
| 1 | Leak de target | Cerrado previamente | El retorno objetivo es futuro (`shift(-1)`); el checkpoint de auditoría reportó caída de Sharpe de 20.9 a 0.27 en Ridge. |
| 2 | Escala SR anualizada/por periodo | Cerrado | La rama anualizada y la periódica producen el mismo DSR; 62 tests pasan. |
| 3 | TCN duplicado | Cerrado | 30 semillas: correlación TCN/Attention media `-0.0798`, rango `[-0.3323, 0.3050]`, duplicados `0/30`. |
| 4 | Anualización hardcodeada | Cerrado | El factor deriva del calendario FX: `15min=24192`, `1h=6048`; también se aplica al bootstrap. |
| 5 | Configuración 15min vs generador 1h | Cerrado | El generador usa `config.data.frequency` y, para `fx_24_5`, omite sábados y domingos. CSV con frecuencia/calendario distinto falla explícitamente. |
| 6 | Rama `annualized_input=True` sin cobertura | Cerrado | Test de equivalencia anualizado/periódico añadido. |
| 7 | Selección global de `d` | Cerrado bajo política causal alternativa | ADF usa el prefijo pre-test de cada fold y el entrenamiento no usa observaciones posteriores al test. En datos reales el purge se eleva al lookback FFD; el fallback sintético conserva `train_policy=pre_test_only` porque `2274 > 10/5`. |
| 8 | Ridge elevado bajo H0 | Cerrado con trazabilidad ampliada | 30 semillas finales: DSR medio `0.2863`, referencia `0.2722`, diferencia `0.30 SE`; el reporte ahora separa `T_nominal`, `T_effective` y `n_observations`. |

## Placebo final de 30 semillas

| Modelo | DSR medio | Desv. estándar | Error estándar | T nominal mediano | T efectivo mediano | N obs mediano |
|---|---:|---:|---:|---:|---:|---:|
| buy_hold | 0.5776 | 0.3388 | 0.0619 | 4999.0 | 4937.1 | 4999.0 |
| sma_crossover | 0.4183 | 0.3054 | 0.0558 | 4999.0 | 4995.7 | 4999.0 |
| ridge | 0.2863 | 0.2721 | 0.0497 | 3999.0 | 3948.4 | 3999.0 |
| lightgbm | 0.3034 | 0.2947 | 0.0538 | 3999.0 | 4034.9 | 3999.0 |

El lookback FFD máximo observado fue `2274` barras, superior a `purge=10` y
`embargo=5`. La garantía aplicada es: cada fold solo entrena con observaciones
anteriores al inicio del test (`train_policy=pre_test_only`). Por tanto, la
ventana FFD causal no cruza el bloque de test, pero esto es una política
alternativa; no equivale a demostrar la desigualdad literal solicitada.

## Ledger

Cada trial nuevo registra:

- frecuencia y factor de anualización;
- versión de configuración;
- método de selección de `d`;
- `d_used` cuando es único, y `d_used_by_fold` completo dentro de `hyperparams`
  cuando varía entre folds.

Tres filas reales de una ejecución produjeron:

| modelo | método `d` | factor | frecuencia | versión | `d` por fold |
|---|---|---:|---|---|---|
| `buy_and_hold` | — | 24192 | `15min` | `2` | — |
| `sma_crossover` | — | 24192 | `15min` | `2` | — |
| `ridge_regression` | `rule_based_adf` | 24192 | `15min` | `2` | `[0.4, 0.5, 0.4, 0.4, 0.3]` |

El valor por fold se serializa dentro de `hyperparams`; `d_used` queda vacío
cuando no existe un único `d` global.

## Gate de ablación

Tras corregir el lookup y activar el bootstrap selection-aware, las corridas
end-to-end dieron:

- Nula (`mu=0`, 300 semillas): `GO=20/300`, `NO-GO=280/300`, FPR `6.6667%`,
  IC95% Wilson `[4.3567%, 10.0724%]`.
- Control positivo fuerte (`mu=5.0`, 30 semillas): `GO=30/30`, `NO-GO=0/30`,
  IC95% Wilson `[88.6483%, 100%]`.
- recomendación nula: Buy&Hold `142`, SMA `130`, Ridge `18`, LightGBM `10`.

La nula no demuestra una desviación estadísticamente concluyente respecto de
5%, pero sí justifica conservar el gate bajo observación y no cambiar `alpha`
sin una nueva especificación. El control positivo demuestra sensibilidad.

## Verificación técnica

- `pytest -q`: **62 passed**, 2 canarios largos omitidos por defecto.
- Canarios largos: **2 passed** (`300` semillas nulas y `30` de control positivo)
  con `RUN_LONG_CALIBRATION=1 pytest -m calibration`.
- Equivalencia numérica (`A=35040`): DSR anualizado `0.7374535521`, periódico
  `0.7374535521`, diferencia absoluta `0.0`.
- Entorno `FxTradingEnv` y PPO discreto causal implementados; el pipeline lo
  activa únicamente con `models.rl.enabled=true`.
- Adaptador MT5 aislado en `src/execution/mt5.py`: no forma parte del pipeline,
  `dry_run=true` por defecto y el modo live exige `allow_live_orders=True` más
  `MT5_LIVE_CONFIRMATION=I_UNDERSTAND_LIVE_MT5`. La configuración limita el
  canary por defecto a `0.01` lotes.
- Canario estadístico registrado en `tests/test_calibration.py`: el test rápido
  verifica el reporte de Wilson; las corridas de 300/30 semillas se ejecutan
  con `RUN_LONG_CALIBRATION=1 pytest -m calibration`.
- Auditoría reproducible: `scratch/remediation_audit.py`.
- Datos sintéticos configurados a `15min`; el uso de datos reales requiere que
  el CSV respete esa frecuencia o el cargador rechazará la entrada.

## Límite del cierre

El cierre valida la infraestructura estadística y temporal sobre datos
sintéticos, y deja preparado un adaptador seguro para paper/canary. Antes de
usar dinero real todavía se necesita: cargar un CSV real y su manifest,
reentrenar con OOS reservado, conectar un cliente MT5 autenticado de forma
externa, validar señal en la siguiente barra, tamaño de posición, comisiones,
spread, slippage, liquidez, límite diario y circuit breaker. Ver
`docs/mt5_canary.md`.
