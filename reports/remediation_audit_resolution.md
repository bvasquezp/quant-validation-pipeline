# Resolución de la auditoría de remediación

Fecha: 2026-08-17
Repositorio: `quant-validation-pipeline`

## Resultado

Se implementaron los cambios de remediación en cinco frentes coordinados:

1. empaquetado y dependencias;
2. reserva y evaluación OOS;
3. selección causal de diferenciación fraccional y purge dinámico;
4. nomenclatura honesta del esquema de validación y causalidad de la TCN;
5. pruebas e integración del gate.

La suite automatizada queda en **67 passed, 0 failed, 2 skipped**.

## Hallazgos cerrados en código

### Instalación reproducible

- Se corrigió el backend inválido de `pyproject.toml` a `setuptools.build_meta`.
- Se reemplazó el pin inexistente de scikit-learn por `>=1.4.1,<1.6`.
- PyTorch quedó declarado como dependencia principal, porque los módulos deep/RL lo importan al arrancar.
- `pip install -e . --no-deps` construyó e instaló correctamente el paquete editable.

### OOS reservado y usado por el gate

- Los datos reales se separan en desarrollo y reserva final antes de construir los folds.
- La reserva final se evalúa después de seleccionar el campeón de desarrollo.
- En datos reales, el gate fuerza `NO-GO` si no existe evidencia OOS final, si hay pocas observaciones o si el DSR final no supera el umbral.
- En calibraciones sintéticas se marca explícitamente `not_applicable_synthetic`, sin fingir que existe OOS independiente.

Smoke real sobre una muestra de 16.000 barras:

```text
data_mode: real
final_oos_status: evaluated
final_oos_models: buy_hold, lightgbm, ridge, sma_crossover
recommended_stage: Stage 0 (buy_and_hold)
gate_decision: NO-GO
```

El `NO-GO` fue producido por la evidencia OOS final, no por ausencia de evaluación: el DSR OOS de Buy&Hold fue `0.3954`, inferior al mínimo configurado `0.95`.

### FFD point-in-time

- La primera fold ya no puede seleccionar `d` usando posiciones futuras de `train`.
- Si no existe historial pre-test suficiente, el fold queda explícitamente no evaluable, con `d_used=None`.
- El purge se recalcula hasta punto fijo para cubrir el lookback efectivo de FFD.
- El ledger registra `d_used_by_fold`, `ffd_lookback_by_fold`, método de selección y `effective_purge_window`.
- El smoke real produjo `max_ffd_lookback=2274` y el diagnóstico registró `empty_fold_ids=[0]`; el fold no evaluable no se oculta.

### Validación temporal

La implementación utilizada es un **purged/embargoed temporal k-fold**. El nombre `purged_kfold_cv` reemplaza el nombre CPCV incorrecto; queda un wrapper de compatibilidad para consumidores antiguos, pero no se afirma que se generen combinaciones CPCV completas.

### TCN causal

Se corrigió el padding causal de la TCN y se añadió una prueba que verifica que la salida conserva la longitud y no usa observaciones futuras.

## Pendiente antes del sign-off definitivo

La corrida completa sobre todo el histórico real debe ejecutarse en el entorno con memoria/tiempo suficiente, por ejemplo Colab. Las pruebas y el smoke de integración validan el comportamiento, pero no sustituyen esa corrida final de calibración ni la validación económica con costos, spread, slippage y ejecución.

Por tanto, el estado correcto es: **remediación implementada y verificada en tests + smoke real; sign-off final condicionado a la corrida completa y a la validación de costos de transacción**.
