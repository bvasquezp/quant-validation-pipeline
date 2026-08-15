# MT5: protocolo mínimo de paper/canary

El adaptador de `src/execution/mt5.py` no inicia MetaTrader, no hace login y no
envía órdenes por defecto. La configuración queda en `execution.dry_run: true`
y `execution.allow_live_orders: false`.

Secuencia antes de usar dinero real:

1. Cargar y validar el CSV real; conservar su `data_manifest.json` y su commit.
2. Entrenar solo con el tramo de desarrollo y reservar el OOS final.
3. Ejecutar el modelo en paper, registrando señal, timestamp, precio observado,
   spread, slippage y decisión del gate.
4. Inyectar un cliente MT5 explícitamente y probar primero una orden ficticia.
5. Pasar a canary con el volumen mínimo permitido por el broker, límite diario
   de pérdida, límite de exposición y circuit breaker.
6. Revisar manualmente cada ejecución antes de permitir más de una posición.

Ejemplo de uso seguro:

```python
from src.execution import MT5Executor, MarketOrder

executor = MT5Executor(dry_run=True, max_volume=0.01)
result = executor.send_market_order(MarketOrder("EURUSD", 0.01, side=1))
assert result.accepted and result.raw["dry_run"]
```

Para live/paper conectado, el cliente MT5 debe ser inyectado por el proceso de
ejecución ya autenticado. Además de `dry_run=False`, el código exige
`allow_live_orders=True` y la variable de entorno
`MT5_LIVE_CONFIRMATION=I_UNDERSTAND_LIVE_MT5`. Este repositorio no guarda
credenciales ni contiene un comando que active la conexión automáticamente.
