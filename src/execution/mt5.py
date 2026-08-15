"""Minimal, safe MetaTrader 5 market-order adapter.

The adapter never logs in, starts a terminal, or sends an order by itself.  A
live client must be injected explicitly and ``dry_run`` defaults to ``True``.
This keeps research/backtests independent from broker state and makes the
adapter straightforward to test with a fake client.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any


LIVE_CONFIRMATION = "I_UNDERSTAND_LIVE_MT5"


@dataclass(frozen=True)
class MarketOrder:
    """A normalized market order: ``side`` is ``1`` for buy, ``-1`` for sell."""

    symbol: str
    volume: float
    side: int
    comment: str = "quant-validation"


@dataclass(frozen=True)
class ExecutionResult:
    """Broker response normalized without hiding the raw platform response."""

    retcode: int
    order_id: int | None
    deal_id: int | None
    price: float | None
    accepted: bool
    raw: Any = None


class MT5Executor:
    """Send one validated market order through an injected MT5 client.

    ``client`` is expected to expose the small subset of the MetaTrader5
    module used here.  When omitted in live mode, the official package is
    imported, but no ``initialize`` or authentication call is made.
    """

    def __init__(
        self,
        client: Any = None,
        *,
        dry_run: bool = True,
        allow_live_orders: bool = False,
        magic: int = 260814,
        deviation_points: int = 20,
        max_volume: float | None = None,
    ) -> None:
        if magic < 0:
            raise ValueError("magic must be non-negative")
        if deviation_points < 0:
            raise ValueError("deviation_points must be non-negative")
        if max_volume is not None and (not math.isfinite(max_volume) or max_volume <= 0):
            raise ValueError("max_volume must be positive when provided")
        self.client = client
        self.dry_run = dry_run
        self.allow_live_orders = allow_live_orders
        self.magic = magic
        self.deviation_points = deviation_points
        self.max_volume = max_volume
        if not dry_run:
            if not allow_live_orders:
                raise RuntimeError(
                    "Live MT5 disabled: set allow_live_orders=True explicitly"
                )
            if os.getenv("MT5_LIVE_CONFIRMATION") != LIVE_CONFIRMATION:
                raise RuntimeError(
                    "Live MT5 disabled: MT5_LIVE_CONFIRMATION is missing"
                )

    def _client(self) -> Any:
        if self.client is not None:
            return self.client
        try:
            import MetaTrader5 as mt5  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "MetaTrader5 no esta instalado; inyecta un cliente o instala el paquete "
                "solo para ejecucion en paper/live"
            ) from exc
        self.client = mt5
        return mt5

    def send_market_order(self, order: MarketOrder) -> ExecutionResult:
        self._validate(order)
        if self.dry_run:
            return ExecutionResult(
                retcode=0,
                order_id=None,
                deal_id=None,
                price=None,
                accepted=True,
                raw={"dry_run": True, "order": order},
            )

        mt5 = self._client()
        if not mt5.symbol_select(order.symbol, True):
            raise RuntimeError(f"MT5 no pudo seleccionar el simbolo {order.symbol}")
        tick = mt5.symbol_info_tick(order.symbol)
        if tick is None:
            raise RuntimeError(f"MT5 no devolvio tick para {order.symbol}")

        is_buy = order.side == 1
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": order.symbol,
            "volume": order.volume,
            "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
            "price": float(tick.ask if is_buy else tick.bid),
            "deviation": self.deviation_points,
            "magic": self.magic,
            "comment": order.comment,
            "type_time": getattr(mt5, "ORDER_TIME_GTC", 0),
            "type_filling": getattr(mt5, "ORDER_FILLING_IOC", 0),
        }
        raw = mt5.order_send(request)
        if raw is None:
            raise RuntimeError("MT5 order_send devolvio None")

        retcode = int(getattr(raw, "retcode", -1))
        accepted_codes = {
            int(getattr(mt5, "TRADE_RETCODE_DONE", 10009)),
            int(getattr(mt5, "TRADE_RETCODE_PLACED", 10008)),
            int(getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010)),
        }
        return ExecutionResult(
            retcode=retcode,
            order_id=_optional_int(getattr(raw, "order", None)),
            deal_id=_optional_int(getattr(raw, "deal", None)),
            price=_optional_float(getattr(raw, "price", None)),
            accepted=retcode in accepted_codes,
            raw=raw,
        )

    def _validate(self, order: MarketOrder) -> None:
        if not order.symbol or order.symbol.strip() != order.symbol:
            raise ValueError("symbol must be a non-empty, trimmed string")
        if not math.isfinite(order.volume) or order.volume <= 0:
            raise ValueError("volume must be finite and positive")
        if self.max_volume is not None and order.volume > self.max_volume:
            raise ValueError("volume exceeds configured max_volume")
        if order.side not in (-1, 1):
            raise ValueError("side must be 1 (buy) or -1 (sell)")


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
