from types import SimpleNamespace

import pytest

from src.execution.mt5 import LIVE_CONFIRMATION, MT5Executor, MarketOrder


class FakeMT5:
    TRADE_ACTION_DEAL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_IOC = 1
    TRADE_RETCODE_DONE = 10009

    def __init__(self):
        self.requests = []

    def symbol_select(self, symbol, enabled):
        return symbol == "EURUSD" and enabled

    def symbol_info_tick(self, symbol):
        return SimpleNamespace(ask=1.1002, bid=1.1000)

    def order_send(self, request):
        self.requests.append(request)
        return SimpleNamespace(retcode=10009, order=7, deal=8, price=request["price"])


def test_dry_run_never_needs_mt5_client():
    result = MT5Executor(dry_run=True).send_market_order(MarketOrder("EURUSD", 0.01, 1))
    assert result.accepted is True
    assert result.raw["dry_run"] is True


def test_live_order_is_validated_and_normalized(monkeypatch):
    client = FakeMT5()
    monkeypatch.setenv("MT5_LIVE_CONFIRMATION", LIVE_CONFIRMATION)
    result = MT5Executor(
        client,
        dry_run=False,
        allow_live_orders=True,
    ).send_market_order(
        MarketOrder("EURUSD", 0.01, 1)
    )
    assert result.accepted is True
    assert result.order_id == 7
    assert client.requests[0]["type"] == client.ORDER_TYPE_BUY
    assert client.requests[0]["price"] == 1.1002


def test_live_order_requires_two_explicit_guards(monkeypatch):
    monkeypatch.delenv("MT5_LIVE_CONFIRMATION", raising=False)
    with pytest.raises(RuntimeError, match="allow_live_orders"):
        MT5Executor(FakeMT5(), dry_run=False)

    with pytest.raises(RuntimeError, match="MT5_LIVE_CONFIRMATION"):
        MT5Executor(FakeMT5(), dry_run=False, allow_live_orders=True)


@pytest.mark.parametrize("order", [
    MarketOrder("EURUSD", 0.0, 1),
    MarketOrder("EURUSD", 0.01, 0),
    MarketOrder(" EURUSD", 0.01, 1),
])
def test_invalid_orders_fail_closed(order):
    with pytest.raises(ValueError):
        MT5Executor().send_market_order(order)
