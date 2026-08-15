"""Execution adapters kept separate from research and simulation code."""

from .mt5 import ExecutionResult, MarketOrder, MT5Executor

__all__ = ["ExecutionResult", "MarketOrder", "MT5Executor"]
