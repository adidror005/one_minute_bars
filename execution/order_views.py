from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


def active_orders(ib: Any, refresh: bool = True) -> pd.DataFrame:
    """Return active/open IB trades as a DataFrame."""
    if refresh and hasattr(ib, "reqOpenOrders"):
        ib.reqOpenOrders()

    trades = list(_call_or_empty(ib, "openTrades"))
    if not trades:
        trades = [
            trade
            for trade in _call_or_empty(ib, "trades")
            if _trade_is_active(trade)
        ]

    return trades_df(trades)


def filled_orders(ib: Any) -> pd.DataFrame:
    """Return filled trades as a DataFrame."""
    trades = [
        trade
        for trade in _call_or_empty(ib, "trades")
        if _trade_is_filled(trade)
    ]

    return trades_df(trades)


def all_orders(ib: Any) -> pd.DataFrame:
    """Return all trades known to the current IB session."""
    return trades_df(_call_or_empty(ib, "trades"))


def recent_fills(ib: Any) -> pd.DataFrame:
    """Return execution fills known to the current IB session."""
    rows = []

    for fill in _call_or_empty(ib, "fills"):
        contract = getattr(fill, "contract", None)
        execution = getattr(fill, "execution", None)
        commission = getattr(fill, "commissionReport", None)

        rows.append({
            "time": getattr(execution, "time", None),
            "symbol": getattr(contract, "symbol", None),
            "secType": getattr(contract, "secType", None),
            "exchange": getattr(contract, "exchange", None),
            "side": getattr(execution, "side", None),
            "shares": getattr(execution, "shares", None),
            "price": getattr(execution, "price", None),
            "avgPrice": getattr(execution, "avgPrice", None),
            "orderId": getattr(execution, "orderId", None),
            "execId": getattr(execution, "execId", None),
            "commission": getattr(commission, "commission", None),
            "realizedPNL": getattr(commission, "realizedPNL", None),
        })

    return pd.DataFrame(rows)


def trades_df(trades) -> pd.DataFrame:
    rows = []

    for trade in trades:
        contract = getattr(trade, "contract", None)
        order = getattr(trade, "order", None)
        status = getattr(trade, "orderStatus", None)

        rows.append({
            "symbol": getattr(contract, "symbol", None),
            "secType": getattr(contract, "secType", None),
            "lastTradeDateOrContractMonth": getattr(contract, "lastTradeDateOrContractMonth", None),
            "strike": getattr(contract, "strike", None),
            "right": getattr(contract, "right", None),
            "exchange": getattr(contract, "exchange", None),
            "action": getattr(order, "action", None),
            "orderType": getattr(order, "orderType", None),
            "totalQuantity": getattr(order, "totalQuantity", None),
            "lmtPrice": getattr(order, "lmtPrice", None),
            "tif": getattr(order, "tif", None),
            "outsideRth": getattr(order, "outsideRth", None),
            "orderId": getattr(order, "orderId", None),
            "permId": getattr(order, "permId", None),
            "orderRef": getattr(order, "orderRef", None),
            "status": getattr(status, "status", None),
            "filled": getattr(status, "filled", None),
            "remaining": getattr(status, "remaining", None),
            "avgFillPrice": getattr(status, "avgFillPrice", None),
            "lastFillPrice": getattr(status, "lastFillPrice", None),
            "whyHeld": getattr(status, "whyHeld", None),
        })

    return pd.DataFrame(rows)


def strategy_open_trades(algo: Any) -> pd.DataFrame:
    """Return the strategy's persisted open-trade state."""
    rows = []

    for trade in getattr(algo, "open_trades", []):
        rows.append({
            "qty": trade.get("qty"),
            "entry_price": trade.get("entry_price"),
            "entry_underlying_price": trade.get("entry_underlying_price"),
            "entry_date": trade.get("entry_date"),
            "entry_time": trade.get("entry_time"),
            "entry_reason": trade.get("entry_reason"),
            "instrument_type": trade.get("instrument_type"),
            "ml_prob": trade.get("ml_prob"),
        })

    return pd.DataFrame(rows)


def print_order_snapshot(ib: Any, algo: Any | None = None) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    print(f"Order snapshot: {now}")
    print(f"Active IB orders: {len(active_orders(ib, refresh=False))}")
    print(f"Filled IB orders: {len(filled_orders(ib))}")
    print(f"Recent fills: {len(recent_fills(ib))}")
    if algo is not None:
        print(f"Strategy open trades: {len(getattr(algo, 'open_trades', []))}")


def _call_or_empty(obj: Any, method_name: str):
    method = getattr(obj, method_name, None)
    if method is None:
        return []
    result = method()
    return result if result is not None else []


def _trade_is_active(trade: Any) -> bool:
    status = getattr(getattr(trade, "orderStatus", None), "status", None)
    remaining = getattr(getattr(trade, "orderStatus", None), "remaining", None)
    if remaining is not None and remaining > 0:
        return status not in {"Cancelled", "ApiCancelled", "Inactive"}
    return status in {"PendingSubmit", "PreSubmitted", "Submitted"}


def _trade_is_filled(trade: Any) -> bool:
    status = getattr(getattr(trade, "orderStatus", None), "status", None)
    remaining = getattr(getattr(trade, "orderStatus", None), "remaining", None)
    filled = getattr(getattr(trade, "orderStatus", None), "filled", None)
    if status == "Filled":
        return True
    if filled is not None and filled > 0 and remaining == 0:
        return True
    return False
