# ============================================================
# LIMIT ORDER ROUTING
# Default = IBKR
# Optional = Alpaca
# ============================================================
from dataclasses import dataclass
from enum import Enum
from ib_async import Option, Stock
from typing import Any, Union


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

TradeContract = Union[Stock, Option]


@dataclass
class LimitOrderSpec:
    contract: TradeContract
    side: OrderSide
    qty: int
    limit_price: float
    tif: str = "DAY"
    outside_rth: bool = True
    client_order_id: str | None = None
    extra: dict | None = None

class IBKRLimitOrderRouter:
    def __init__(self, ib, contract: TradeContract | None = None):
        self.ib = ib
        self.contract = contract

    def submit(self, order_spec: LimitOrderSpec):
        from ib_async import LimitOrder

        contract = order_spec.contract or self.contract
        if contract is None:
            raise ValueError("LimitOrderSpec.contract is required when router has no default contract.")

        order = LimitOrder(
            order_spec.side.value,
            order_spec.qty,
            order_spec.limit_price,
            tif=order_spec.tif,
            outsideRth=order_spec.outside_rth,
        )

        if order_spec.client_order_id:
            order.orderRef = order_spec.client_order_id

        if order_spec.extra:
            for k, v in order_spec.extra.items():
                setattr(order, k, v)

        return self.ib.placeOrder(contract, order)

class AlpacaLimitOrderRouter:
    def __init__(self, client):
        self.client = client

    def submit(self, order_spec: LimitOrderSpec):
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce

        side = AlpacaSide.BUY if order_spec.side == OrderSide.BUY else AlpacaSide.SELL
        tif = TimeInForce.DAY if order_spec.tif == "DAY" else TimeInForce.GTC

        kwargs = {
            "symbol": order_spec.contract.symbol,#@TODO double check if Alpaca holds this
            "qty": order_spec.qty,
            "side": side,
            "limit_price": order_spec.limit_price,
            "time_in_force": tif,
            "extended_hours": order_spec.outside_rth,
        }

        if order_spec.client_order_id:
            kwargs["client_order_id"] = order_spec.client_order_id

        if order_spec.extra:
            kwargs.update(order_spec.extra)

        req = LimitOrderRequest(**kwargs)
        return self.client.submit_order(req)
