from dataclasses import dataclass
from enum import Enum
from ib_async import Option, Stock
from typing import Union


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
