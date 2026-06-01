from __future__ import annotations

from typing import Any

from ib_async import Stock


def build_underlying_stock_contract(symbol: str, market_data_cfg: dict, currency: str = "USD") -> Stock:
    kwargs = {}
    if market_data_cfg.get("primaryExchange"):
        kwargs["primaryExchange"] = market_data_cfg["primaryExchange"]

    return Stock(
        symbol,
        market_data_cfg.get("exchange", "SMART"),
        market_data_cfg.get("currency", currency),
        **kwargs,
    )


def request_underlying_realtime_bars(
    ib: Any,
    symbol: str,
    market_data_cfg: dict,
    currency: str = "USD",
):
    contract = qualify_underlying_stock_contract(
        ib=ib,
        symbol=symbol,
        market_data_cfg=market_data_cfg,
        currency=currency,
    )

    bars = ib.reqRealTimeBars(
        contract,
        barSize=int(market_data_cfg.get("barSize", 5)),
        whatToShow=market_data_cfg.get("whatToShow", "TRADES"),
        useRTH=bool(market_data_cfg.get("useRTH", False)),
    )

    return contract, bars


def qualify_underlying_stock_contract(
    ib: Any,
    symbol: str,
    market_data_cfg: dict,
    currency: str = "USD",
) -> Stock:
    exchanges = [
        market_data_cfg.get("exchange", "SMART"),
        *market_data_cfg.get("fallback_exchanges", []),
    ]
    errors = []

    for exchange in dict.fromkeys(exchanges):
        cfg = dict(market_data_cfg)
        cfg["exchange"] = exchange
        contract = build_underlying_stock_contract(
            symbol=symbol,
            market_data_cfg=cfg,
            currency=currency,
        )

        try:
            qualified = ib.qualifyContracts(contract)
        except Exception as exc:
            errors.append(f"{exchange}: {exc}")
            continue

        if qualified:
            return qualified[0]

        errors.append(f"{exchange}: no qualified contract returned")

    raise RuntimeError(
        f"Could not qualify stock contract for {symbol}. Tried exchanges={exchanges}. "
        f"Errors: {errors}"
    )
