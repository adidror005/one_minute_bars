from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ib_async import IB, Option, Stock

from options import IbAsyncOptionPricingEngine, OptionContractSpec


InstrumentType = Literal["stock", "option"]


@dataclass
class ExecutionInstrumentConfig:
    type: InstrumentType = "stock"
    symbol: str = "META"
    exchange: str = "SMART"
    currency: str = "USD"
    expiry: str | None = None
    strike: float | None = None
    right: str | None = None
    multiplier: float = 100.0
    limit_entry_offset_pct: float = 0.02
    limit_exit_offset_pct: float = 0.02


class ExecutionInstrument:
    def resolve_contract(self) -> Any:
        raise NotImplementedError

    def subscribe_market_data(self) -> Any:
        raise NotImplementedError

    def cancel_market_data(self) -> None:
        raise NotImplementedError

    def estimate_market_price(self) -> float:
        raise NotImplementedError

    def estimate_entry_limit_price(self, underlying_reference_price: float) -> float:
        raise NotImplementedError

    def estimate_exit_limit_price(self, underlying_reference_price: float) -> float:
        raise NotImplementedError

    @property
    def instrument_type(self) -> InstrumentType:
        raise NotImplementedError


class IbkrExecutionInstrument(ExecutionInstrument):
    def __init__(
        self,
        ib: IB,
        config: ExecutionInstrumentConfig,
        option_engine: IbAsyncOptionPricingEngine | None = None,
    ):
        self.ib = ib
        self.config = config
        self.option_engine = option_engine or IbAsyncOptionPricingEngine(ib=ib)
        self._contract: Any | None = None
        self._ticker: Any | None = None

    @property
    def instrument_type(self) -> InstrumentType:
        return self.config.type

    def resolve_contract(self) -> Any:
        if self._contract is not None:
            return self._contract

        if self.config.type == "stock":
            self._contract = Stock(
                symbol=self.config.symbol,
                exchange=self.config.exchange,
                currency=self.config.currency,
            )
            self.ib.qualifyContracts(self._contract)
            return self._contract

        if self.config.type == "option":
            self._contract = self._build_option_contract(self._option_spec())
            return self._contract

        raise ValueError(f"Unsupported instrument type: {self.config.type}")

    def subscribe_market_data(self) -> Any:
        if self.config.type == "option":
            self.resolve_contract()
            return self.option_engine.subscribe_option_market_data(self._option_spec())

        if self._ticker is None:
            self._ticker = self.ib.reqMktData(self.resolve_contract(), "", False, False)

        return self._ticker

    def cancel_market_data(self) -> None:
        if self._ticker is not None:
            self.ib.cancelMktData(self.resolve_contract())
            self._ticker = None

        if hasattr(self.option_engine, "cancel_market_data"):
            self.option_engine.cancel_market_data()

    def estimate_market_price(self) -> float:
        if self.config.type == "stock":
            return self._estimate_stock_price()
        if self.config.type == "option":
            return self.option_engine.estimate_option_price(self._option_spec()).price
        raise ValueError(f"Unsupported instrument type: {self.config.type}")

    def estimate_entry_limit_price(self, underlying_reference_price: float) -> float:
        return self._limit_price(self.config.limit_entry_offset_pct)

    def estimate_exit_limit_price(self, underlying_reference_price: float) -> float:
        return self._limit_price(self.config.limit_exit_offset_pct)

    def _limit_price(self, offset: float) -> float:
        return round(max(0.01, self.estimate_market_price() * (1.0 - offset)), 2)

    def _estimate_stock_price(self) -> float:
        ticker = self.subscribe_market_data()

        for attr in ("last", "close", "marketPrice"):
            val = _ticker_value(ticker, attr)
            if val is not None and val > 0:
                return val

        bid_val = _safe_float(getattr(ticker, "bid", None))
        ask_val = _safe_float(getattr(ticker, "ask", None))
        if bid_val is not None and ask_val is not None and bid_val <= ask_val:
            return (bid_val + ask_val) / 2.0

        raise ValueError(f"Could not estimate stock price for {self.config.symbol}")

    def _option_spec(self) -> OptionContractSpec:
        return OptionContractSpec(
            symbol=self.config.symbol,
            expiry=self.config.expiry or "",
            strike=float(self.config.strike or 0.0),
            right=(self.config.right or "C").upper(),
            exchange=self.config.exchange,
            currency=self.config.currency,
            multiplier=self.config.multiplier,
        )

    def _build_option_contract(self, spec: OptionContractSpec) -> Option:
        option = Option(
            symbol=spec.symbol,
            lastTradeDateOrContractMonth=spec.expiry,
            strike=float(spec.strike),
            right=spec.right.upper(),
            exchange=spec.exchange,
            currency=spec.currency,
            multiplier=str(int(spec.multiplier)),
        )
        self.ib.qualifyContracts(option)
        return option


def build_execution_instrument(
    ib: object,
    execution_cfg: dict,
    symbol: str,
    option_engine: IbAsyncOptionPricingEngine | None = None,
) -> IbkrExecutionInstrument:
    instrument_cfg = execution_cfg.get("instrument", {})
    instrument_type = instrument_cfg.get("type", "stock")
    if instrument_type not in ("stock", "option"):
        raise ValueError(
            f"Unsupported instrument type {instrument_type!r}. "
            "Live execution currently supports 'stock' and single 'option' only."
        )

    config = ExecutionInstrumentConfig(
        type=instrument_type,
        symbol=symbol,
        exchange=instrument_cfg.get("exchange", "SMART"),
        currency=instrument_cfg.get("currency", "USD"),
        expiry=instrument_cfg.get("expiry"),
        strike=instrument_cfg.get("strike"),
        right=instrument_cfg.get("right"),
        multiplier=float(instrument_cfg.get("multiplier", 100.0)),
        limit_entry_offset_pct=float(
            instrument_cfg.get(
                "limit_entry_offset_pct",
                execution_cfg.get("limit_entry_offset_pct", 0.02),
            )
        ),
        limit_exit_offset_pct=float(
            instrument_cfg.get(
                "limit_exit_offset_pct",
                execution_cfg.get("limit_exit_offset_pct", 0.02),
            )
        ),
    )
    instrument = IbkrExecutionInstrument(ib=ib, config=config, option_engine=option_engine)
    instrument.subscribe_market_data()
    return instrument


def _ticker_value(ticker: Any, attr: str) -> float | None:
    value = getattr(ticker, attr, None)
    if callable(value):
        value = value()
    return _safe_float(value)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:
        return None
    return result
