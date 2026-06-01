from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ib_async import Option, Stock

from options import IbAsyncOptionPricingEngine, OptionContractSpec


InstrumentType = Literal["stock", "option"]


@dataclass
class ExecutionInstrumentConfig:
    type: InstrumentType = "stock"
    symbol: str = "META"
    exchange: str = "SMART"
    currency: str = "USD"
    # Single option fields
    expiry: str | None = None
    strike: float | None = None
    right: str | None = None
    multiplier: float = 100.0
    # Limit offsets applied to estimated execution-instrument price
    limit_entry_offset_pct: float = 0.02
    limit_exit_offset_pct: float = 0.02


class ExecutionInstrument:
    def resolve_contract(self) -> Any:
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
    """Routes orders to stock or a single option while signals stay on underlying."""

    def __init__(
        self,
        ib: object,
        config: ExecutionInstrumentConfig,
        option_engine: IbAsyncOptionPricingEngine | None = None,
    ):
        self.ib = ib
        self.config = config
        self.option_engine = option_engine or IbAsyncOptionPricingEngine(ib=ib)
        self._contract: Any | None = None

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
            self._contract = self._build_option_contract(
                OptionContractSpec(
                    symbol=self.config.symbol,
                    expiry=self.config.expiry or "",
                    strike=float(self.config.strike or 0.0),
                    right=(self.config.right or "C").upper(),
                    exchange=self.config.exchange,
                    currency=self.config.currency,
                    multiplier=self.config.multiplier,
                )
            )
            return self._contract

        raise ValueError(f"Unsupported instrument type: {self.config.type}")

    def estimate_market_price(self) -> float:
        return self._estimate_market_price()

    def estimate_entry_limit_price(self, underlying_reference_price: float) -> float:
        base = self._estimate_market_price()
        offset = self.config.limit_entry_offset_pct
        return round(max(0.01, base * (1.0 - offset)), 2)

    def estimate_exit_limit_price(self, underlying_reference_price: float) -> float:
        base = self._estimate_market_price()
        offset = self.config.limit_exit_offset_pct
        return round(max(0.01, base * (1.0 - offset)), 2)

    def _estimate_market_price(self) -> float:
        if self.config.type == "stock":
            return self._estimate_stock_price()

        if self.config.type == "option":
            spec = OptionContractSpec(
                symbol=self.config.symbol,
                expiry=self.config.expiry or "",
                strike=float(self.config.strike or 0.0),
                right=(self.config.right or "C").upper(),
                exchange=self.config.exchange,
                currency=self.config.currency,
                multiplier=self.config.multiplier,
            )
            return self.option_engine.estimate_option_price(spec).price

        raise ValueError(f"Unsupported instrument type: {self.config.type}")

    def _estimate_stock_price(self) -> float:
        stock = Stock(
            symbol=self.config.symbol,
            exchange=self.config.exchange,
            currency=self.config.currency,
        )
        self.ib.qualifyContracts(stock)
        ticker = self.ib.reqMktData(stock, "", False, False)
        self.ib.sleep(0.2)
        for attr in ("last", "close", "marketPrice"):
            val = getattr(ticker, attr, None)
            if val is not None and float(val) > 0:
                return float(val)
        bid = getattr(ticker, "bid", None)
        ask = getattr(ticker, "ask", None)
        if bid is not None and ask is not None and bid <= ask:
            return (float(bid) + float(ask)) / 2.0
        raise ValueError(f"Could not estimate stock price for {self.config.symbol}")

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
    return IbkrExecutionInstrument(ib=ib, config=config, option_engine=option_engine)
