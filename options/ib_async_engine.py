from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from ib_async import Option

from .engine import (
    DefaultOptionPricingEngine,
    OptionContractSpec,
    OptionMarketDataProvider,
    OptionQuote,
)


@dataclass
class IbAsyncOptionMarketDataProvider(OptionMarketDataProvider):
    ib: object
    _tickers: dict[OptionContractSpec, object] | None = None

    def get_option_quote(self, contract: OptionContractSpec) -> OptionQuote:
        ticker = self.subscribe_option_market_data(contract)

        bid = _safe_float(getattr(ticker, "bid", None))
        ask = _safe_float(getattr(ticker, "ask", None))
        last = _safe_float(getattr(ticker, "last", None))
        midpoint = _compute_mid(bid, ask)
        model_greeks = getattr(ticker, "modelGreeks", None)
        under = _safe_float(getattr(ticker, "undPrice", None))
        if under is None and model_greeks is not None:
            under = _safe_float(getattr(model_greeks, "undPrice", None))

        return OptionQuote(
            contract=contract,
            bid=bid,
            ask=ask,
            last=last,
            midpoint=midpoint,
            model_price=_safe_float(getattr(model_greeks, "optPrice", None)) if model_greeks else None,
            implied_vol=_safe_float(getattr(model_greeks, "impliedVol", None)) if model_greeks else None,
            delta=_safe_float(getattr(model_greeks, "delta", None)) if model_greeks else None,
            gamma=_safe_float(getattr(model_greeks, "gamma", None)) if model_greeks else None,
            vega=_safe_float(getattr(model_greeks, "vega", None)) if model_greeks else None,
            theta=_safe_float(getattr(model_greeks, "theta", None)) if model_greeks else None,
            underlier_price=under,
            timestamp=datetime.now(timezone.utc),
        )

    def subscribe_option_market_data(self, contract: OptionContractSpec):
        if self._tickers is None:
            self._tickers = {}

        ticker = self._tickers.get(contract)
        if ticker is not None:
            return ticker

        ib_contract = self._to_ib_option(contract)
        self.ib.qualifyContracts(ib_contract)
        ticker = self.ib.reqMktData(ib_contract, "", False, False)
        self._tickers[contract] = ticker
        return ticker

    def cancel_market_data(self) -> None:
        if not self._tickers:
            return

        for ticker in self._tickers.values():
            contract = getattr(ticker, "contract", None)
            if contract is not None:
                self.ib.cancelMktData(contract)
        self._tickers.clear()

    def get_similar_option_quotes(
        self,
        contract: OptionContractSpec,
        strike_span: int = 2,
    ) -> list[OptionQuote]:
        # Pull a local strike neighborhood for smile-based interpolation.
        chain = self.ib.reqSecDefOptParams(
            underlyingSymbol=contract.symbol,
            futFopExchange="",
            underlyingSecType="STK",
            underlyingConId=0,
        )
        if not chain:
            return []

        strikes = sorted(set(chain[0].strikes))
        if not strikes:
            return []

        nearest_ix = min(range(len(strikes)), key=lambda i: abs(strikes[i] - contract.strike))
        lo = max(0, nearest_ix - strike_span)
        hi = min(len(strikes), nearest_ix + strike_span + 1)
        selected = strikes[lo:hi]

        quotes: list[OptionQuote] = []
        for strike in selected:
            neighbor = OptionContractSpec(
                symbol=contract.symbol,
                expiry=contract.expiry,
                strike=float(strike),
                right=contract.right,
                exchange=contract.exchange,
                currency=contract.currency,
                multiplier=contract.multiplier,
            )
            try:
                quotes.append(self.get_option_quote(neighbor))
            except Exception:
                continue
        return quotes

    def get_put_call_quotes_same_strike(
        self,
        contract: OptionContractSpec,
    ) -> tuple[OptionQuote | None, OptionQuote | None]:
        call_contract = OptionContractSpec(
            symbol=contract.symbol,
            expiry=contract.expiry,
            strike=contract.strike,
            right="C",
            exchange=contract.exchange,
            currency=contract.currency,
            multiplier=contract.multiplier,
        )
        put_contract = OptionContractSpec(
            symbol=contract.symbol,
            expiry=contract.expiry,
            strike=contract.strike,
            right="P",
            exchange=contract.exchange,
            currency=contract.currency,
            multiplier=contract.multiplier,
        )
        call_q = None
        put_q = None
        try:
            call_q = self.get_option_quote(call_contract)
        except Exception:
            call_q = None
        try:
            put_q = self.get_option_quote(put_contract)
        except Exception:
            put_q = None
        return call_q, put_q

    @staticmethod
    def _to_ib_option(contract: OptionContractSpec) -> Option:
        return Option(
            symbol=contract.symbol,
            lastTradeDateOrContractMonth=contract.expiry,
            strike=float(contract.strike),
            right=contract.right.upper(),
            exchange=contract.exchange,
            currency=contract.currency,
            multiplier=str(int(contract.multiplier)),
        )


class IbAsyncOptionPricingEngine(DefaultOptionPricingEngine):
    def __init__(
        self,
        ib: object,
        risk_free_rate: float = 0.03,
        risk_free_rate_provider=None,
        expiry_timezone: str = "America/New_York",
        expiry_close_hour: int = 16,
        expiry_close_minute: int = 0,
    ):
        self.ib_data_provider = IbAsyncOptionMarketDataProvider(ib=ib)
        super().__init__(
            data_provider=self.ib_data_provider,
            risk_free_rate=risk_free_rate,
            risk_free_rate_provider=risk_free_rate_provider,
            expiry_timezone=expiry_timezone,
            expiry_close_hour=expiry_close_hour,
            expiry_close_minute=expiry_close_minute,
        )

    def subscribe_option_market_data(self, contract: OptionContractSpec):
        return self.ib_data_provider.subscribe_option_market_data(contract)

    def cancel_market_data(self) -> None:
        self.ib_data_provider.cancel_market_data()


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        val = float(v)
    except (TypeError, ValueError):
        return None
    if val != val:  # NaN
        return None
    return val


def _compute_mid(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None:
        return None
    if bid > ask:
        return None
    return (bid + ask) / 2.0
