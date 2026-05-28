from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Callable, Protocol
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class OptionContractSpec:
    symbol: str
    expiry: str  # YYYYMMDD for IB-like contracts
    strike: float
    right: str  # "C" or "P"
    exchange: str = "SMART"
    currency: str = "USD"
    multiplier: float = 100.0


@dataclass(frozen=True)
class OptionQuote:
    contract: OptionContractSpec
    bid: float | None
    ask: float | None
    last: float | None
    midpoint: float | None
    model_price: float | None
    implied_vol: float | None
    delta: float | None
    gamma: float | None
    vega: float | None
    theta: float | None
    underlier_price: float | None
    timestamp: datetime


@dataclass(frozen=True)
class ComboLegSpec:
    contract: OptionContractSpec
    ratio: int = 1
    side: str = "BUY"  # "BUY" or "SELL"

    @property
    def signed_ratio(self) -> int:
        return self.ratio if self.side.upper() == "BUY" else -self.ratio


@dataclass(frozen=True)
class EngineQuote:
    price: float
    source: str
    confidence: float
    details: dict


@dataclass(frozen=True)
class PredictedMove:
    option_price_change: float
    stock_price_change: float
    estimated_new_option_price: float
    confidence: float
    details: dict


class OptionMarketDataProvider(Protocol):
    def get_option_quote(self, contract: OptionContractSpec) -> OptionQuote:
        ...

    def get_similar_option_quotes(
        self,
        contract: OptionContractSpec,
        strike_span: int = 2,
    ) -> list[OptionQuote]:
        ...

    def get_put_call_quotes_same_strike(self, contract: OptionContractSpec) -> tuple[OptionQuote | None, OptionQuote | None]:
        ...


class OptionPricingEngine(ABC):
    def __init__(
        self,
        data_provider: OptionMarketDataProvider,
        risk_free_rate: float = 0.03,
        risk_free_rate_provider: Callable[[OptionContractSpec, float], float] | None = None,
        expiry_timezone: str = "America/New_York",
        expiry_close_hour: int = 16,
        expiry_close_minute: int = 0,
    ):
        self.data_provider = data_provider
        self.risk_free_rate = risk_free_rate
        self.risk_free_rate_provider = risk_free_rate_provider
        self.expiry_timezone = expiry_timezone
        self.expiry_close_hour = expiry_close_hour
        self.expiry_close_minute = expiry_close_minute

    def time_to_expiry_years(
        self,
        contract: OptionContractSpec,
        as_of: datetime | None = None,
    ) -> float:
        return _time_to_expiry_years(
            contract.expiry,
            now=as_of,
            market_tz=self.expiry_timezone,
            close_hour=self.expiry_close_hour,
            close_minute=self.expiry_close_minute,
        )

    def estimate_option_price(self, contract: OptionContractSpec) -> EngineQuote:
        base_quote = self.data_provider.get_option_quote(contract)
        neighbors = self.data_provider.get_similar_option_quotes(contract, strike_span=2)

        curve_price, curve_confidence, curve_meta = self._curve_based_price(contract, base_quote, neighbors)
        direct_price, direct_confidence = self._direct_price(base_quote)

        if curve_price is None and direct_price is None:
            raise ValueError(f"Could not estimate option price for {contract}")

        if curve_price is None:
            return EngineQuote(
                price=direct_price,
                source="direct_quote_only",
                confidence=direct_confidence,
                details={"direct_quote": base_quote},
            )

        if direct_price is None:
            return EngineQuote(
                price=curve_price,
                source="vol_curve_only",
                confidence=curve_confidence,
                details={"curve": curve_meta},
            )

        curve_weight = min(max(curve_confidence, 0.15), 0.7)
        direct_weight = 1.0 - curve_weight
        blended = (curve_price * curve_weight) + (direct_price * direct_weight)

        return EngineQuote(
            price=blended,
            source="blended_direct_plus_curve",
            confidence=max(direct_confidence, curve_confidence),
            details={
                "direct_price": direct_price,
                "curve_price": curve_price,
                "curve_weight": curve_weight,
                "direct_weight": direct_weight,
                "curve": curve_meta,
            },
        )

    def estimate_combo_price(self, legs: list[ComboLegSpec]) -> EngineQuote:
        if not legs:
            raise ValueError("Combo must include at least one leg.")

        leg_quotes: list[tuple[ComboLegSpec, EngineQuote]] = []
        total = 0.0
        confidence_parts = []
        for leg in legs:
            q = self.estimate_option_price(leg.contract)
            signed = leg.signed_ratio
            total += q.price * signed
            confidence_parts.append(q.confidence)
            leg_quotes.append((leg, q))

        return EngineQuote(
            price=total,
            source="sum_of_leg_estimates",
            confidence=sum(confidence_parts) / len(confidence_parts),
            details={"legs": leg_quotes},
        )

    def predict_option_change_for_stock_move(
        self,
        contract: OptionContractSpec,
        stock_price_change: float,
        use_bs_reanchor: bool = True,
    ) -> PredictedMove:
        quote = self.data_provider.get_option_quote(contract)
        base_price = self._must_have_price(quote)
        if use_bs_reanchor:
            bs_move = self._predict_option_change_bs_reanchored(quote, stock_price_change)
            if bs_move is not None:
                return bs_move
        delta = quote.delta if quote.delta is not None else self._finite_diff_delta(contract, quote)
        gamma = quote.gamma or 0.0

        d_option = (delta * stock_price_change) + (0.5 * gamma * (stock_price_change**2))
        confidence = 0.75 if quote.delta is not None else 0.55
        if quote.gamma is None:
            confidence -= 0.1

        return PredictedMove(
            option_price_change=d_option,
            stock_price_change=stock_price_change,
            estimated_new_option_price=max(0.0, base_price + d_option),
            confidence=max(0.1, min(confidence, 0.95)),
            details={
                "base_price": base_price,
                "delta": delta,
                "gamma": gamma,
                "method": "delta_gamma",
            },
        )

    def required_stock_move_for_option_change(
        self,
        contract: OptionContractSpec,
        option_price_change: float,
    ) -> PredictedMove:
        quote = self.data_provider.get_option_quote(contract)
        base_price = self._must_have_price(quote)
        delta = quote.delta if quote.delta is not None else self._finite_diff_delta(contract, quote)
        gamma = quote.gamma or 0.0

        if abs(gamma) < 1e-8:
            if abs(delta) < 1e-8:
                raise ValueError("Cannot infer required stock move with near-zero delta and gamma.")
            stock_change = option_price_change / delta
        else:
            # Solve 0.5*gamma*x^2 + delta*x - option_change = 0
            a = 0.5 * gamma
            b = delta
            c = -option_price_change
            disc = (b * b) - (4 * a * c)
            if disc < 0:
                raise ValueError("No real stock move solution for requested option move.")
            root = math.sqrt(disc)
            x1 = (-b + root) / (2 * a)
            x2 = (-b - root) / (2 * a)
            stock_change = x1 if abs(x1) < abs(x2) else x2

        return PredictedMove(
            option_price_change=option_price_change,
            stock_price_change=stock_change,
            estimated_new_option_price=max(0.0, base_price + option_price_change),
            confidence=0.7 if quote.delta is not None else 0.5,
            details={
                "base_price": base_price,
                "delta": delta,
                "gamma": gamma,
                "method": "inverted_delta_gamma",
            },
        )

    def estimate_risk_free_rate(self, contract: OptionContractSpec, quote: OptionQuote | None = None) -> float:
        if quote is None:
            quote = self.data_provider.get_option_quote(contract)
        t = _time_to_expiry_years(
            contract.expiry,
            now=quote.timestamp,
            market_tz=self.expiry_timezone,
            close_hour=self.expiry_close_hour,
            close_minute=self.expiry_close_minute,
        )
        base = self._resolve_risk_free_rate(contract, t)
        implied = self._implied_rate_from_put_call_parity(contract, quote, t)
        if implied is None:
            return base
        # Blend base curve/provider with market-implied short-dated parity rate.
        blend_weight = 0.65 if t < (21.0 / 365.0) else 0.45
        return (blend_weight * implied) + ((1.0 - blend_weight) * base)

    def predict_combo_change_for_stock_move(
        self,
        legs: list[ComboLegSpec],
        stock_price_change: float,
    ) -> PredictedMove:
        if not legs:
            raise ValueError("Combo must include at least one leg.")

        total_change = 0.0
        base_combo = self.estimate_combo_price(legs).price
        confidences = []
        parts = []
        for leg in legs:
            leg_move = self.predict_option_change_for_stock_move(leg.contract, stock_price_change)
            signed = leg.signed_ratio
            leg_change = leg_move.option_price_change * signed
            total_change += leg_change
            confidences.append(leg_move.confidence)
            parts.append((leg, leg_move))

        return PredictedMove(
            option_price_change=total_change,
            stock_price_change=stock_price_change,
            estimated_new_option_price=max(0.0, base_combo + total_change),
            confidence=sum(confidences) / len(confidences),
            details={"legs": parts, "base_combo": base_combo},
        )

    @abstractmethod
    def _direct_price(self, quote: OptionQuote) -> tuple[float | None, float]:
        raise NotImplementedError

    def _curve_based_price(
        self,
        contract: OptionContractSpec,
        base_quote: OptionQuote,
        neighbors: list[OptionQuote],
    ) -> tuple[float | None, float, dict]:
        underlier = base_quote.underlier_price
        t = _time_to_expiry_years(
            contract.expiry,
            now=base_quote.timestamp,
            market_tz=self.expiry_timezone,
            close_hour=self.expiry_close_hour,
            close_minute=self.expiry_close_minute,
        )
        if underlier is None or t <= 0:
            return None, 0.0, {"reason": "missing_underlier_or_expiry"}

        valid = [q for q in neighbors if q.implied_vol is not None]
        if base_quote.implied_vol is not None:
            valid.append(base_quote)
        if len(valid) < 3:
            return None, 0.0, {"reason": "insufficient_iv_neighbors"}

        points = [(q.contract.strike, q.implied_vol) for q in valid if q.implied_vol is not None]
        points.sort(key=lambda x: x[0])
        fitted_iv = _fit_local_smile(contract.strike, points)
        if fitted_iv is None:
            return None, 0.0, {"reason": "smile_fit_failed"}

        rf = self._resolve_risk_free_rate(contract, t)
        theo = _black_scholes_price(
            underlier,
            contract.strike,
            t,
            rf,
            fitted_iv,
            contract.right.upper(),
        )
        return max(theo, 0.0), 0.65, {"fitted_iv": fitted_iv, "iv_points": points, "risk_free_rate": rf, "time_to_expiry_years": t}

    def _must_have_price(self, quote: OptionQuote) -> float:
        direct, _ = self._direct_price(quote)
        if direct is None:
            curve, _, _ = self._curve_based_price(quote.contract, quote, [])
            if curve is None:
                raise ValueError("Could not infer base option price.")
            return curve
        return direct

    def _finite_diff_delta(self, contract: OptionContractSpec, quote: OptionQuote) -> float:
        underlier = quote.underlier_price
        iv = quote.implied_vol
        if underlier is None or iv is None:
            return 0.0
        t = _time_to_expiry_years(
            contract.expiry,
            now=quote.timestamp,
            market_tz=self.expiry_timezone,
            close_hour=self.expiry_close_hour,
            close_minute=self.expiry_close_minute,
        )
        if t <= 0:
            return 0.0
        rf = self._resolve_risk_free_rate(contract, t)

        bump = max(underlier * 0.005, 0.05)
        p_up = _black_scholes_price(
            underlier + bump,
            contract.strike,
            t,
            rf,
            iv,
            contract.right.upper(),
        )
        p_dn = _black_scholes_price(
            underlier - bump,
            contract.strike,
            t,
            rf,
            iv,
            contract.right.upper(),
        )
        return (p_up - p_dn) / (2 * bump)

    def _resolve_risk_free_rate(self, contract: OptionContractSpec, t_years: float) -> float:
        if self.risk_free_rate_provider is not None:
            try:
                rate = float(self.risk_free_rate_provider(contract, t_years))
                if math.isfinite(rate):
                    return rate
            except Exception:
                pass
        return self.risk_free_rate

    def _predict_option_change_bs_reanchored(
        self,
        quote: OptionQuote,
        stock_price_change: float,
    ) -> PredictedMove | None:
        underlier = quote.underlier_price
        if underlier is None or quote.implied_vol is None:
            return None
        base_price = self._must_have_price(quote)
        t = _time_to_expiry_years(
            quote.contract.expiry,
            now=quote.timestamp,
            market_tz=self.expiry_timezone,
            close_hour=self.expiry_close_hour,
            close_minute=self.expiry_close_minute,
        )
        if t <= 0:
            return None
        rf = self.estimate_risk_free_rate(quote.contract, quote)
        s_old = underlier
        s_new = max(0.01, underlier + stock_price_change)
        bs_old = _black_scholes_price(
            s_old,
            quote.contract.strike,
            t,
            rf,
            quote.implied_vol,
            quote.contract.right.upper(),
        )
        bs_new = _black_scholes_price(
            s_new,
            quote.contract.strike,
            t,
            rf,
            quote.implied_vol,
            quote.contract.right.upper(),
        )
        d_option = bs_new - bs_old
        anchored_new = max(0.0, base_price + d_option)
        return PredictedMove(
            option_price_change=d_option,
            stock_price_change=stock_price_change,
            estimated_new_option_price=anchored_new,
            confidence=0.82,
            details={
                "method": "bs_reanchored",
                "base_market_price": base_price,
                "bs_old": bs_old,
                "bs_new": bs_new,
                "implied_vol": quote.implied_vol,
                "risk_free_rate": rf,
                "time_to_expiry_years": t,
            },
        )

    def _implied_rate_from_put_call_parity(
        self,
        contract: OptionContractSpec,
        quote: OptionQuote,
        t_years: float,
    ) -> float | None:
        if t_years <= 0 or quote.underlier_price is None:
            return None
        get_pair = getattr(self.data_provider, "get_put_call_quotes_same_strike", None)
        if not callable(get_pair):
            return None
        try:
            call_q, put_q = get_pair(contract)
        except Exception:
            return None
        if call_q is None or put_q is None:
            return None

        call_px, _ = self._direct_price(call_q)
        put_px, _ = self._direct_price(put_q)
        if call_px is None or put_px is None:
            return None

        # C - P = S - K*exp(-rT)  (ignoring dividends/carry)
        s = quote.underlier_price
        k = contract.strike
        discounted_strike = s - (call_px - put_px)
        if discounted_strike <= 0:
            return None
        try:
            r = -math.log(discounted_strike / k) / t_years
        except ValueError:
            return None
        if not math.isfinite(r):
            return None
        # Robust clipping for short-dated options where parity noise can explode.
        return min(max(r, -0.02), 0.15)


class DefaultOptionPricingEngine(OptionPricingEngine):
    def _direct_price(self, quote: OptionQuote) -> tuple[float | None, float]:
        if quote.bid is not None and quote.ask is not None and quote.bid <= quote.ask:
            mid = (quote.bid + quote.ask) / 2.0
            spread = max(quote.ask - quote.bid, 0.0)
            spread_ratio = spread / max(mid, 0.01)
            confidence = 0.9 if spread_ratio < 0.05 else 0.75 if spread_ratio < 0.15 else 0.55
            return mid, confidence

        if quote.midpoint is not None:
            return quote.midpoint, 0.65
        if quote.model_price is not None:
            return quote.model_price, 0.6
        if quote.last is not None:
            return quote.last, 0.45
        return None, 0.0


def _time_to_expiry_years(
    expiry: str,
    now: datetime | None = None,
    market_tz: str = "America/New_York",
    close_hour: int = 16,
    close_minute: int = 0,
) -> float:
    tz = ZoneInfo(market_tz)
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    local_now = now.astimezone(tz)

    expiry_date = datetime.strptime(expiry, "%Y%m%d").date()
    expiry_local = datetime(
        year=expiry_date.year,
        month=expiry_date.month,
        day=expiry_date.day,
        hour=close_hour,
        minute=close_minute,
        tzinfo=tz,
    )
    expiry_utc = expiry_local.astimezone(timezone.utc)
    seconds = (expiry_utc - local_now.astimezone(timezone.utc)).total_seconds()
    return max(seconds / (365.0 * 24 * 3600), 0.0)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _black_scholes_price(
    spot: float,
    strike: float,
    t: float,
    r: float,
    sigma: float,
    right: str,
) -> float:
    if spot <= 0 or strike <= 0 or t <= 0 or sigma <= 0:
        intrinsic = max(spot - strike, 0.0) if right == "C" else max(strike - spot, 0.0)
        return intrinsic

    vol_sqrt_t = sigma * math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma * sigma) * t) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    if right == "C":
        return (spot * _norm_cdf(d1)) - (strike * math.exp(-r * t) * _norm_cdf(d2))
    return (strike * math.exp(-r * t) * _norm_cdf(-d2)) - (spot * _norm_cdf(-d1))


def _fit_local_smile(target_strike: float, points: list[tuple[float, float]]) -> float | None:
    if len(points) < 3:
        return None

    points = sorted(points, key=lambda x: abs(x[0] - target_strike))
    p = points[:5]
    n = len(p)
    # Quadratic least squares in strike space: iv = a + bK + cK^2
    s0 = float(n)
    s1 = sum(x for x, _ in p)
    s2 = sum((x * x) for x, _ in p)
    s3 = sum((x * x * x) for x, _ in p)
    s4 = sum((x * x * x * x) for x, _ in p)
    t0 = sum(y for _, y in p)
    t1 = sum(x * y for x, y in p)
    t2 = sum((x * x * y) for x, y in p)

    det = (
        s0 * (s2 * s4 - s3 * s3)
        - s1 * (s1 * s4 - s3 * s2)
        + s2 * (s1 * s3 - s2 * s2)
    )
    if abs(det) < 1e-12:
        # Fallback to weighted average by strike distance
        weights = [1.0 / (abs(x - target_strike) + 0.5) for x, _ in p]
        w_sum = sum(weights)
        if w_sum <= 0:
            return None
        return sum(w * y for w, (_, y) in zip(weights, p)) / w_sum

    def det3(a11, a12, a13, a21, a22, a23, a31, a32, a33):
        return (a11 * (a22 * a33 - a23 * a32)) - (a12 * (a21 * a33 - a23 * a31)) + (a13 * (a21 * a32 - a22 * a31))

    det_a = det3(t0, s1, s2, t1, s2, s3, t2, s3, s4)
    det_b = det3(s0, t0, s2, s1, t1, s3, s2, t2, s4)
    det_c = det3(s0, s1, t0, s1, s2, t1, s2, s3, t2)

    a = det_a / det
    b = det_b / det
    c = det_c / det
    iv = a + (b * target_strike) + (c * target_strike * target_strike)
    return max(iv, 0.01)

