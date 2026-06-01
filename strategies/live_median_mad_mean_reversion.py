import pandas as pd
from collections import deque
from features.feature_calculator import LiveMedianMadFeatureCalculator
from model_filters.always_pass_model_filter import AlwaysPassModelFilter
from orders import LimitOrderSpec, OrderSide
from strategies.open_trades_store import OpenTradesStore


class LiveMedianMadMeanReversion:
    def __init__(
        self,
        config,
        order_router,
        feature_calculator=None,
        model_filter=None,
        execution_instrument=None,
    ):
        self.config = config
        self.raw_config = config.raw

        self.symbol = config.symbol
        self.order_router = order_router
        self.execution_instrument = execution_instrument

        paths_cfg = config.paths
        features_cfg = config.features
        strategy_cfg = config.strategy
        double_down_cfg = config.double_down
        sizing_cfg = config.sizing
        execution_cfg = config.execution
        saving_cfg = config.saving
        debug_cfg = config.debug

        # ----------------------------
        # Feature config
        # ----------------------------
        self.minute_window = features_cfg.get("minute_window", 390)

        self.feature_calculator = feature_calculator or LiveMedianMadFeatureCalculator(
            window=features_cfg.get("window", 20),
            min_bars=features_cfg.get("min_bars", 60),
        )

        # ----------------------------
        # Model filter
        # Default: always pass with prob=1.0
        # ----------------------------
        self.model_filter = model_filter or AlwaysPassModelFilter()

        # ----------------------------
        # Strategy config
        # ----------------------------
        self.k = strategy_cfg.get("k", 2)

        self.max_new_trades_per_day = strategy_cfg.get("max_new_trades_per_day", 6)
        self.max_open_trades_total = strategy_cfg.get("max_open_trades_total", 200)
        self.max_open_trades_today = strategy_cfg.get("max_open_trades_today", 2)

        self.no_trade_first_minutes = strategy_cfg.get("no_trade_first_minutes", 60)
        self.no_new_entries_last_minutes = strategy_cfg.get("no_new_entries_last_minutes", 60)

        self.min_take_profit = strategy_cfg.get("min_take_profit", 0.01)
        self.fuckup_take_profit = strategy_cfg.get("fuckup_take_profit", 0.10)

        self.RSI_THRESH = strategy_cfg.get("rsi_thresh", 300)

        self.reentry_cooldown_days = strategy_cfg.get("reentry_cooldown_days", 7)
        self.reentry_discount_pct = strategy_cfg.get("reentry_discount_pct", 0.01)

        self.limit_entry_offset_pct = strategy_cfg.get("limit_entry_offset_pct", 0.0005)

        # ----------------------------
        # Double-down config
        # ----------------------------
        self.double_down_enabled = double_down_cfg.get("enabled", True)
        self.double_down_bb_mult = double_down_cfg.get("bb_mult", 3)
        self.double_down_discount_pct = double_down_cfg.get("discount_pct", 0.01)

        # ----------------------------
        # Sizing config
        # ----------------------------
        self.DEFAULT_QTY = sizing_cfg.get("default_qty", 100)

        raw_size_schedule = sizing_cfg.get("size_schedule", {})
        self.size_schedule = {
            int(k): int(v)
            for k, v in raw_size_schedule.items()
        }

        # ----------------------------
        # Execution config
        # ----------------------------
        self.order_tif = execution_cfg.get("tif", "DAY")
        self.outside_rth = execution_cfg.get("outside_rth", True)

        # ----------------------------
        # Save/load config
        # ----------------------------
        self.open_trades_path = paths_cfg.get(
            "open_trades_path",
            f"{self.symbol}_open_trades.csv",
        )

        self.autosave_open_trades = saving_cfg.get("autosave_open_trades", True)

        # ----------------------------
        # Debug config
        # ----------------------------
        self.DEBUG_BARS = debug_cfg.get("bars", True)
        self.DEBUG_FEATURES = debug_cfg.get("features", True)
        self.DEBUG_SIGNAL = debug_cfg.get("signal", True)
        self.DEBUG_ORDERS = debug_cfg.get("orders", True)
        self.DEBUG_SAVE = debug_cfg.get("save", True)
        self.DEBUG_MODEL = debug_cfg.get("model", True)

        # ----------------------------
        # Internal state
        # ----------------------------
        self.bootstrapped = False

        self.five_sec_buffer = deque(maxlen=12)
        self.minute_bars = deque(maxlen=self.minute_window)

        self.trades_today = 0
        self.current_day = None
        self.max_active_trades_seen = 0

        self.open_trades = []
        self.open_trades_today = 0

        self.last_sell_price = None
        self.last_sell_date = None
        self.last_signal = None

        self.open_trades_store = OpenTradesStore(
            path=self.open_trades_path,
            symbol=self.symbol,
            debug_save=self.DEBUG_SAVE,
            logger=self.log,
        )

        self.open_trades = self.open_trades_store.load()

    def log(self, *args):
        print(*args)

    # ========================================================
    # Main callback
    # ========================================================

    def on_bar(self, bars, hasNewBar):
        if not hasNewBar:
            return

        new_minute_bar = self.update_bars(bars)

        if new_minute_bar is None:
            return

        features = self.calculate_features()

        if features is None:
            return

        self.update_daily_state(features)
        self.manage_open_trades(features)

        signal = self.should_enter_trade(features)

        if signal is not None:
            self.place_order(signal)

    # ========================================================
    # Bootstrap + live bar aggregation
    # ========================================================

    def bootstrap_from_existing_bars(self, bars):
        df = pd.DataFrame([
            {
                "time": b.time,
                "open": b.open_,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": max(b.volume, 0),
            }
            for b in bars
        ])

        df = df.set_index("time").sort_index()

        df1 = (
            df.resample("1min")
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })
            .dropna()
        )

        self.minute_bars.clear()

        for row in df1.tail(self.minute_bars.maxlen).itertuples():
            self.minute_bars.append({
                "time": row.Index,
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "volume": row.volume,
            })

        self.five_sec_buffer.clear()

        for b in bars[-12:]:
            self.five_sec_buffer.append(b)

        self.bootstrapped = True

        if self.DEBUG_BARS:
            self.log(
                f"[BOOTSTRAP] loaded {len(self.minute_bars)} minute bars | "
                f"5s buffer={len(self.five_sec_buffer)}"
            )

    def update_bars(self, bars):
        if not self.bootstrapped:
            self.bootstrap_from_existing_bars(bars)
            return None

        latest = bars[-1]
        self.five_sec_buffer.append(latest)

        if self.DEBUG_BARS:
            self.log(
                f"[5S BAR] {latest.time} "
                f"O={latest.open_} H={latest.high} L={latest.low} C={latest.close} "
                f"buffer={len(self.five_sec_buffer)}"
            )

        if latest.time.second != 55:
            return None

        if len(self.five_sec_buffer) < 12:
            return None

        chunk = self.five_sec_buffer

        minute_bar = {
            "time": latest.time.replace(second=0, microsecond=0),
            "open": chunk[0].open_,
            "high": max(b.high for b in chunk),
            "low": min(b.low for b in chunk),
            "close": chunk[-1].close,
            "volume": sum(max(b.volume, 0) for b in chunk),
        }

        self.minute_bars.append(minute_bar)

        if self.DEBUG_BARS:
            self.log(
                f"[NEW MINUTE BAR] {minute_bar['time']} "
                f"O={minute_bar['open']} H={minute_bar['high']} "
                f"L={minute_bar['low']} C={minute_bar['close']} "
                f"V={minute_bar['volume']} | stored={len(self.minute_bars)}"
            )

        return minute_bar

    # ========================================================
    # Features + model score
    # ========================================================

    def calculate_features(self):
        features = self.feature_calculator.calculate(self.minute_bars)

        if features is None:
            return None

        model_result = self.model_filter.score(features)

        features["ml_enabled"] = model_result["enabled"]
        features["ml_prob"] = model_result["prob"]
        features["ml_passed"] = model_result["passed"]
        features["ml_missing_features"] = model_result["missing_features"]

        if self.DEBUG_MODEL:
            ml_prob = features["ml_prob"]
            ml_text = "None" if ml_prob is None else f"{ml_prob:.4f}"

            self.log(
                f"[MODEL] {features['time']} "
                f"enabled={features['ml_enabled']} "
                f"prob={ml_text} "
                f"passed={features['ml_passed']} "
                f"missing={features['ml_missing_features']}"
            )

        if self.DEBUG_FEATURES:
            rsi = features["rsi"]
            rsi_text = "None" if pd.isna(rsi) else f"{rsi:.2f}"

            ml_prob = features["ml_prob"]
            ml_text = "None" if ml_prob is None else f"{ml_prob:.4f}"

            self.log(
                f"[FEATURES] {features['time']} "
                f"close={features['close']:.2f} "
                f"median={features['median']:.2f} "
                f"mad={features['mad']:.4f} "
                f"bb_score={features['bb_score']:.2f} "
                f"rsi={rsi_text} "
                f"ml_prob={ml_text} "
                f"session_min={features['session_minutes']} "
                f"mins_to_close={features['minutes_until_close']}"
            )

        return features

    # ========================================================
    # Daily state 
    # ========================================================

    def update_daily_state(self, features):
        today = features["date"]

        if self.current_day != today:
            self.current_day = today
            self.trades_today = 0
            self.open_trades_today = 0

            if self.DEBUG_BARS:
                self.log(f"[NEW DAY] {today}")

        self.max_active_trades_seen = max(
            self.max_active_trades_seen,
            len(self.open_trades),
        )

    # ========================================================
    # Entry logic
    #
    # First entry:
    #   bb_score <= -k
    #
    # Double down:
    #   bb_score <= -(double_down_bb_mult * k)
    #   close <= previous_entry_price * (1 - double_down_discount_pct)
    #
    # Model:
    #   default AlwaysPassModelFilter means ml_prob=1.0 and ml_passed=True
    #   CatBoost filter can replace it.
    # ========================================================

    def should_enter_trade(self, features):
        close = features["close"]
        today = features["date"]

        reasons = []

        if self.trades_today >= self.max_new_trades_per_day:
            reasons.append("max_new_trades_per_day")

        if len(self.open_trades) >= self.max_open_trades_total:
            reasons.append("max_open_trades_total")

        if self.open_trades_today >= self.max_open_trades_today:
            reasons.append("max_open_trades_today")

        if features["session_minutes"] < self.no_trade_first_minutes:
            reasons.append("too_early")

        if features["minutes_until_close"] < self.no_new_entries_last_minutes:
            reasons.append("too_late")

        if features["rsi"] is None or pd.isna(features["rsi"]):
            reasons.append("rsi_missing")
        elif features["rsi"] > self.RSI_THRESH:
            reasons.append("rsi_too_high")

        # ====================================================
        # CatBoost / model filter
        # Default always passes.
        # ====================================================

        if features.get("ml_prob") is None:
            reasons.append("ml_prob_missing")
        elif not features.get("ml_passed", False):
            reasons.append("ml_prob_too_low")

        # ====================================================
        # First entry vs double-down logic
        # ====================================================

        if not self.open_trades:
            entry_type = "first_entry"

            if features["bb_score"] > -self.k:
                reasons.append("bb_score_not_low_enough_for_first_entry")

        else:
            entry_type = "double_down"

            if not self.double_down_enabled:
                reasons.append("double_down_disabled")

            double_down_threshold = -(self.double_down_bb_mult * self.k)

            if features["bb_score"] > double_down_threshold:
                reasons.append("bb_score_not_low_enough_for_double_down")

            previous_entry_price = self.open_trades[-1]["entry_price"]
            required_price = previous_entry_price * (1 - self.double_down_discount_pct)

            if close > required_price:
                reasons.append("not_discounted_enough_from_previous_entry")

        # ====================================================
        # Reentry cooldown after sells
        # Existing logic kept.
        # ====================================================

        if self.last_sell_price is not None and self.last_sell_date is not None:
            days_since_sell = (today - self.last_sell_date).days

            if days_since_sell < self.reentry_cooldown_days:
                required_price = self.last_sell_price * (1 - self.reentry_discount_pct)

                if close > required_price:
                    reasons.append("reentry_cooldown")

        if reasons:
            if self.DEBUG_SIGNAL:
                rsi_text = "None" if pd.isna(features["rsi"]) else f"{features['rsi']:.2f}"

                ml_prob = features.get("ml_prob")
                ml_text = "None" if ml_prob is None else f"{ml_prob:.4f}"

                if self.open_trades:
                    previous_entry_price = self.open_trades[-1]["entry_price"]
                    required_price = previous_entry_price * (1 - self.double_down_discount_pct)
                    double_down_threshold = -(self.double_down_bb_mult * self.k)

                    self.log(
                        f"[NO ENTRY] {features['time']} "
                        f"type=double_down "
                        f"close={close:.2f} "
                        f"bb={features['bb_score']:.2f} "
                        f"needed_bb<={double_down_threshold:.2f} "
                        f"prev_entry={previous_entry_price:.2f} "
                        f"needed_close<={required_price:.2f} "
                        f"rsi={rsi_text} "
                        f"ml_prob={ml_text} "
                        f"reasons={reasons}"
                    )
                else:
                    self.log(
                        f"[NO ENTRY] {features['time']} "
                        f"type=first_entry "
                        f"close={close:.2f} "
                        f"bb={features['bb_score']:.2f} "
                        f"needed_bb<={-self.k:.2f} "
                        f"rsi={rsi_text} "
                        f"ml_prob={ml_text} "
                        f"reasons={reasons}"
                    )

            return None

        qty = self.calculate_order_qty()

        if qty <= 0:
            return None

        limit_price = close * (1 - self.limit_entry_offset_pct)

        signal = {
            "side": "BUY",
            "qty": qty,
            "limit_price": round(limit_price, 2),
            "underlying_reference_price": close,
            "features": features,
            "reason": entry_type,
            "ml_prob": features.get("ml_prob"),
        }

        if self.DEBUG_SIGNAL:
            self.log(f"[ENTRY SIGNAL] {signal}")

        return signal

    def calculate_order_qty(self):
        active_n = len(self.open_trades)

        qty = self.size_schedule.get(
            active_n,
            self.DEFAULT_QTY,
        )

        if self.DEBUG_SIGNAL:
            self.log(f"[SIZE] active_n={active_n} qty={qty}")

        return qty

    # ========================================================
    # Order placement
    # ========================================================

    def place_order(self, signal):
        underlying_ref = signal.get("underlying_reference_price", signal["features"]["close"])
        if self.execution_instrument is not None:
            trade_contract = self.execution_instrument.resolve_contract()
            execution_limit = self.execution_instrument.estimate_entry_limit_price(underlying_ref)
        else:
            trade_contract = self.order_router.contract
            execution_limit = signal["limit_price"]

        order_spec = LimitOrderSpec(
            contract=trade_contract,
            side=OrderSide.BUY,
            qty=signal["qty"],
            limit_price=execution_limit,
            tif=self.order_tif,
            outside_rth=self.outside_rth,
        )

        trade = self.order_router.submit(order_spec)

        self.open_trades.append({
            "trade": trade,
            "qty": signal["qty"],
            "entry_price": execution_limit,
            "entry_underlying_price": underlying_ref,
            "entry_date": signal["features"]["date"],
            "entry_time": signal["features"]["time"],
            "entry_reason": signal["reason"],
            "ml_prob": signal.get("ml_prob"),
            "instrument_type": (
                self.execution_instrument.instrument_type
                if self.execution_instrument is not None
                else "stock"
            ),
        })

        self.trades_today += 1
        self.open_trades_today += 1
        self.last_signal = signal

        if self.autosave_open_trades:
            self.open_trades_store.save(self.open_trades)

        if self.DEBUG_ORDERS:
            self.log(
                f"[BUY SENT] {self.symbol} "
                f"reason={signal['reason']} "
                f"qty={signal['qty']} "
                f"limit={execution_limit} "
                f"underlying_ref={underlying_ref:.2f} "
                f"instrument={self.open_trades[-1]['instrument_type']} "
                f"ml_prob={signal.get('ml_prob')} "
                f"trades_today={self.trades_today} "
                f"open_trades={len(self.open_trades)}"
            )

        return trade

    # ========================================================
    # Exit logic
    # Existing logic kept.
    #
    # This manages ALL open trades:
    #   - trades opened today
    #   - trades loaded from previous days
    # ========================================================

    def manage_open_trades(self, features):
        close = features["close"]
        today = features["date"]

        still_open = []

        for t in self.open_trades:
            entry_underlying = t.get("entry_underlying_price", t["entry_price"])
            take_profit = entry_underlying * (1 + self.min_take_profit)

            if close >= take_profit:
                self.close_trade(t, close, today)
            else:
                still_open.append(t)

        self.open_trades = still_open

        if self.autosave_open_trades:
            self.open_trades_store.save(self.open_trades)

    def close_trade(self, trade_info, close_price, today):
        if self.execution_instrument is not None:
            trade_contract = self.execution_instrument.resolve_contract()
            exit_limit = self.execution_instrument.estimate_exit_limit_price(close_price)
        else:
            trade_contract = self.order_router.contract
            exit_limit = round(close_price, 2)

        order_spec = LimitOrderSpec(
            contract=trade_contract,
            side=OrderSide.SELL,
            qty=trade_info["qty"],
            limit_price=exit_limit,
            tif=self.order_tif,
            outside_rth=self.outside_rth,
        )

        trade = self.order_router.submit(order_spec)

        self.last_sell_price = close_price
        self.last_sell_date = today

        if self.DEBUG_ORDERS:
            self.log(
                f"[SELL SENT] {self.symbol} qty={trade_info['qty']} "
                f"limit={exit_limit} "
                f"underlying_ref={close_price:.2f} "
                f"instrument={trade_info.get('instrument_type', 'stock')} "
                f"last_sell_date={today}"
            )

        return trade