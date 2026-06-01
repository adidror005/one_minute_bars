import pandas as pd
from features.bar_builders import build_signal_bar_builder
from features.feature_calculator import LiveMedianMadFeatureCalculator
from model_filters.always_pass_model_filter import AlwaysPassModelFilter
from orders import LimitOrderSpec, OrderSide
from strategies.entry_conditions import EntryConditionEvaluator
from strategies.open_trades_store import OpenTradesStore


class LiveMeanReversion:
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
        self.signal_bar_builder = build_signal_bar_builder(
            features_cfg=features_cfg,
            logger=self.log,
            debug=debug_cfg.get("bars", True),
        )
        self.signal_bars = self.signal_bar_builder.bars
        self.minute_bars = self.signal_bars

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
        self.entry_condition_evaluator = EntryConditionEvaluator.from_config(
            strategy_cfg=strategy_cfg,
            double_down_cfg=double_down_cfg,
        )
        self.entry_price_limits = strategy_cfg.get("entry_price_limits", {})
        self.latest_trade_price_rules = strategy_cfg.get("latest_trade_price_rules", {})

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
        self.double_down_price_rules = self._build_double_down_price_rules(double_down_cfg)

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
        self.trades_today = 0
        self.current_day = None
        self.max_active_trades_seen = 0

        self.open_trades = []
        self.open_trades_today = 0

        self.last_sell_price = None
        self.last_sell_date = None
        self.last_sell_time = None
        self.last_sell_underlying_price = None
        self.last_sell_execution_price = None
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

    def _build_double_down_price_rules(self, double_down_cfg):
        rules = double_down_cfg.get("price_rules")

        if rules is not None:
            return list(rules)

        return [
            {
                "basis": "underlying",
                "mode": "pct",
                "max_change": -float(self.double_down_discount_pct),
            }
        ]

    # ========================================================
    # Main callback
    # ========================================================

    def on_bar(self, bars, hasNewBar):
        if not hasNewBar:
            return

        new_signal_bar = self.update_bars(bars)

        if new_signal_bar is None:
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
    # Signal bar aggregation
    # ========================================================

    def bootstrap_from_existing_bars(self, bars):
        self.signal_bar_builder.bootstrap(bars)

    def update_bars(self, bars):
        return self.signal_bar_builder.update(bars)

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
            entry_condition_reasons, entry_condition_debug = (
                self.entry_condition_evaluator.evaluate(entry_type, features)
            )
            reasons.extend(entry_condition_reasons)

        else:
            entry_type = "double_down"

            if not self.double_down_enabled:
                reasons.append("double_down_disabled")

            entry_condition_reasons, entry_condition_debug = (
                self.entry_condition_evaluator.evaluate(entry_type, features)
            )
            reasons.extend(entry_condition_reasons)

            if self.double_down_enabled and not entry_condition_reasons:
                double_down_reasons, double_down_debug = self.check_double_down_price_rules(features)
                reasons.extend(double_down_reasons)
            else:
                double_down_debug = []

        # ====================================================
        # Hard price limits
        # ====================================================

        price_limit_debug = []
        if not reasons:
            price_limit_reasons, price_limit_debug = self.check_entry_price_limits(
                entry_type=entry_type,
                features=features,
            )
            reasons.extend(price_limit_reasons)

        latest_trade_debug = []
        if not reasons:
            latest_trade_reasons, latest_trade_debug = self.check_latest_trade_price_rules(
                entry_type=entry_type,
                features=features,
            )
            reasons.extend(latest_trade_reasons)

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
                    entry_condition_text = self.entry_condition_evaluator.format_debug(
                        entry_condition_debug
                    )
                    price_rule_text = self.format_double_down_debug(double_down_debug)
                    price_limit_text = self.format_entry_price_limit_debug(price_limit_debug)
                    latest_trade_text = self.format_latest_trade_price_debug(latest_trade_debug)

                    self.log(
                        f"[NO ENTRY] {features['time']} "
                        f"type=double_down "
                        f"close={close:.2f} "
                        f"entry_conditions={entry_condition_text} "
                        f"price_limits={price_limit_text} "
                        f"latest_trade={latest_trade_text} "
                        f"price_rules={price_rule_text} "
                        f"rsi={rsi_text} "
                        f"ml_prob={ml_text} "
                        f"reasons={reasons}"
                    )
                else:
                    self.log(
                        f"[NO ENTRY] {features['time']} "
                        f"type=first_entry "
                        f"close={close:.2f} "
                        f"entry_conditions={self.entry_condition_evaluator.format_debug(entry_condition_debug)} "
                        f"price_limits={self.format_entry_price_limit_debug(price_limit_debug)} "
                        f"latest_trade={self.format_latest_trade_price_debug(latest_trade_debug)} "
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

    def check_entry_price_limits(self, entry_type, features):
        limits = self.entry_price_limits.get(entry_type, {})
        if not limits:
            return [], []

        reasons = []
        debug = []
        stock_price = features["close"]
        buy_price = None

        for limit_name, limit_value in limits.items():
            if limit_value is None:
                continue

            if limit_name in ("max_stock_price", "min_stock_price"):
                price = stock_price
                price_basis = "stock"
            elif limit_name in ("max_buy_price", "min_buy_price"):
                if buy_price is None:
                    buy_price = self.estimate_entry_buy_price(features)
                price = buy_price
                price_basis = "buy"
            else:
                reasons.append(f"unsupported_entry_price_limit_{limit_name}")
                debug.append({
                    "limit": limit_name,
                    "basis": None,
                    "price": None,
                    "limit_value": limit_value,
                    "passed": False,
                })
                continue

            passed = self.entry_price_limit_passed(limit_name, price, float(limit_value))
            debug.append({
                "limit": limit_name,
                "basis": price_basis,
                "price": price,
                "limit_value": float(limit_value),
                "passed": passed,
            })

            if not passed:
                reasons.append(f"{entry_type}_{limit_name}_blocked")

        return reasons, debug

    def estimate_entry_buy_price(self, features):
        close = features["close"]

        if self.execution_instrument is not None:
            return self.execution_instrument.estimate_entry_limit_price(close)

        return round(close * (1 - self.limit_entry_offset_pct), 2)

    def entry_price_limit_passed(self, limit_name, price, limit_value):
        if limit_name.startswith("max_"):
            return price <= limit_value

        if limit_name.startswith("min_"):
            return price >= limit_value

        return False

    def format_entry_price_limit_debug(self, debug):
        if not debug:
            return "[]"

        parts = []
        for item in debug:
            price = item["price"]
            price_text = "None" if price is None else f"{price:.2f}"
            parts.append(
                f"{item['limit']} {item['basis']}={price_text} "
                f"limit={item['limit_value']} passed={item['passed']}"
            )

        return "[" + "; ".join(parts) + "]"

    def check_latest_trade_price_rules(self, entry_type, features):
        cfg = self.latest_trade_price_rules
        if not cfg or not cfg.get("enabled", False):
            return [], []

        rules = cfg.get(entry_type, [])
        if not rules:
            return [], []

        previous_event = self.latest_trade_price_rule_previous_event(entry_type)
        if previous_event is None:
            return [], []

        reasons = []
        debug = []

        for rule in rules:
            basis = rule.get("basis", "underlying")
            mode = rule.get("mode", "pct")
            current_price = self.current_latest_trade_rule_price(basis, features)
            previous_price = self.previous_latest_trade_rule_price(basis, previous_event)
            max_change = self.rule_max_change(rule)

            if current_price is None or previous_price is None:
                reasons.append(f"latest_trade_{basis}_price_missing")
                debug.append({
                    "basis": basis,
                    "mode": mode,
                    "previous_event": previous_event["kind"],
                    "current": current_price,
                    "previous": previous_price,
                    "change": None,
                    "max_change": max_change,
                    "passed": False,
                })
                continue

            change = current_price - previous_price

            if mode == "pct":
                if previous_price == 0:
                    reasons.append(f"latest_trade_{basis}_previous_price_zero")
                    passed = False
                    change_value = None
                else:
                    change_value = change / previous_price
                    passed = change_value <= max_change
            elif mode == "abs":
                change_value = change
                passed = change_value <= max_change
            else:
                reasons.append(f"latest_trade_{basis}_unsupported_mode")
                passed = False
                change_value = None

            debug.append({
                "basis": basis,
                "mode": mode,
                "previous_event": previous_event["kind"],
                "current": current_price,
                "previous": previous_price,
                "change": change_value,
                "max_change": max_change,
                "passed": passed,
            })

            if not passed:
                reasons.append(f"latest_trade_{basis}_{mode}_change_not_low_enough")

        return reasons, debug

    def latest_trade_price_rule_previous_event(self, entry_type):
        if entry_type == "double_down":
            return self.latest_buy_event()

        return self.latest_buy_or_sell_event()

    def latest_buy_or_sell_event(self):
        latest_buy = self.latest_buy_event()
        latest_sell = self.latest_sell_event()

        if latest_buy is None:
            return latest_sell

        if latest_sell is None:
            return latest_buy

        if pd.to_datetime(latest_sell["time"]) > pd.to_datetime(latest_buy["time"]):
            return latest_sell

        return latest_buy

    def latest_buy_event(self):
        if not self.open_trades:
            return None

        latest_trade = max(
            self.open_trades,
            key=lambda trade: pd.to_datetime(trade.get("entry_time")),
        )

        return {
            "kind": "buy",
            "time": latest_trade.get("entry_time"),
            "underlying_price": latest_trade.get(
                "entry_underlying_price",
                latest_trade.get("entry_price"),
            ),
            "execution_price": latest_trade.get("entry_price"),
        }

    def latest_sell_event(self):
        if self.last_sell_time is None:
            return None

        return {
            "kind": "sell",
            "time": self.last_sell_time,
            "underlying_price": self.last_sell_underlying_price,
            "execution_price": self.last_sell_execution_price,
        }

    def current_latest_trade_rule_price(self, basis, features):
        if basis in ("underlying", "stock"):
            return features["close"]

        if basis in ("execution", "instrument", "option", "bag"):
            return self.estimate_entry_buy_price(features)

        return None

    def previous_latest_trade_rule_price(self, basis, previous_event):
        if basis in ("underlying", "stock"):
            return previous_event.get("underlying_price")

        if basis in ("execution", "instrument", "option", "bag"):
            return previous_event.get("execution_price")

        return None

    def format_latest_trade_price_debug(self, debug):
        if not debug:
            return "[]"

        parts = []
        for item in debug:
            change = item["change"]
            if change is None:
                change_text = "None"
            elif item["mode"] == "pct":
                change_text = f"{change:.2%}"
            else:
                change_text = f"{change:.2f}"

            parts.append(
                f"{item['basis']}:{item['mode']} "
                f"prev_event={item['previous_event']} "
                f"prev={item['previous']} "
                f"cur={item['current']} "
                f"change={change_text} "
                f"max={item['max_change']} "
                f"passed={item['passed']}"
            )

        return "[" + "; ".join(parts) + "]"

    def check_double_down_price_rules(self, features):
        if not self.open_trades:
            return [], []

        previous_trade = self.open_trades[-1]
        reasons = []
        debug = []

        for rule in self.double_down_price_rules:
            basis = rule.get("basis", "underlying")
            mode = rule.get("mode", "pct")
            current_price = self.current_double_down_price(basis, features)
            previous_price = self.previous_double_down_price(basis, previous_trade)

            if current_price is None or previous_price is None:
                reasons.append(f"double_down_{basis}_price_missing")
                debug.append({
                    "basis": basis,
                    "mode": mode,
                    "current": current_price,
                    "previous": previous_price,
                    "change": None,
                    "max_change": self.rule_max_change(rule),
                    "passed": False,
                })
                continue

            change = current_price - previous_price

            if mode == "pct":
                if previous_price == 0:
                    reasons.append(f"double_down_{basis}_previous_price_zero")
                    passed = False
                    change_value = None
                else:
                    change_value = change / previous_price
                    passed = change_value <= self.rule_max_change(rule)
            elif mode == "abs":
                change_value = change
                passed = change_value <= self.rule_max_change(rule)
            else:
                reasons.append(f"double_down_{basis}_unsupported_mode")
                passed = False
                change_value = None

            debug.append({
                "basis": basis,
                "mode": mode,
                "current": current_price,
                "previous": previous_price,
                "change": change_value,
                "max_change": self.rule_max_change(rule),
                "passed": passed,
            })

            if not passed:
                reasons.append(f"double_down_{basis}_{mode}_change_not_low_enough")

        return reasons, debug

    def current_double_down_price(self, basis, features):
        if basis in ("underlying", "stock"):
            return features["close"]

        if basis in ("execution", "instrument", "option", "bag"):
            if self.execution_instrument is None:
                return features["close"]
            return self.execution_instrument.estimate_market_price()

        return None

    def previous_double_down_price(self, basis, previous_trade):
        if basis in ("underlying", "stock"):
            return previous_trade.get("entry_underlying_price", previous_trade.get("entry_price"))

        if basis in ("execution", "instrument", "option", "bag"):
            return previous_trade.get("entry_price")

        return None

    def rule_max_change(self, rule):
        if "max_change" in rule:
            return float(rule["max_change"])

        if "discount_pct" in rule:
            return -float(rule["discount_pct"])

        if "discount_abs" in rule:
            return -float(rule["discount_abs"])

        return -float(self.double_down_discount_pct)

    def format_double_down_debug(self, debug):
        parts = []

        for item in debug:
            change = item["change"]
            if change is None:
                change_text = "None"
            elif item["mode"] == "pct":
                change_text = f"{change:.2%}"
            else:
                change_text = f"{change:.2f}"

            parts.append(
                f"{item['basis']}:{item['mode']} "
                f"prev={item['previous']} "
                f"cur={item['current']} "
                f"change={change_text} "
                f"max={item['max_change']} "
                f"passed={item['passed']}"
            )

        return "[" + "; ".join(parts) + "]"

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
                self.close_trade(t, features)
            else:
                still_open.append(t)

        self.open_trades = still_open

        if self.autosave_open_trades:
            self.open_trades_store.save(self.open_trades)

    def close_trade(self, trade_info, features):
        close_price = features["close"]
        today = features["date"]

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
        self.last_sell_time = features["time"]
        self.last_sell_underlying_price = close_price
        self.last_sell_execution_price = exit_limit

        if self.DEBUG_ORDERS:
            self.log(
                f"[SELL SENT] {self.symbol} qty={trade_info['qty']} "
                f"limit={exit_limit} "
                f"underlying_ref={close_price:.2f} "
                f"instrument={trade_info.get('instrument_type', 'stock')} "
                f"last_sell_date={today}"
            )

        return trade
