from dataclasses import dataclass
from pathlib import Path

# ============================================================
# DEFAULT CONFIG
# If you want YAML later, this dict can become config_live.yaml.
# For now, this keeps everything in one giant cell.
# ============================================================

DEFAULT_LIVE_CONFIG = {
    "symbol": "META",

    "paths": {
        "open_trades_path": "META_open_trades.csv",
        "catboost_model_path": "models/live/catboost/meta_catboost.cbm",
    },

    "features": {
        "window": 20,
        "min_bars": 60,
        # Feature windows are measured in completed signal bars. If timeframe
        # is "5min", window=20 means 20 five-minute bars.
        # Completed signal bars used by the strategy.
        # Current implementation supports time and dollar bars using the same
        # raw IB stream.
        "bar": {
            "type": "time",
            "timeframe": "1min",
            "history_window": 390,
            # Examples:
            # "timeframe": "5min",
            # "type": "dollar",
            # "dollar_threshold": 1_000_000,
        },
        # Backward-compatible fallback for older configs.
        "minute_window": 390,
    },

    "model": {
        # If enabled=False, the default AlwaysPassModelFilter is used.
        # It returns prob=1.0 and passed=True.
        "enabled": False,
        "prob_threshold": 0.50,

        "feature_cols": [
            "bb_score",
            "bb_score_prev",
            "bb_change_1m",
            "bb_change_5m",
            "bb_change_15m",
            "bb_min_30m",
            "bb_max_30m",
            "bb_mean_30m",

            "ret_1m",
            "ret_5m",
            "ret_15m",
            "ret_30m",
            "ret_60m",

            "vol_15m",
            "vol_30m",
            "vol_60m",
            "vol_ratio_15_60",

            "dist_sma_15m",
            "dist_sma_60m",
            "trend_15m",
            "trend_60m",

            "dist_day_vwap",
            "volume_ratio_5_60",

            "bar_range",
            "close_location",
            "range_z_30m",

            "dist_from_low_60m",
            "dist_from_high_60m",

            "dist_opening_high",
            "dist_opening_low",

            "minute_of_day",
            "day_bar_num",
            "first_hour",
            "before_noon",
            "lunch_hour",

            "rsi_14",
            "rsi_30",
            "rsi_14_change_5m",
            "rsi_14_change_15m",
            "rsi_14_min_30m",
            "rsi_14_max_30m",
            "rsi_14_mean_30m",
            "rsi_14_dist_from_min_30m",
            "rsi_14_dist_from_max_30m",

            "atr_14m_pct",
            "atr_30m_pct",
            "atr_60m_pct",
            "mad_pct",
            "atr_ratio_14_60",
            "atr_ratio_30_60",
        ],
    },

    "strategy": {
        "k": 2,

        "max_new_trades_per_day": 6,
        "max_open_trades_total": 200,
        "max_open_trades_today": 2,

        "no_trade_first_minutes": 60,
        "no_new_entries_last_minutes": 60,

        "min_take_profit": 0.01,
        "fuckup_take_profit": 0.10,

        "rsi_thresh": 300,

        "reentry_cooldown_days": 7,
        "reentry_discount_pct": 0.01,

        "limit_entry_offset_pct": 0.0005,

        # Selectable entry strategy. This only decides whether to enter.
        # Signal data still comes from the underlying stock, and execution can
        # still route to stock/options/bags through execution.instrument.
        #
        # median_mad: old behavior, using median/MAD z-score in bb_score.
        # standard_bb: rolling mean/std Bollinger z-score, not median-based.
        # always_true: no signal predicate; risk/price/double-down guards still apply.
        "entry_strategy": {
            "name": "median_mad",
            # "name": "standard_bb",
            # "name": "always_true",
            # "z": 2.0,
            # "double_down_mult": 3.0,
        },

        # Hard price gates for entries. These are safety limits, independent
        # of the selected entry strategy.
        #
        # max_stock_price / min_stock_price use the underlying signal close.
        # max_buy_price / min_buy_price use the estimated execution buy limit.
        # For stock execution, buy price is effectively the stock limit price.
        # For options/bags, buy price is the option/combo estimated limit.
        "entry_price_limits": {
            "first_entry": {
                "max_stock_price": None,
                "min_stock_price": None,
                "max_buy_price": None,
                "min_buy_price": None,
            },
            "double_down": {
                "max_stock_price": None,
                "min_stock_price": None,
                "max_buy_price": None,
                "min_buy_price": None,
            },
        },

        # Optional raw feature-based overrides.
        # If first_entry or double_down is set here, those rules override
        # entry_strategy for that entry type.
        #
        # Any feature returned by LiveMedianMadFeatureCalculator can be used,
        # for example standard_bb_z, rsi_14, dist_day_vwap, ret_5m.
        # Supported ops: <, <=, >, >=, ==, !=.
        # Every rule in the selected list must pass.
        "entry_conditions": {
            # "first_entry": [
            #     {"field": "standard_bb_z", "op": "<=", "value": -2.0},
            #     {"field": "rsi_14", "op": "<=", "value": 35.0},
            # ],
            # "double_down": [
            #     {"field": "standard_bb_z", "op": "<=", "value": -6.0},
            # ],
        },
    },

    "double_down": {
        # Double down requires:
        #   1. the configured double_down entry strategy/rules pass
        #   2. every configured price rule passes
        "enabled": True,
        "bb_mult": 3,
        # Backward-compatible default. Used if price_rules is not set.
        "discount_pct": 0.01,
        # basis: underlying/stock uses signal stock price.
        # basis: execution/instrument/option/bag uses traded instrument price.
        # mode: pct compares (current - previous) / previous.
        # mode: abs compares current - previous in dollars.
        # max_change should usually be negative for averaging down.
        "price_rules": [
            {"basis": "underlying", "mode": "pct", "max_change": -0.01},
            # {"basis": "underlying", "mode": "abs", "max_change": -5.00},
            # {"basis": "execution", "mode": "pct", "max_change": -0.20},
            # {"basis": "execution", "mode": "abs", "max_change": -0.50},
        ],
    },

    "sizing": {
        "default_qty": 100,

        # active open trades -> next order qty
        # 0 open trades = first entry
        # 1 open trade  = first double down
        # 2 open trades = second double down
        "size_schedule": {
            0: 100,
            1: 100,
            2: 200,
            3: 300,
        },
    },

    "execution": {
        "tif": "DAY",
        "outside_rth": True,
        # Signal bars come from underlying stock. Orders can target stock/option/combo.
        "instrument": {
            "type": "stock",  # stock | option | combo
            "exchange": "SMART",
            "currency": "USD",
            # Option example:
            # "type": "option",
            # "expiry": "20260605",
            # "strike": 540,
            # "right": "C",
            # Combo example:
            # "type": "combo",
            # "legs": [
            #     {"expiry": "20260605", "strike": 540, "right": "C", "ratio": 1, "side": "BUY"},
            #     {"expiry": "20260605", "strike": 545, "right": "C", "ratio": 1, "side": "SELL"},
            # ],
            "limit_entry_offset_pct": 0.02,
            "limit_exit_offset_pct": 0.02,
        },
    },

    "saving": {
        "autosave_open_trades": True,
    },

    "debug": {
        "bars": True,
        "features": True,
        "signal": True,
        "orders": True,
        "save": True,
        "model": True,
    },
}


# ============================================================
# CONFIG WRAPPER
# Lets strategy read everything from config.
# ============================================================

@dataclass
class LiveTradingConfig:
    raw: dict

    @classmethod
    def from_dict(cls, raw):
        return cls(raw=raw)

    @classmethod
    def from_yaml(cls, path):
        import yaml

        path = Path(path)

        with path.open("r") as f:
            raw = yaml.safe_load(f)

        return cls(raw=raw)

    @property
    def symbol(self):
        return self.raw["symbol"]

    @property
    def paths(self):
        return self.raw.get("paths", {})

    @property
    def features(self):
        return self.raw.get("features", {})

    @property
    def model(self):
        return self.raw.get("model", {})

    @property
    def strategy(self):
        return self.raw.get("strategy", {})

    @property
    def double_down(self):
        return self.raw.get("double_down", {})

    @property
    def sizing(self):
        return self.raw.get("sizing", {})

    @property
    def execution(self):
        return self.raw.get("execution", {})

    @property
    def saving(self):
        return self.raw.get("saving", {})

    @property
    def debug(self):
        return self.raw.get("debug", {})
