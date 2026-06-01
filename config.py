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
    },

    "double_down": {
        # First entry:
        #   bb_score <= -k
        #
        # Double down:
        #   bb_score <= -(bb_mult * k)
        #   and close <= previous_entry_price * (1 - discount_pct)
        "enabled": True,
        "bb_mult": 3,
        "discount_pct": 0.01,
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
