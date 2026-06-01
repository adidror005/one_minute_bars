import pandas as pd


def raw_bar_to_dict(bar):
    return {
        "time": pd.to_datetime(bar.time).to_pydatetime(),
        "open": float(getattr(bar, "open_", getattr(bar, "open", 0.0))),
        "high": float(bar.high),
        "low": float(bar.low),
        "close": float(bar.close),
        "volume": max(float(getattr(bar, "volume", 0.0)), 0.0),
    }


def merge_ohlcv_bar(target, raw_bar):
    target["high"] = max(target["high"], raw_bar["high"])
    target["low"] = min(target["low"], raw_bar["low"])
    target["close"] = raw_bar["close"]
    target["volume"] += raw_bar["volume"]
