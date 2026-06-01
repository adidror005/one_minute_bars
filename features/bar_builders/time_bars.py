from collections import deque

import pandas as pd

from features.bar_builders.base import merge_ohlcv_bar, raw_bar_to_dict


class TimeBarBuilder:
    def __init__(self, timeframe="1min", history_window=390, logger=print, debug=False):
        self.timeframe = timeframe
        self.history_window = history_window
        self.log = logger
        self.debug = debug
        self.bars = deque(maxlen=history_window)
        self.current_bar = None
        self.last_raw_time = None
        self.bootstrapped = False

    def update(self, raw_bars):
        if not raw_bars:
            return None

        if not self.bootstrapped:
            self.bootstrap(raw_bars)
            return None

        completed_bar = None
        unseen_bars = []
        for bar in raw_bars:
            raw_bar = raw_bar_to_dict(bar)
            if self.last_raw_time is None or raw_bar["time"] > self.last_raw_time:
                unseen_bars.append(raw_bar)

        for raw_bar in sorted(unseen_bars, key=lambda item: item["time"]):
            maybe_completed = self.update_one(raw_bar)
            if maybe_completed is not None:
                completed_bar = maybe_completed

        return completed_bar

    def update_one(self, raw_bar):
        self.last_raw_time = raw_bar["time"]
        bucket_start = self.bucket_start(raw_bar["time"])

        if self.current_bar is None:
            self.current_bar = self.start_bar(raw_bar, bucket_start)
            return None

        if bucket_start == self.current_bar["time"]:
            merge_ohlcv_bar(self.current_bar, raw_bar)
            return None

        completed_bar = self.current_bar
        self.bars.append(completed_bar)
        self.current_bar = self.start_bar(raw_bar, bucket_start)

        if self.debug:
            self.log(
                f"[NEW SIGNAL BAR] timeframe={self.timeframe} "
                f"time={completed_bar['time']} "
                f"O={completed_bar['open']} H={completed_bar['high']} "
                f"L={completed_bar['low']} C={completed_bar['close']} "
                f"V={completed_bar['volume']} | stored={len(self.bars)}"
            )

        return completed_bar

    def bootstrap(self, raw_bars):
        rows = [raw_bar_to_dict(bar) for bar in raw_bars]
        df = pd.DataFrame(rows)

        if df.empty:
            self.bootstrapped = True
            return

        df = df.set_index("time").sort_index()
        self.last_raw_time = df.index.max().to_pydatetime()
        resampled = self.resample(df)

        self.bars.clear()
        if len(resampled) > 1:
            for row in resampled.iloc[:-1].tail(self.bars.maxlen).itertuples():
                self.bars.append(self.row_to_bar(row))

        if len(resampled) > 0:
            self.current_bar = self.row_to_bar(resampled.iloc[-1])

        self.bootstrapped = True

        if self.debug:
            self.log(
                f"[BOOTSTRAP] loaded {len(self.bars)} completed signal bars | "
                f"timeframe={self.timeframe}"
            )

    def resample(self, df):
        return (
            df.resample(self.timeframe)
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })
            .dropna()
        )

    def bucket_start(self, timestamp):
        return pd.Timestamp(timestamp).floor(self.timeframe).to_pydatetime()

    def start_bar(self, raw_bar, bucket_start):
        return {
            "time": bucket_start,
            "open": raw_bar["open"],
            "high": raw_bar["high"],
            "low": raw_bar["low"],
            "close": raw_bar["close"],
            "volume": raw_bar["volume"],
        }

    def row_to_bar(self, row):
        if hasattr(row, "Index"):
            return {
                "time": row.Index.to_pydatetime() if hasattr(row.Index, "to_pydatetime") else row.Index,
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(row.volume),
            }

        return {
            "time": row.name.to_pydatetime() if hasattr(row.name, "to_pydatetime") else row.name,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }
