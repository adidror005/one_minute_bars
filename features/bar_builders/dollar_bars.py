from collections import deque

from features.bar_builders.base import merge_ohlcv_bar, raw_bar_to_dict


class DollarBarBuilder:
    def __init__(self, dollar_threshold, history_window=390, logger=print, debug=False):
        if dollar_threshold <= 0:
            raise ValueError("dollar_threshold must be positive.")

        self.dollar_threshold = float(dollar_threshold)
        self.history_window = history_window
        self.log = logger
        self.debug = debug
        self.bars = deque(maxlen=history_window)
        self.current_bar = None
        self.current_dollar_value = 0.0
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

    def bootstrap(self, raw_bars):
        self.bars.clear()
        self.current_bar = None
        self.current_dollar_value = 0.0

        rows = sorted(
            [raw_bar_to_dict(bar) for bar in raw_bars],
            key=lambda item: item["time"],
        )

        for raw_bar in rows:
            self.update_one(raw_bar, append_completed=True)

        self.bootstrapped = True

        if self.debug:
            self.log(
                f"[BOOTSTRAP] loaded {len(self.bars)} completed dollar bars | "
                f"threshold={self.dollar_threshold}"
            )

    def update_one(self, raw_bar, append_completed=True):
        self.last_raw_time = raw_bar["time"]

        if self.current_bar is None:
            self.current_bar = self.start_bar(raw_bar)
        else:
            self.current_bar["time"] = raw_bar["time"]
            merge_ohlcv_bar(self.current_bar, raw_bar)

        self.current_dollar_value += self.raw_dollar_value(raw_bar)

        if self.current_dollar_value < self.dollar_threshold:
            return None

        completed_bar = self.current_bar
        completed_bar["dollar_value"] = self.current_dollar_value

        if append_completed:
            self.bars.append(completed_bar)

        self.current_bar = None
        self.current_dollar_value = 0.0

        if self.debug:
            self.log(
                f"[NEW SIGNAL BAR] type=dollar "
                f"time={completed_bar['time']} "
                f"O={completed_bar['open']} H={completed_bar['high']} "
                f"L={completed_bar['low']} C={completed_bar['close']} "
                f"V={completed_bar['volume']} "
                f"D={completed_bar['dollar_value']:.2f} | stored={len(self.bars)}"
            )

        return completed_bar

    def start_bar(self, raw_bar):
        return {
            "time": raw_bar["time"],
            "start_time": raw_bar["time"],
            "open": raw_bar["open"],
            "high": raw_bar["high"],
            "low": raw_bar["low"],
            "close": raw_bar["close"],
            "volume": raw_bar["volume"],
        }

    def raw_dollar_value(self, raw_bar):
        return raw_bar["close"] * raw_bar["volume"]
