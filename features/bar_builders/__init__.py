from features.bar_builders.dollar_bars import DollarBarBuilder
from features.bar_builders.time_bars import TimeBarBuilder


def build_signal_bar_builder(features_cfg, logger=print, debug=False):
    bar_cfg = features_cfg.get("bar", {})
    bar_type = bar_cfg.get("type", "time")

    if bar_type == "time":
        return TimeBarBuilder(
            timeframe=bar_cfg.get("timeframe", features_cfg.get("timeframe", "1min")),
            history_window=int(bar_cfg.get("history_window", features_cfg.get("minute_window", 390))),
            logger=logger,
            debug=debug,
        )

    if bar_type == "dollar":
        return DollarBarBuilder(
            dollar_threshold=float(bar_cfg["dollar_threshold"]),
            history_window=int(bar_cfg.get("history_window", features_cfg.get("minute_window", 390))),
            logger=logger,
            debug=debug,
        )

    raise ValueError(f"Unsupported signal bar type: {bar_type}")


__all__ = [
    "DollarBarBuilder",
    "TimeBarBuilder",
    "build_signal_bar_builder",
]
