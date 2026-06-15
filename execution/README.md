# Execution Instrument

Use underlying bars for **signals**, and route orders to **stock** or a single **option**.

## Wiring

```python
from config import DEFAULT_LIVE_CONFIG, LiveTradingConfig
from execution import build_execution_instrument
from orders import IBKRLimitOrderRouter
from strategies.live_mean_reversion import LiveMeanReversion

config = LiveTradingConfig.from_dict(DEFAULT_LIVE_CONFIG)

# Trade a single call while signals come from META stock bars.
config.raw["execution"]["instrument"] = {
    "type": "option",
    "expiry": "20260605",
    "strike": 540,
    "right": "C",
    "limit_entry_offset_pct": 0.02,
    "limit_exit_offset_pct": 0.02,
}

execution_instrument = build_execution_instrument(
    ib=ib,
    execution_cfg=config.execution,
    symbol=config.symbol,
)

order_router = IBKRLimitOrderRouter(ib=ib)

algo = LiveMeanReversion(
    config=config,
    order_router=order_router,
    execution_instrument=execution_instrument,
)

# Keep streaming underlying bars:
# real_time_bars = ib.reqRealTimeBars(stock, ...)
# real_time_bars.updateEvent += algo.on_bar
```

## Behavior

- Entry/exit **decisions** use underlying features (`close`, BB score, RSI, etc.).
- Entry/exit **orders** use the configured stock or option contract and live market-data limit prices.
- Take-profit checks compare underlying close vs `entry_underlying_price`.
