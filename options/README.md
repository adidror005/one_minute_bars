# Option Engine Notes

The pricing engine blends:

- direct live quote information (`bid/ask`, midpoint, model price, last), and
- a local implied-volatility smile fit from nearby strikes.

## Calibration inputs

- Market data is fetched live from the configured provider (for IB: `ib_async`).
- The smile is fit from neighbor strikes in the same expiry/right.
- `underlier_price` comes from option ticker fields (`undPrice` / model greeks fallback).

## Risk-free rate

- Default is a flat annualized rate (`risk_free_rate`, default `0.03`).
- You can inject `risk_free_rate_provider(contract, t_years)` to supply dynamic term rates.
- Engine additionally estimates a market-implied short-dated rate from put-call parity and blends it with the base rate.

## Time to expiry (intraday)

- Time to expiry is computed to expiry-day market close (default 16:00 in `America/New_York`).
- You can override timezone/close time via engine constructor.
- Computation is second-level and intended for 0DTE/weekly options.

## Stock-move projection

- `predict_option_change_for_stock_move(..., use_bs_reanchor=True)` uses:
  - `new_price ~= current_market_price + (BS(S_new) - BS(S_old))`
- This keeps projections anchored to current market pricing while using BS for local move dynamics.
