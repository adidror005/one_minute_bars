import pandas as pd
from pathlib import Path


class OpenTradesStore:
    def __init__(self, path, symbol, debug_save=True, logger=print):
        self.path = path
        self.symbol = symbol
        self.debug_save = debug_save
        self.log = logger

    def save(self, open_trades):
        rows = []

        for t in open_trades:
            rows.append({
                "symbol": self.symbol,
                "qty": t["qty"],
                "entry_price": t["entry_price"],
                "entry_underlying_price": t.get("entry_underlying_price"),
                "entry_date": t["entry_date"],
                "entry_time": t["entry_time"],
                "entry_reason": t.get("entry_reason"),
                "ml_prob": t.get("ml_prob"),
                "instrument_type": t.get("instrument_type"),
            })

        pd.DataFrame(rows).to_csv(self.path, index=False)

        if self.debug_save:
            self.log(
                f"[SAVE OPEN TRADES] saved {len(rows)} rows "
                f"to {self.path}"
            )

    def load(self):
        path = Path(self.path)

        if not path.exists():
            return []

        if path.stat().st_size == 0:
            if self.debug_save:
                self.log(f"[LOAD OPEN TRADES] empty file ignored: {self.path}")
            return []

        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            if self.debug_save:
                self.log(f"[LOAD OPEN TRADES] empty/corrupt file ignored: {self.path}")
            return []

        if df.empty:
            return []

        open_trades = []

        for row in df.itertuples(index=False):
            entry_reason = getattr(row, "entry_reason", None)
            ml_prob = getattr(row, "ml_prob", None)
            entry_underlying_price = getattr(row, "entry_underlying_price", None)
            instrument_type = getattr(row, "instrument_type", None)

            if pd.isna(entry_reason):
                entry_reason = None

            if pd.isna(ml_prob):
                ml_prob = None

            if pd.isna(entry_underlying_price):
                entry_underlying_price = float(row.entry_price)

            if pd.isna(instrument_type):
                instrument_type = "stock"

            open_trades.append({
                "trade": None,
                "qty": int(row.qty),
                "entry_price": float(row.entry_price),
                "entry_underlying_price": float(entry_underlying_price),
                "entry_date": pd.to_datetime(row.entry_date).date(),
                "entry_time": pd.to_datetime(row.entry_time),
                "entry_reason": entry_reason,
                "ml_prob": ml_prob,
                "instrument_type": instrument_type,
            })

        if self.debug_save:
            self.log(
                f"[LOAD OPEN TRADES] loaded {len(open_trades)} rows "
                f"from {self.path}"
            )

        return open_trades
