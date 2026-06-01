# ============================================================
# FEATURE CALCULATOR
# Keeps feature engineering out of the trading class.
# Feature names match the offline CatBoost feature_cols.
# ============================================================
import numpy as np
import pandas as pd


class LiveMedianMadFeatureCalculator:
    def __init__(self, window=20, min_bars=30):
        self.window = window
        self.min_bars = min_bars

    def calculate(self, minute_bars):
        if len(minute_bars) < self.min_bars:
            return None

        df = self._minute_bars_to_df(minute_bars)

        if df is None or len(df) < self.min_bars:
            return None

        df = self._add_base_time_columns(df)
        df = self._add_median_mad_features(df)
        df = self._add_standard_bb_features(df)
        df = self._add_return_vol_features(df)
        df = self._add_bb_features(df)
        df = self._add_sma_trend_features(df)
        df = self._add_vwap_features(df)
        df = self._add_volume_features(df)
        df = self._add_candle_range_features(df)
        df = self._add_local_extreme_features(df)
        df = self._add_opening_range_features(df)
        df = self._add_time_features(df)
        df = self._add_rsi_features(df)
        df = self._add_atr_features(df)

        df = df.replace([np.inf, -np.inf], np.nan)

        row = df.iloc[-1]

        if pd.isna(row["close"]):
            return None

        latest_time = row["time"]
        session_minutes = self.get_session_minutes(latest_time)
        minutes_until_close = max(0, 390 - session_minutes - 1)

        features = self._row_to_features(
            row=row,
            session_minutes=session_minutes,
            minutes_until_close=minutes_until_close,
        )

        return features

    def _minute_bars_to_df(self, minute_bars):
        df = pd.DataFrame(list(minute_bars)).copy()

        if df.empty:
            return None

        required_cols = ["time", "open", "high", "low", "close", "volume"]
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            return None

        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)

        latest_day = df["time"].iloc[-1].date()

        df["day"] = df["time"].dt.date
        df = df[df["day"] == latest_day].copy().reset_index(drop=True)

        return df

    def _add_base_time_columns(self, df):
        df["minute_of_day"] = df["time"].dt.hour * 60 + df["time"].dt.minute
        df["day_bar_num"] = df.groupby("day").cumcount()
        return df

    def _add_median_mad_features(self, df):
        df["median"] = (
            df.groupby("day")["close"]
            .transform(lambda s: s.rolling(self.window).median())
        )

        df["absolute_deviation"] = (df["close"] - df["median"]).abs()

        df["mad"] = (
            df.groupby("day")["absolute_deviation"]
            .transform(lambda s: s.rolling(self.window).median())
        )

        df["bb_score"] = (
            (df["close"] - df["median"])
            / df["mad"].replace(0, np.nan)
        )

        df["bb_score_prev"] = df.groupby("day")["bb_score"].shift(1)

        return df

    def _add_standard_bb_features(self, df):
        df["standard_bb_mean"] = (
            df.groupby("day")["close"]
            .transform(lambda s: s.rolling(self.window).mean())
        )

        df["standard_bb_std"] = (
            df.groupby("day")["close"]
            .transform(lambda s: s.rolling(self.window).std())
        )

        df["standard_bb_z"] = (
            (df["close"] - df["standard_bb_mean"])
            / df["standard_bb_std"].replace(0, np.nan)
        )

        df["standard_bb_lower_2"] = df["standard_bb_mean"] - (2 * df["standard_bb_std"])
        df["standard_bb_upper_2"] = df["standard_bb_mean"] + (2 * df["standard_bb_std"])
        df["standard_bb_pct_b"] = (
            (df["close"] - df["standard_bb_lower_2"])
            / (df["standard_bb_upper_2"] - df["standard_bb_lower_2"]).replace(0, np.nan)
        )

        return df

    def _add_return_vol_features(self, df):
        df["ret_1m"] = df.groupby("day")["close"].pct_change(1)
        df["ret_5m"] = df.groupby("day")["close"].pct_change(5)
        df["ret_15m"] = df.groupby("day")["close"].pct_change(15)
        df["ret_30m"] = df.groupby("day")["close"].pct_change(30)
        df["ret_60m"] = df.groupby("day")["close"].pct_change(60)

        df["vol_15m"] = (
            df.groupby("day")["ret_1m"]
            .transform(lambda s: s.rolling(15).std())
        )

        df["vol_30m"] = (
            df.groupby("day")["ret_1m"]
            .transform(lambda s: s.rolling(30).std())
        )

        df["vol_60m"] = (
            df.groupby("day")["ret_1m"]
            .transform(lambda s: s.rolling(60).std())
        )

        df["vol_ratio_15_60"] = (
            df["vol_15m"] / df["vol_60m"].replace(0, np.nan)
        )

        return df

    def _add_bb_features(self, df):
        df["bb_change_1m"] = df.groupby("day")["bb_score"].diff(1)
        df["bb_change_5m"] = df.groupby("day")["bb_score"].diff(5)
        df["bb_change_15m"] = df.groupby("day")["bb_score"].diff(15)

        df["bb_min_30m"] = (
            df.groupby("day")["bb_score"]
            .transform(lambda s: s.rolling(30).min())
        )

        df["bb_max_30m"] = (
            df.groupby("day")["bb_score"]
            .transform(lambda s: s.rolling(30).max())
        )

        df["bb_mean_30m"] = (
            df.groupby("day")["bb_score"]
            .transform(lambda s: s.rolling(30).mean())
        )

        return df

    def _add_sma_trend_features(self, df):
        df["sma_15m"] = (
            df.groupby("day")["close"]
            .transform(lambda s: s.rolling(15).mean())
        )

        df["sma_60m"] = (
            df.groupby("day")["close"]
            .transform(lambda s: s.rolling(60).mean())
        )

        df["dist_sma_15m"] = (df["close"] - df["sma_15m"]) / df["close"]
        df["dist_sma_60m"] = (df["close"] - df["sma_60m"]) / df["close"]

        df["trend_15m"] = df.groupby("day")["sma_15m"].diff(15) / df["close"]
        df["trend_60m"] = df.groupby("day")["sma_60m"].diff(60) / df["close"]

        return df

    def _add_vwap_features(self, df):
        if "average" in df.columns:
            avg_price = df["average"].fillna(df["close"])
        else:
            avg_price = df["close"]

        df["pv"] = avg_price * df["volume"]

        df["cum_pv"] = df.groupby("day")["pv"].cumsum()
        df["cum_vol"] = df.groupby("day")["volume"].cumsum()

        df["day_vwap"] = df["cum_pv"] / df["cum_vol"].replace(0, np.nan)
        df["dist_day_vwap"] = (df["close"] - df["day_vwap"]) / df["close"]

        return df

    def _add_volume_features(self, df):
        df["volume_mean_5m"] = (
            df.groupby("day")["volume"]
            .transform(lambda s: s.rolling(5).mean())
        )

        df["volume_mean_60m"] = (
            df.groupby("day")["volume"]
            .transform(lambda s: s.rolling(60).mean())
        )

        df["volume_ratio_5_60"] = (
            df["volume_mean_5m"]
            / df["volume_mean_60m"].replace(0, np.nan)
        )

        return df

    def _add_candle_range_features(self, df):
        df["bar_range"] = (df["high"] - df["low"]) / df["close"]

        df["close_location"] = (
            (df["close"] - df["low"])
            / (df["high"] - df["low"]).replace(0, np.nan)
        )

        df["range_mean_30m"] = (
            df.groupby("day")["bar_range"]
            .transform(lambda s: s.rolling(30).mean())
        )

        df["range_std_30m"] = (
            df.groupby("day")["bar_range"]
            .transform(lambda s: s.rolling(30).std())
        )

        df["range_z_30m"] = (
            (df["bar_range"] - df["range_mean_30m"])
            / df["range_std_30m"].replace(0, np.nan)
        )

        return df

    def _add_local_extreme_features(self, df):
        df["rolling_low_60m"] = (
            df.groupby("day")["low"]
            .transform(lambda s: s.rolling(60).min())
        )

        df["rolling_high_60m"] = (
            df.groupby("day")["high"]
            .transform(lambda s: s.rolling(60).max())
        )

        df["dist_from_low_60m"] = df["close"] / df["rolling_low_60m"] - 1
        df["dist_from_high_60m"] = df["close"] / df["rolling_high_60m"] - 1

        return df

    def _add_opening_range_features(self, df):
        # Live-safe version:
        # Before first 30 minutes complete, this uses the bars available so far.
        opening = df[df["day_bar_num"] < 30]

        first_30m_high = opening["high"].max() if len(opening) > 0 else np.nan
        first_30m_low = opening["low"].min() if len(opening) > 0 else np.nan

        df["first_30m_high"] = first_30m_high
        df["first_30m_low"] = first_30m_low

        df["dist_opening_high"] = df["close"] / df["first_30m_high"] - 1
        df["dist_opening_low"] = df["close"] / df["first_30m_low"] - 1

        return df

    def _add_time_features(self, df):
        df["first_hour"] = (df["minute_of_day"] < 10 * 60 + 30).astype(int)

        df["lunch_hour"] = (
            (df["minute_of_day"] >= 12 * 60)
            & (df["minute_of_day"] < 13 * 60)
        ).astype(int)

        df["last_hour"] = (df["minute_of_day"] >= 15 * 60).astype(int)

        df["before_noon"] = (df["minute_of_day"] < 12 * 60).astype(int)

        return df

    def _add_rsi_features(self, df):
        df["rsi_14"] = self.rsi_by_day(df["close"], df["day"], 14)
        df["rsi_30"] = self.rsi_by_day(df["close"], df["day"], 30)

        df["rsi_14_change_5m"] = df.groupby("day")["rsi_14"].diff(5)
        df["rsi_14_change_15m"] = df.groupby("day")["rsi_14"].diff(15)

        df["rsi_14_min_30m"] = (
            df.groupby("day")["rsi_14"]
            .transform(lambda s: s.rolling(30).min())
        )

        df["rsi_14_max_30m"] = (
            df.groupby("day")["rsi_14"]
            .transform(lambda s: s.rolling(30).max())
        )

        df["rsi_14_mean_30m"] = (
            df.groupby("day")["rsi_14"]
            .transform(lambda s: s.rolling(30).mean())
        )

        df["rsi_14_dist_from_min_30m"] = df["rsi_14"] - df["rsi_14_min_30m"]
        df["rsi_14_dist_from_max_30m"] = df["rsi_14"] - df["rsi_14_max_30m"]

        return df

    def _add_atr_features(self, df):
        prev_close = df.groupby("day")["close"].shift(1)

        df["true_range"] = np.maximum.reduce(
            [
                (df["high"] - df["low"]).to_numpy(),
                (df["high"] - prev_close).abs().to_numpy(),
                (df["low"] - prev_close).abs().to_numpy(),
            ]
        )

        df["atr_14m"] = (
            df.groupby("day")["true_range"]
            .transform(lambda s: s.rolling(14).mean())
        )

        df["atr_30m"] = (
            df.groupby("day")["true_range"]
            .transform(lambda s: s.rolling(30).mean())
        )

        df["atr_60m"] = (
            df.groupby("day")["true_range"]
            .transform(lambda s: s.rolling(60).mean())
        )

        df["atr_14m_pct"] = df["atr_14m"] / df["close"]
        df["atr_30m_pct"] = df["atr_30m"] / df["close"]
        df["atr_60m_pct"] = df["atr_60m"] / df["close"]

        df["mad_pct"] = df["mad"] / df["close"]

        df["atr_ratio_14_60"] = df["atr_14m"] / df["atr_60m"].replace(0, np.nan)
        df["atr_ratio_30_60"] = df["atr_30m"] / df["atr_60m"].replace(0, np.nan)

        return df

    def rsi_by_day(self, close, day, window):
        delta = close.groupby(day).diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = (
            gain.groupby(day)
            .transform(
                lambda s: s.ewm(
                    alpha=1 / window,
                    adjust=False,
                    min_periods=window,
                ).mean()
            )
        )

        avg_loss = (
            loss.groupby(day)
            .transform(
                lambda s: s.ewm(
                    alpha=1 / window,
                    adjust=False,
                    min_periods=window,
                ).mean()
            )
        )

        rs = avg_gain / avg_loss.replace(0, np.nan)

        return 100 - (100 / (1 + rs))

    def get_session_minutes(self, ts):
        return (ts.hour - 9) * 60 + (ts.minute - 30)

    def _row_to_features(self, row, session_minutes, minutes_until_close):
        return {
            # Existing strategy fields
            "time": row["time"],
            "date": row["time"].date(),
            "close": row["close"],
            "median": row["median"],
            "mad": row["mad"],
            "bb_score": row["bb_score"],
            "standard_bb_mean": row["standard_bb_mean"],
            "standard_bb_std": row["standard_bb_std"],
            "standard_bb_z": row["standard_bb_z"],
            "standard_bb_lower_2": row["standard_bb_lower_2"],
            "standard_bb_upper_2": row["standard_bb_upper_2"],
            "standard_bb_pct_b": row["standard_bb_pct_b"],
            "rsi": row["rsi_14"],
            "session_minutes": session_minutes,
            "minutes_until_close": minutes_until_close,

            # Matching offline feature columns
            "bb_score_prev": row["bb_score_prev"],
            "bb_change_1m": row["bb_change_1m"],
            "bb_change_5m": row["bb_change_5m"],
            "bb_change_15m": row["bb_change_15m"],
            "bb_min_30m": row["bb_min_30m"],
            "bb_max_30m": row["bb_max_30m"],
            "bb_mean_30m": row["bb_mean_30m"],

            "ret_1m": row["ret_1m"],
            "ret_5m": row["ret_5m"],
            "ret_15m": row["ret_15m"],
            "ret_30m": row["ret_30m"],
            "ret_60m": row["ret_60m"],

            "vol_15m": row["vol_15m"],
            "vol_30m": row["vol_30m"],
            "vol_60m": row["vol_60m"],
            "vol_ratio_15_60": row["vol_ratio_15_60"],

            "dist_sma_15m": row["dist_sma_15m"],
            "dist_sma_60m": row["dist_sma_60m"],
            "trend_15m": row["trend_15m"],
            "trend_60m": row["trend_60m"],

            "dist_day_vwap": row["dist_day_vwap"],
            "volume_ratio_5_60": row["volume_ratio_5_60"],

            "bar_range": row["bar_range"],
            "close_location": row["close_location"],
            "range_z_30m": row["range_z_30m"],

            "dist_from_low_60m": row["dist_from_low_60m"],
            "dist_from_high_60m": row["dist_from_high_60m"],

            "dist_opening_high": row["dist_opening_high"],
            "dist_opening_low": row["dist_opening_low"],

            "minute_of_day": row["minute_of_day"],
            "day_bar_num": row["day_bar_num"],
            "first_hour": row["first_hour"],
            "before_noon": row["before_noon"],
            "lunch_hour": row["lunch_hour"],

            "rsi_14": row["rsi_14"],
            "rsi_30": row["rsi_30"],
            "rsi_14_change_5m": row["rsi_14_change_5m"],
            "rsi_14_change_15m": row["rsi_14_change_15m"],
            "rsi_14_min_30m": row["rsi_14_min_30m"],
            "rsi_14_max_30m": row["rsi_14_max_30m"],
            "rsi_14_mean_30m": row["rsi_14_mean_30m"],
            "rsi_14_dist_from_min_30m": row["rsi_14_dist_from_min_30m"],
            "rsi_14_dist_from_max_30m": row["rsi_14_dist_from_max_30m"],

            "atr_14m_pct": row["atr_14m_pct"],
            "atr_30m_pct": row["atr_30m_pct"],
            "atr_60m_pct": row["atr_60m_pct"],
            "mad_pct": row["mad_pct"],
            "atr_ratio_14_60": row["atr_ratio_14_60"],
            "atr_ratio_30_60": row["atr_ratio_30_60"],
        }
