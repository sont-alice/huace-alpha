from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS


def build_feature_frame(market: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    data = market.sort_values(["code", "date"]).copy()
    grouped = data.groupby("code", group_keys=False)

    data["ret_20"] = grouped["close"].pct_change(20)
    data["ret_60"] = grouped["close"].pct_change(60)
    data["ret_120"] = grouped["close"].pct_change(120)
    data["ma_20"] = grouped["close"].transform(lambda s: s.rolling(20).mean())
    data["ma_60"] = grouped["close"].transform(lambda s: s.rolling(60).mean())
    data["ma_20_gap"] = data["close"] / data["ma_20"] - 1
    data["ma_60_gap"] = data["close"] / data["ma_60"] - 1
    data["daily_ret"] = grouped["close"].pct_change()
    data["volatility_20"] = grouped["daily_ret"].transform(lambda s: s.rolling(20).std())
    data["turnover_20"] = grouped["turnover_rate"].transform(lambda s: s.rolling(20).mean())
    data["amount_20"] = grouped["amount"].transform(lambda s: s.rolling(20).mean())
    data["money_flow_20"] = grouped["money_flow"].transform(lambda s: s.rolling(20).sum())
    data["market_cap_log"] = np.log(data["market_cap"].clip(lower=1))
    market_daily = data.groupby("date")["daily_ret"].median().rename("market_daily_ret").reset_index()
    market_daily["market_return_20"] = market_daily["market_daily_ret"].rolling(20, min_periods=20).sum()
    data = data.merge(market_daily[["date", "market_return_20"]], on="date", how="left")

    industry_daily = data.groupby(["industry", "date"])["daily_ret"].mean().reset_index()
    industry_daily["industry_strength_20"] = industry_daily.groupby("industry")["daily_ret"].transform(
        lambda s: s.rolling(20, min_periods=20).sum()
    )
    date_state = data.groupby("date").agg(
        market_breadth_20=("ret_20", lambda s: float((s > 0).mean())),
        market_above_ma20=("ma_20_gap", lambda s: float((s > 0).mean())),
    ).reset_index()
    data = data.merge(
        industry_daily[["industry", "date", "industry_strength_20"]],
        on=["industry", "date"],
        how="left",
    ).sort_values(["code", "date"])
    data = data.merge(date_state, on="date", how="left").sort_values(["code", "date"])
    grouped = data.groupby("code", group_keys=False)

    future_close = grouped["close"].shift(-horizon_days)
    data["future_return"] = future_close / data["close"] - 1
    date_median = data.groupby("date")["future_return"].transform("median")
    data["excess_return"] = data["future_return"] - date_median
    data["target_win"] = (data["excess_return"] > 0).astype(float)

    return data.drop(columns=["ma_20", "ma_60"]).replace([np.inf, -np.inf], np.nan)


def latest_features(features: pd.DataFrame) -> pd.DataFrame:
    latest = features.sort_values(["code", "date"]).groupby("code", as_index=False).tail(1).copy()
    return latest.dropna(subset=FEATURE_COLUMNS, how="all")
