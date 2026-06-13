from __future__ import annotations

import pandas as pd

from .config import StrategyConfig


def enrich_scores(scored: pd.DataFrame, config: StrategyConfig | None = None) -> pd.DataFrame:
    frame = scored.copy()
    frame["model_score"] = frame["score"]
    frame["model_rank"] = frame.groupby("date")["model_score"].rank(pct=True)
    frame["trend_score"] = (
        _rank(frame, "ret_20", True) * 0.28
        + _rank(frame, "ret_60", True) * 0.24
        + _rank(frame, "ret_120", True) * 0.16
        + _rank(frame, "ma_20_gap", True) * 0.18
        + _rank(frame, "ma_60_gap", True) * 0.14
    ).clip(0, 1)
    frame["risk_score"] = (
        _rank(frame, "volatility_20", False) * 0.45
        + _rank(frame, "amount_20", True) * 0.35
        + _rank(frame, "turnover_20", True) * 0.20
    ).clip(0, 1)
    frame["fundamental_score"] = (
        _rank(frame, "roe", True) * 0.45
        + _rank(frame, "net_profit_growth", True) * 0.30
        + _rank(frame, "pe_ttm", False) * 0.25
    ).clip(0, 1)
    frame["industry_score"] = _rank(frame, "industry_strength_20", True)
    frame["market_regime_score"] = (
        frame["market_breadth_20"].fillna(0.5) * 0.45
        + frame["market_above_ma20"].fillna(0.5) * 0.35
        + _bounded(frame["market_return_20"], -0.08, 0.08) * 0.20
    ).clip(0, 1)
    frame["composite_score"] = (
        frame["model_rank"].fillna(0.5) * 0.36
        + frame["trend_score"].fillna(0.5) * 0.24
        + frame["industry_score"].fillna(0.5) * 0.14
        + frame["fundamental_score"].fillna(0.5) * 0.14
        + frame["risk_score"].fillna(0.5) * 0.12
    ).clip(0, 1)
    frame["win_probability"] = (
        0.32
        + frame["composite_score"] * 0.34
        + frame["market_regime_score"] * 0.12
        - (1 - frame["risk_score"].fillna(0.5)) * 0.08
    ).clip(0.05, 0.88)
    frame["rating"] = pd.cut(
        frame["composite_score"],
        bins=[-0.01, 0.45, 0.62, 0.75, 0.86, 1.01],
        labels=["D", "C", "B", "A", "S"],
    ).astype(str)
    frame["score_rank"] = frame.groupby("date")["composite_score"].rank(pct=True)
    frame["score"] = frame["composite_score"]
    return frame


def _rank(frame: pd.DataFrame, column: str, ascending: bool) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.5, index=frame.index)
    return frame.groupby("date")[column].rank(pct=True, ascending=ascending).fillna(0.5)


def _bounded(series: pd.Series, low: float, high: float) -> pd.Series:
    return ((series.fillna(0.0).clip(low, high) - low) / (high - low)).fillna(0.5)
