from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import _cap_industry, _tradable
from .config import StrategyConfig


def make_recommendations(latest_scored: pd.DataFrame, config: StrategyConfig, allow_buy: bool) -> pd.DataFrame:
    candidates = _tradable(latest_scored, config).sort_values("score", ascending=False)
    candidates = _cap_industry(candidates.head(config.top_n * 4), config).head(config.top_n).copy()
    if candidates.empty:
        return candidates

    candidates["action"] = "买入观察" if allow_buy else "仅观察"
    candidates["holding_plan"] = f"{max(20, config.horizon_days - 5)}-{config.horizon_days + 10}个交易日"
    stop_pct = np.clip(candidates["volatility_20"].fillna(0.03) * 3.0, 0.06, 0.14)
    candidates["stop_loss"] = candidates["close"] * (1 - stop_pct)
    candidates["position_limit"] = 1 / max(config.top_n, 1)
    candidates["risk_tags"] = candidates.apply(_risk_tags, axis=1)
    candidates["reason"] = candidates.apply(_reason, axis=1)
    return candidates[
        [
            "code",
            "name",
            "industry",
            "board",
            "action",
            "score",
            "score_rank",
            "close",
            "holding_plan",
            "position_limit",
            "stop_loss",
            "reason",
            "risk_tags",
        ]
    ].reset_index(drop=True)


def _risk_tags(row: pd.Series) -> str:
    tags = []
    if row["volatility_20"] > 0.035:
        tags.append("波动偏高")
    if row["pe_ttm"] > 60:
        tags.append("估值偏高")
    if row["amount_20"] < 120_000_000:
        tags.append("流动性一般")
    if row["industry_strength_20"] < 0:
        tags.append("行业偏弱")
    return "、".join(tags) if tags else "常规"


def _reason(row: pd.Series) -> str:
    reasons = []
    if row["ret_20"] > 0:
        reasons.append("20日动量为正")
    if row["ma_20_gap"] > 0:
        reasons.append("价格站上20日均线")
    if row["money_flow_20"] > 0:
        reasons.append("近20日资金净流入")
    if row["roe"] > 0.1:
        reasons.append("ROE较好")
    return "；".join(reasons[:3]) if reasons else "模型综合评分靠前"
