from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import _cap_industry, _tradable
from .config import StrategyConfig


def make_recommendations(latest_scored: pd.DataFrame, config: StrategyConfig, allow_buy: bool) -> pd.DataFrame:
    sort_column = "composite_score" if "composite_score" in latest_scored.columns else "score"
    candidates = _tradable(latest_scored, config).sort_values(sort_column, ascending=False)
    candidates = _cap_board(candidates.head(config.top_n * 10), config)
    candidates = _cap_industry(candidates.head(config.top_n * 8), config).head(config.top_n).copy()
    if candidates.empty:
        return candidates

    candidates["rating"] = _candidate_ratings(candidates)
    candidates["action"] = candidates.apply(lambda row: _action(row, config, allow_buy), axis=1)
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
            "rating",
            "win_probability",
            "composite_score",
            "trend_score",
            "risk_score",
            "fundamental_score",
            "industry_score",
            "market_regime_score",
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
    if row.get("market_regime_score", 0.5) < 0.45:
        tags.append("市场状态偏弱")
    if row.get("composite_score", 0.0) < 0.62:
        tags.append("综合评级不足")
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
    if row.get("rating") == "优选":
        reasons.append("候选等级优选")
    return "；".join(reasons[:4]) if reasons else "模型综合评分靠前"


def _action(row: pd.Series, config: StrategyConfig, allow_buy: bool) -> str:
    if not allow_buy:
        return "仅观察"
    if row.get("composite_score", 0) >= config.min_composite_score and row.get("market_regime_score", 0) >= config.min_market_regime_score and row.get("risk_score", 0) >= 0.45:
        return "买入观察"
    if row.get("trend_score", 0) >= 0.7 and row.get("risk_score", 0) >= 0.4:
        return "等回调"
    return "仅观察"


def _candidate_ratings(candidates: pd.DataFrame) -> pd.Series:
    scores = candidates["composite_score"] if "composite_score" in candidates.columns else candidates["score"]
    ranks = scores.rank(pct=True)
    labels = []
    for idx, rank in ranks.items():
        absolute = float(scores.loc[idx])
        if absolute >= 0.82 and rank >= 0.80:
            labels.append("优选")
        elif absolute >= 0.64:
            labels.append("备选")
        else:
            labels.append("观察")
    return pd.Series(labels, index=candidates.index)


def _cap_board(frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    if "board" not in frame.columns or frame.empty:
        return frame
    max_per_board = max(1, int(np.ceil(config.top_n * 0.45)))
    selected = []
    board_counts: dict[str, int] = {}
    for idx, row in frame.iterrows():
        board = str(row.get("board", "未知"))
        if board_counts.get(board, 0) >= max_per_board:
            continue
        selected.append(idx)
        board_counts[board] = board_counts.get(board, 0) + 1
        if len(selected) >= config.top_n:
            break
    if len(selected) < min(config.top_n, len(frame)):
        for idx in frame.index:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= min(config.top_n, len(frame)):
                break
    return frame.loc[selected]
