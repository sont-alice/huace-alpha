from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import StrategyConfig


@dataclass(frozen=True)
class StockEvaluation:
    found: bool
    code: str
    summary: dict[str, str | float]
    signals: dict[str, float]
    risks: list[str]
    conclusion: str
    explanation: str
    price_history: pd.DataFrame


def normalize_stock_code(raw: str) -> str:
    text = str(raw).strip().upper()
    if not text:
        return ""
    if "." in text:
        digits = text.split(".")[0]
    else:
        digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return ""
    digits = digits.zfill(6)
    if digits.startswith(("6", "9", "688")):
        return f"{digits}.SH"
    return f"{digits}.SZ"


def evaluate_stock(
    raw_code: str,
    market: pd.DataFrame,
    latest_scored: pd.DataFrame,
    config: StrategyConfig,
    gate_ok: bool,
) -> StockEvaluation:
    code = normalize_stock_code(raw_code)
    if not code:
        return _not_found(raw_code, "请输入 6 位 A 股代码，例如 000001 或 600519。")

    latest = latest_scored.loc[latest_scored["code"] == code]
    history = market.loc[market["code"] == code].sort_values("date").copy()
    if latest.empty or history.empty:
        return _not_found(code, "当前数据集中没有这只股票。请增大“真实数据股票数量”或确认代码是否正确。")

    row = latest.iloc[0]
    history["ma20"] = history["close"].rolling(20).mean()
    history["ma60"] = history["close"].rolling(60).mean()

    risks = _risk_list(row)
    conclusion = _conclusion(row, risks, gate_ok)
    explanation = _explanation(row, risks, gate_ok)

    summary = {
        "代码": code,
        "名称": row["name"],
        "行业": row["industry"],
        "市场板块": row.get("board", "未知"),
        "最新收盘价": float(row["close"]),
        "模型分": float(row["score"]),
        "股票池分位": float(row["score_rank"]),
        "建议持有期": f"{max(20, config.horizon_days - 5)}-{config.horizon_days + 10}个交易日",
    }
    signals = {
        "20日收益": _float(row["ret_20"]),
        "60日收益": _float(row["ret_60"]),
        "120日收益": _float(row["ret_120"]),
        "20日均线偏离": _float(row["ma_20_gap"]),
        "60日均线偏离": _float(row["ma_60_gap"]),
        "20日波动": _float(row["volatility_20"]),
        "20日平均成交额": _float(row["amount_20"]),
        "PE TTM": _float(row["pe_ttm"]),
        "ROE": _float(row["roe"]),
        "净利润增长率": _float(row["net_profit_growth"]),
        "行业20日强度": _float(row["industry_strength_20"]),
    }
    return StockEvaluation(True, code, summary, signals, risks, conclusion, explanation, history.tail(260))


def _not_found(code: str, message: str) -> StockEvaluation:
    return StockEvaluation(False, str(code), {}, {}, [message], "无法评估", message, pd.DataFrame())


def _risk_list(row: pd.Series) -> list[str]:
    risks = []
    if bool(row.get("is_st", False)):
        risks.append("ST 或退市风险标签")
    if row.get("list_days", 9999) < 180:
        risks.append("上市时间不足 180 天")
    if row.get("amount_20", 0) < 120_000_000:
        risks.append("20日平均成交额偏低")
    if row.get("volatility_20", 0) > 0.035:
        risks.append("短期波动偏高")
    if row.get("industry_strength_20", 0) < 0:
        risks.append("行业短期相对偏弱")
    if row.get("pe_ttm", 0) > 60:
        risks.append("估值偏高")
    if row.get("ma_20_gap", 0) < -0.03:
        risks.append("价格低于20日均线较多")
    return risks


def _conclusion(row: pd.Series, risks: list[str], gate_ok: bool) -> str:
    rank = row.get("score_rank", 0)
    hard_risk = any("ST" in risk or "上市时间" in risk for risk in risks)
    if hard_risk or rank < 0.35:
        return "不建议介入"
    if gate_ok and rank >= 0.8 and len(risks) <= 2:
        return "买入观察"
    return "仅观察"


def _explanation(row: pd.Series, risks: list[str], gate_ok: bool) -> str:
    parts = []
    rank = row.get("score_rank", 0)
    parts.append(f"模型分位为 {rank:.1%}")
    if row.get("ret_20", 0) > 0:
        parts.append("20日动量为正")
    if row.get("ma_20_gap", 0) > 0:
        parts.append("价格站上20日均线")
    if row.get("roe", 0) > 0.1:
        parts.append("ROE较好")
    if not gate_ok:
        parts.append("当前全局策略回测门槛未通过，因此不升级为买入观察")
    if risks:
        parts.append("主要风险：" + "、".join(risks[:4]))
    return "；".join(parts)


def _float(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)
