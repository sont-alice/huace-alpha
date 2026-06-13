from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StrategyConfig


def run_backtest(scored: pd.DataFrame, config: StrategyConfig, start_date: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, float]]:
    test = scored[(scored["date"] >= start_date) & scored["future_return"].notna()].copy()
    test = _tradable(test, config)
    rebalance_dates = sorted(test["date"].drop_duplicates())[:: config.horizon_days]
    rows = []

    for date in rebalance_dates:
        sort_column = "composite_score" if "composite_score" in test.columns else "score"
        day = test[test["date"] == date].sort_values(sort_column, ascending=False)
        picks = _cap_industry(day.head(config.top_n * 3), config).head(config.top_n)
        if picks.empty:
            continue
        gross = picks["future_return"].mean()
        benchmark = day["future_return"].median()
        net = gross - config.transaction_cost
        rows.append(
            {
                "date": date,
                "picks": len(picks),
                "gross_return": gross,
                "net_return": net,
                "benchmark_return": benchmark,
                "excess_return": net - benchmark,
                "win": float(net > benchmark),
            }
        )

    curve = pd.DataFrame(rows)
    if curve.empty:
        return curve, _empty_metrics()

    curve["equity"] = (1 + curve["net_return"]).cumprod()
    curve["benchmark_equity"] = (1 + curve["benchmark_return"]).cumprod()
    running_max = curve["equity"].cummax()
    drawdown = curve["equity"] / running_max - 1
    total_return = curve["equity"].iloc[-1] - 1
    max_drawdown = abs(drawdown.min())
    metrics = {
        "sample_win_rate": float(curve["win"].mean()),
        "benchmark_win_rate": 0.5,
        "total_return": float(total_return),
        "benchmark_total_return": float(curve["benchmark_equity"].iloc[-1] - 1),
        "max_drawdown": float(max_drawdown),
        "return_drawdown_ratio": float(total_return / max(max_drawdown, 1e-9)),
        "periods": float(len(curve)),
    }
    return curve, metrics


def gate_passed(metrics: dict[str, float], config: StrategyConfig) -> tuple[bool, list[str]]:
    reasons = []
    if metrics["periods"] < 6:
        reasons.append("样本外调仓期数不足")
    if metrics["sample_win_rate"] < metrics["benchmark_win_rate"] + config.min_sample_win_rate_edge:
        reasons.append("样本外胜率未超过基准门槛")
    if metrics["return_drawdown_ratio"] < config.min_return_drawdown_ratio:
        reasons.append("收益回撤比未达标")
    if metrics["max_drawdown"] > config.max_drawdown_limit:
        reasons.append("最大回撤超过上限")
    return not reasons, reasons


def _tradable(frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    return frame[
        (~frame["is_st"])
        & (~frame["suspended"])
        & (frame["list_days"] >= 180)
        & (frame["amount_20"] >= config.min_amount)
    ].copy()


def _cap_industry(frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    max_per_industry = max(1, int(np.ceil(config.top_n * config.max_industry_weight)))
    return frame.groupby("industry", group_keys=False).head(max_per_industry)


def _empty_metrics() -> dict[str, float]:
    return {
        "sample_win_rate": 0.0,
        "benchmark_win_rate": 0.5,
        "total_return": 0.0,
        "benchmark_total_return": 0.0,
        "max_drawdown": 1.0,
        "return_drawdown_ratio": 0.0,
        "periods": 0.0,
    }
