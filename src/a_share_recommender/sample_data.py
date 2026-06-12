from __future__ import annotations

import numpy as np
import pandas as pd


INDUSTRIES = [
    "电子",
    "医药生物",
    "电力设备",
    "计算机",
    "食品饮料",
    "有色金属",
    "机械设备",
    "银行",
]


def make_sample_market(n_stocks: int = 90, n_days: int = 900, seed: int = 42) -> pd.DataFrame:
    """Generate deterministic market-like data for offline demos and tests."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    market_cycle = rng.normal(0.00015, 0.007, n_days).cumsum()

    rows: list[pd.DataFrame] = []
    for i in range(n_stocks):
        code = f"{600000 + i:06d}.SH" if i % 2 == 0 else f"{300000 + i:06d}.SZ"
        industry = INDUSTRIES[i % len(INDUSTRIES)]
        quality = rng.normal(0, 1)
        momentum_beta = rng.uniform(0.4, 1.4)
        base_price = rng.uniform(8, 90)
        noise = rng.normal(0, 0.014, n_days)
        industry_cycle = np.sin(np.linspace(0, 8, n_days) + i % len(INDUSTRIES)) * 0.0008
        daily_ret = 0.00005 + 0.00008 * quality + momentum_beta * np.diff(
            np.r_[market_cycle[0], market_cycle]
        ) + industry_cycle + noise
        close = base_price * np.exp(np.cumsum(daily_ret))
        open_ = close * (1 + rng.normal(0, 0.006, n_days))
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.018, n_days))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.018, n_days))
        volume = rng.lognormal(15.2 + quality * 0.08, 0.45, n_days)
        amount = volume * close
        turnover = rng.uniform(0.4, 6.5, n_days) + quality * 0.15
        money_flow = (daily_ret + rng.normal(0, 0.01, n_days)) * amount * 0.05
        pe = np.clip(28 - quality * 5 + rng.normal(0, 2.5, n_days), 4, 120)
        roe = np.clip(0.08 + quality * 0.025 + rng.normal(0, 0.01, n_days), -0.2, 0.35)
        profit_growth = np.clip(0.12 + quality * 0.08 + rng.normal(0, 0.04, n_days), -0.8, 1.5)
        market_cap = np.exp(rng.normal(23.5, 0.9))

        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "code": code,
                    "name": f"样例{i + 1:03d}",
                    "industry": industry,
                    "board": _sample_board(code),
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "amount": amount,
                    "turnover_rate": turnover,
                    "money_flow": money_flow,
                    "pe_ttm": pe,
                    "roe": roe,
                    "net_profit_growth": profit_growth,
                    "market_cap": market_cap,
                    "is_st": i % 37 == 0,
                    "list_days": n_days + i * 7,
                    "suspended": False,
                }
            )
        )

    data = pd.concat(rows, ignore_index=True)
    return data.sort_values(["date", "code"]).reset_index(drop=True)


def _sample_board(code: str) -> str:
    if code.startswith("688"):
        return "科创板"
    if code.startswith("300"):
        return "创业板"
    if code.endswith(".SH"):
        return "上证主板"
    return "深证主板"
