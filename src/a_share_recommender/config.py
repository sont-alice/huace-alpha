from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyConfig:
    horizon_days: int = 30
    top_n: int = 10
    min_amount: float = 80_000_000
    max_industry_weight: float = 0.3
    transaction_cost: float = 0.003
    min_sample_win_rate_edge: float = 0.05
    min_return_drawdown_ratio: float = 1.0
    max_drawdown_limit: float = 0.25
    min_composite_score: float = 0.72
    min_market_regime_score: float = 0.45


FEATURE_COLUMNS = [
    "ret_20",
    "ret_60",
    "ret_120",
    "ma_20_gap",
    "ma_60_gap",
    "volatility_20",
    "turnover_20",
    "amount_20",
    "money_flow_20",
    "industry_strength_20",
    "pe_ttm",
    "roe",
    "net_profit_growth",
    "market_cap_log",
    "market_return_20",
    "market_breadth_20",
    "market_above_ma20",
]
