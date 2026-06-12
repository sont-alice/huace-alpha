from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import gate_passed, run_backtest
from .config import StrategyConfig
from .data_providers import ProviderRouter, ProviderStatus
from .features import build_feature_frame, latest_features
from .modeling import ModelResult, score_frame, train_model
from .recommend import make_recommendations


@dataclass
class PipelineResult:
    provider_status: ProviderStatus
    model_result: ModelResult
    metrics: dict[str, float]
    gate_ok: bool
    gate_reasons: list[str]
    equity_curve: pd.DataFrame
    recommendations: pd.DataFrame
    data_date: pd.Timestamp
    availability: dict[str, str]


def run_pipeline(config: StrategyConfig, prefer_tushare: bool = False, tushare_token: str | None = None) -> PipelineResult:
    router = ProviderRouter(prefer_tushare=prefer_tushare, tushare_token=tushare_token)
    market, status = router.load_market()
    features = build_feature_frame(market, horizon_days=config.horizon_days)
    model_result = train_model(features)
    scored = score_frame(model_result.model, features.dropna(subset=["excess_return"]).copy())
    curve, metrics = run_backtest(scored, config, model_result.test_start)
    gate_ok, gate_reasons = gate_passed(metrics, config)

    latest = latest_features(features)
    latest_scored = score_frame(model_result.model, latest)
    recommendations = make_recommendations(latest_scored, config, gate_ok)
    return PipelineResult(
        provider_status=status,
        model_result=model_result,
        metrics=metrics,
        gate_ok=gate_ok,
        gate_reasons=gate_reasons,
        equity_curve=curve,
        recommendations=recommendations,
        data_date=pd.Timestamp(market["date"].max()),
        availability=router.availability(),
    )

