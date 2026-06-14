from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import gate_passed, run_backtest
from .config import StrategyConfig
from .data_providers import DataRequest, ProviderRouter, ProviderStatus
from .features import build_feature_frame, latest_features
from .modeling import ModelResult, score_frame, train_model
from .recommend import make_recommendations
from .scoring import enrich_scores


@dataclass
class PipelineResult:
    provider_status: ProviderStatus
    model_result: ModelResult
    metrics: dict[str, float]
    gate_ok: bool
    gate_reasons: list[str]
    equity_curve: pd.DataFrame
    recommendations: pd.DataFrame
    market: pd.DataFrame
    latest_scored: pd.DataFrame
    data_date: pd.Timestamp
    availability: dict[str, str]


def run_pipeline(
    config: StrategyConfig,
    prefer_tushare: bool = False,
    tushare_token: str | None = None,
    data_request: DataRequest | None = None,
) -> PipelineResult:
    data_request = data_request or DataRequest()
    router = ProviderRouter(prefer_tushare=prefer_tushare, tushare_token=tushare_token)
    market, status = router.load_market(data_request)
    features = build_feature_frame(market, horizon_days=config.horizon_days)
    model_result = train_model(features)
    scored = enrich_scores(score_frame(model_result.model, features.dropna(subset=["excess_return"]).copy()), config)
    curve, metrics = run_backtest(scored, config, model_result.test_start)
    gate_ok, gate_reasons = gate_passed(metrics, config)

    latest = latest_features(features)
    latest_scored = enrich_scores(score_frame(model_result.model, latest), config)
    recommendations = make_recommendations(latest_scored, config, gate_ok, strict_rank=data_request.full_market_scan)
    return PipelineResult(
        provider_status=status,
        model_result=model_result,
        metrics=metrics,
        gate_ok=gate_ok,
        gate_reasons=gate_reasons,
        equity_curve=curve,
        recommendations=recommendations,
        market=market,
        latest_scored=latest_scored,
        data_date=pd.Timestamp(market["date"].max()),
        availability=router.availability(),
    )
