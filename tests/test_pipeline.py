from a_share_recommender.config import FEATURE_COLUMNS, StrategyConfig
from a_share_recommender.data_providers import DataRequest
from a_share_recommender.evaluator import evaluate_stock, normalize_stock_code
from a_share_recommender.features import build_feature_frame
from a_share_recommender.pipeline import run_pipeline
from a_share_recommender.sample_data import make_sample_market


def test_feature_frame_has_expected_columns():
    market = make_sample_market(n_stocks=12, n_days=180)
    features = build_feature_frame(market, horizon_days=20)
    for column in FEATURE_COLUMNS + ["future_return", "excess_return"]:
        assert column in features.columns


def test_future_return_uses_same_stock_future_close():
    market = make_sample_market(n_stocks=3, n_days=80)
    features = build_feature_frame(market, horizon_days=20)
    row = features[(features["code"] == "600000.SH")].iloc[10]
    code_prices = market[market["code"] == row["code"]].sort_values("date").reset_index(drop=True)
    current_idx = code_prices.index[code_prices["date"] == row["date"]][0]
    expected = code_prices.loc[current_idx + 20, "close"] / code_prices.loc[current_idx, "close"] - 1
    assert abs(row["future_return"] - expected) < 1e-12


def test_pipeline_returns_recommendations_and_metrics():
    result = run_pipeline(
        StrategyConfig(horizon_days=20, top_n=5, min_amount=1_000_000),
        data_request=DataRequest(force_sample=True),
    )
    assert result.metrics["periods"] > 0
    assert len(result.recommendations) <= 5
    assert {"code", "board", "score", "risk_tags", "reason"}.issubset(result.recommendations.columns)


def test_stock_code_normalization():
    assert normalize_stock_code("000001") == "000001.SZ"
    assert normalize_stock_code("600519") == "600519.SH"
    assert normalize_stock_code("300750.SZ") == "300750.SZ"


def test_stock_evaluation_for_existing_sample_stock():
    config = StrategyConfig(horizon_days=20, top_n=5, min_amount=1_000_000)
    result = run_pipeline(config, data_request=DataRequest(force_sample=True))
    evaluation = evaluate_stock("600000", result.market, result.latest_scored, config, result.gate_ok)
    assert evaluation.found
    assert evaluation.summary["代码"] == "600000.SH"
    assert evaluation.summary["市场板块"] == "上证主板"
    assert evaluation.conclusion in {"买入观察", "等回调", "仅观察", "不建议介入"}
    assert not evaluation.price_history.empty


def test_stock_evaluation_for_missing_code():
    config = StrategyConfig(horizon_days=20, top_n=5, min_amount=1_000_000)
    result = run_pipeline(config, data_request=DataRequest(force_sample=True))
    evaluation = evaluate_stock("999999", result.market, result.latest_scored, config, result.gate_ok)
    assert not evaluation.found
