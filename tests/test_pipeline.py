import pandas as pd

from a_share_recommender.config import FEATURE_COLUMNS, StrategyConfig
from a_share_recommender.data_providers import (
    DataRequest,
    _cache_satisfies_request,
    _load_latest_provider_cache,
    _normalize_akshare_tx_hist,
    _tx_symbol,
    known_stock_identity,
)
from a_share_recommender.data_providers import _core_fallback_universe, _filter_boards, _select_symbols_by_board
from a_share_recommender.evaluator import evaluate_stock, normalize_stock_code
from a_share_recommender.features import build_feature_frame
from a_share_recommender.pipeline import run_pipeline
from a_share_recommender.sample_data import make_sample_market
from a_share_recommender.backtest import _cap_industry
from a_share_recommender.recommend import _cap_board


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


def test_known_stock_identity_for_baotong_technology():
    identity = known_stock_identity("300031")
    assert identity["code"] == "300031.SZ"
    assert identity["name"] == "宝通科技"
    assert identity["board"] == "创业板"


def test_tx_symbol_and_normalization_for_baotong_technology():
    assert _tx_symbol("300031") == "sz300031"
    raw = pd.DataFrame(
        {
            "date": ["2026-06-12"],
            "open": [25.94],
            "close": [24.92],
            "high": [26.02],
            "low": [24.74],
            "amount": [245369.0],
        }
    )
    meta = pd.DataFrame([{"name": "宝通科技", "industry": "I 信息技术", "board": "创业板"}])
    normalized = _normalize_akshare_tx_hist(raw, "300031", meta)
    row = normalized.iloc[0]
    assert row["code"] == "300031.SZ"
    assert row["name"] == "宝通科技"
    assert row["industry"] == "I 信息技术"
    assert row["board"] == "创业板"
    assert row["close"] == 24.92


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


def test_stock_evaluation_for_known_missing_stock_explains_missing_market_data():
    config = StrategyConfig(horizon_days=20, top_n=5, min_amount=1_000_000)
    result = run_pipeline(config, data_request=DataRequest(force_sample=True))
    market = result.market[result.market["code"] != "300031.SZ"]
    latest = result.latest_scored[result.latest_scored["code"] != "300031.SZ"]
    evaluation = evaluate_stock("300031", market, latest, config, result.gate_ok)
    assert not evaluation.found
    assert "宝通科技" in evaluation.explanation
    assert "缺少它的行情" in evaluation.explanation


def test_industry_cap_fills_to_top_n_when_candidates_exist():
    frame = make_sample_market(n_stocks=18, n_days=40).groupby("code").tail(1).copy()
    frame["industry"] = ["同一行业"] * 12 + ["行业B"] * 6
    config = StrategyConfig(top_n=10, max_industry_weight=0.3)
    selected = _cap_industry(frame, config)
    assert len(selected) == 10


def test_symbol_selection_balances_selected_boards():
    universe = _core_fallback_universe()
    filtered = _filter_boards(universe, ("上证主板", "深证主板", "创业板", "科创板"))
    selected = _select_symbols_by_board(filtered, 16, ("上证主板", "深证主板", "创业板", "科创板"))
    selected_boards = filtered[filtered["code"].isin(selected)]["board"].value_counts().to_dict()
    assert selected_boards["上证主板"] > 0
    assert selected_boards["深证主板"] > 0
    assert selected_boards["创业板"] > 0
    assert selected_boards["科创板"] > 0


def test_board_cap_limits_single_board_concentration():
    frame = make_sample_market(n_stocks=18, n_days=40).groupby("code").tail(1).copy()
    frame["board"] = ["上证主板"] * 12 + ["深证主板"] * 3 + ["创业板"] * 3
    config = StrategyConfig(top_n=10)
    selected = _cap_board(frame, config).head(10)
    assert len(selected) == 10
    assert selected["board"].value_counts()["上证主板"] <= 5


def test_stale_cache_merges_multiple_board_files(tmp_path):
    base = make_sample_market(n_stocks=12, n_days=40)
    boards = ["上证主板", "深证主板", "创业板"]
    for idx, board in enumerate(boards):
        part = base[base["board"] == board].copy()
        part.to_parquet(tmp_path / f"akshare_{idx}.parquet", index=False)
    request = DataRequest(max_symbols=9)
    merged = _load_latest_provider_cache(tmp_path, "akshare", request)
    assert merged is not None
    assert merged["board"].nunique() >= 2


def test_narrow_cache_does_not_satisfy_multi_board_request():
    data = make_sample_market(n_stocks=3, n_days=40)
    data["board"] = "深证主板"
    assert not _cache_satisfies_request(data, DataRequest(max_symbols=30))
