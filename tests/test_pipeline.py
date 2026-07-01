from pathlib import Path

import pandas as pd
import pytest

from a_share_recommender.config import FEATURE_COLUMNS, StrategyConfig
from a_share_recommender.data_providers import (
    DataRequest,
    _akshare_universe,
    _cache_satisfies_request,
    _compact_market_types,
    _load_latest_provider_cache,
    _merge_missing_symbol_history,
    _merge_refreshed_with_baseline,
    _normalize_akshare_tx_hist,
    _assert_full_market_universe,
    _select_request_symbols,
    _suffix_code,
    _symbols_from_baseline,
    _tx_symbol,
    known_stock_identity,
)
from a_share_recommender.data_providers import _core_fallback_universe, _filter_boards, _select_symbols_by_board
from a_share_recommender.evaluator import evaluate_stock, normalize_stock_code
from a_share_recommender.features import build_feature_frame, latest_features
from a_share_recommender.pipeline import run_pipeline
from a_share_recommender.sample_data import make_sample_market
from a_share_recommender.backtest import _cap_industry
from a_share_recommender.recommend import _cap_board, make_recommendations
from a_share_recommender.snapshot import APP_SNAPSHOT_FILES, load_snapshot, read_manifest, write_snapshot


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


def test_latest_features_keeps_each_stocks_latest_available_date():
    market = make_sample_market(n_stocks=3, n_days=180)
    missing_code = market["code"].drop_duplicates().iloc[-1]
    latest_date = market["date"].max()
    market = market[~((market["code"] == missing_code) & (market["date"] == latest_date))]
    features = build_feature_frame(market, horizon_days=20)

    latest = latest_features(features)

    assert latest["code"].nunique() == market["code"].nunique()
    assert latest.loc[latest["code"] == missing_code, "date"].iloc[0] < latest_date


def test_pipeline_returns_recommendations_and_metrics():
    result = run_pipeline(
        StrategyConfig(horizon_days=20, top_n=5, min_amount=1_000_000),
        data_request=DataRequest(force_sample=True),
    )
    assert result.metrics["periods"] > 0
    assert len(result.recommendations) <= 5
    assert {"code", "board", "score", "risk_tags", "reason"}.issubset(result.recommendations.columns)


def test_real_data_request_defaults_to_all_listed_symbols():
    request = DataRequest()
    assert request.max_symbols is None
    assert request.all_listed
    assert "北交所" in request.boards


def test_stock_code_normalization():
    assert normalize_stock_code("000001") == "000001.SZ"
    assert normalize_stock_code("600519") == "600519.SH"
    assert normalize_stock_code("300750.SZ") == "300750.SZ"
    assert normalize_stock_code("430047") == "430047.BJ"
    assert normalize_stock_code("832982.BJ") == "832982.BJ"
    assert known_stock_identity("689009")["board"] == "科创板"


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


def test_beijing_stock_exchange_identity_and_tencent_symbol():
    identity = known_stock_identity("430047")
    assert identity["code"] == "430047.BJ"
    assert identity["board"] == "北交所"
    assert _tx_symbol("430047") == "bj430047"


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


def test_full_market_cache_requires_the_requested_symbol_count():
    data = make_sample_market(n_stocks=12, n_days=40)

    assert not _cache_satisfies_request(data, DataRequest(max_symbols=13, full_market_scan=True))
    assert _cache_satisfies_request(data, DataRequest(max_symbols=12, full_market_scan=True))


def test_failed_symbols_are_filled_from_latest_real_history():
    stale = make_sample_market(n_stocks=3, n_days=40)
    codes = stale["code"].drop_duplicates().tolist()
    current = stale[stale["code"] == codes[0]].copy()

    merged, filled = _merge_missing_symbol_history(current, stale, [code.split(".")[0] for code in codes])

    assert filled == 2
    assert set(merged["code"]) == set(codes)


def test_incremental_refresh_keeps_baseline_history_and_prefers_current_rows():
    baseline = make_sample_market(n_stocks=3, n_days=40)
    codes = baseline["code"].drop_duplicates().tolist()
    current = baseline[baseline["code"] == codes[0]].tail(2).copy()
    current.loc[current.index[-1], "close"] = 999.0

    merged, filled = _merge_refreshed_with_baseline(
        current,
        baseline,
        [code.split(".")[0] for code in codes],
    )

    assert filled == 2
    assert set(merged["code"]) == set(codes)
    latest = merged[merged["code"] == codes[0]].sort_values("date").iloc[-1]
    assert latest["close"] == 999.0
    assert len(merged[merged["code"] == codes[0]]) == 40


def test_existing_full_market_baseline_keeps_its_stock_membership():
    baseline = make_sample_market(n_stocks=12, n_days=40)
    request = DataRequest(max_symbols=12, full_market_scan=True)

    symbols = _symbols_from_baseline(baseline, request)

    assert {_suffix_code(symbol) for symbol in symbols} == set(baseline["code"].unique())


def test_large_pool_request_limits_filtered_symbols():
    universe = _core_fallback_universe()
    boards = ("深证主板", "创业板")
    filtered = _filter_boards(universe, boards)
    request = DataRequest(max_symbols=5, boards=boards, full_market_scan=True)
    selected = _select_request_symbols(filtered, request)
    assert len(selected) == request.max_symbols
    assert len(selected) < filtered["code"].nunique()


def test_all_listed_request_keeps_every_filtered_symbol():
    universe = _core_fallback_universe()
    request = DataRequest(all_listed=True, max_symbols=None)

    selected = _select_request_symbols(universe, request)

    assert selected == sorted(universe["code"].astype(str).tolist())


def test_all_listed_universe_requires_every_requested_board():
    universe = pd.DataFrame(
        {
            "code": [f"{i:06d}" for i in range(1000)],
            "board": ["上证主板"] * 1000,
        }
    )
    request = DataRequest(all_listed=True, full_market_scan=True)

    with pytest.raises(RuntimeError, match="当前缺少"):
        _assert_full_market_universe(universe, request, "code")


def test_akshare_universe_fetches_star_market_separately(tmp_path):
    class FakeAkshare:
        def __init__(self):
            self.sh_segments = []

        def stock_info_sh_name_code(self, symbol):
            self.sh_segments.append(symbol)
            start, count = (600000, 500) if symbol == "主板A股" else (688000, 100)
            return pd.DataFrame(
                {
                    "证券代码": [f"{start + i:06d}" for i in range(count)],
                    "证券简称": [f"沪股{i}" for i in range(count)],
                    "上市日期": ["2020-01-01"] * count,
                }
            )

        def stock_info_sz_name_code(self):
            return pd.DataFrame(
                {
                    "A股代码": [f"{i:06d}" for i in range(500)] + [f"{300000 + i:06d}" for i in range(500)],
                    "A股简称": [f"深股{i}" for i in range(1000)],
                    "所属行业": ["制造业"] * 1000,
                    "板块": ["主板"] * 500 + ["创业板"] * 500,
                    "A股上市日期": ["2020-01-01"] * 1000,
                }
            )

        def stock_info_bj_name_code(self):
            return pd.DataFrame(
                {
                    "证券代码": [f"{830000 + i:06d}" for i in range(100)],
                    "证券简称": [f"北股{i}" for i in range(100)],
                    "上市日期": ["2020-01-01"] * 100,
                }
            )

    fake = FakeAkshare()
    universe = _akshare_universe(fake, tmp_path, force_refresh=True)

    assert fake.sh_segments == ["主板A股", "科创板"]
    assert universe["board"].value_counts()["科创板"] >= 100
    assert universe["board"].value_counts()["北交所"] >= 100


def test_market_numeric_columns_are_compacted():
    market = make_sample_market(n_stocks=3, n_days=40)

    compact = _compact_market_types(market)

    assert str(compact["close"].dtype) == "float32"
    assert str(compact["list_days"].dtype) == "int32"


def test_full_market_rejects_core_only_universe():
    request = DataRequest(full_market_scan=True)
    with pytest.raises(RuntimeError, match="大池排名需要完整"):
        _assert_full_market_universe(_core_fallback_universe(), request, "code")


def test_full_market_recommendations_keep_score_order():
    frame = make_sample_market(n_stocks=12, n_days=40).groupby("code").tail(1).copy().reset_index(drop=True)
    frame["industry"] = "同一行业"
    frame["board"] = "上证主板"
    frame["composite_score"] = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.50, 0.49, 0.48, 0.47, 0.46, 0.45]
    frame["score"] = frame["composite_score"]
    frame["score_rank"] = frame["composite_score"].rank(pct=True)
    frame["win_probability"] = frame["composite_score"]
    frame["trend_score"] = frame["composite_score"]
    frame["risk_score"] = 0.8
    frame["fundamental_score"] = 0.8
    frame["industry_score"] = 0.8
    frame["market_regime_score"] = 0.8
    frame["amount_20"] = 100_000_000
    frame["is_st"] = False
    frame["suspended"] = False
    frame["list_days"] = 999
    frame["volatility_20"] = 0.02
    frame["pe_ttm"] = 20
    frame["industry_strength_20"] = 0.1
    frame["ret_20"] = 0.05
    frame["ma_20_gap"] = 0.03
    frame["money_flow_20"] = 1
    frame["roe"] = 0.12

    recommendations = make_recommendations(
        frame,
        StrategyConfig(top_n=5, min_amount=1_000_000),
        allow_buy=True,
        strict_rank=True,
    )

    assert recommendations["composite_score"].tolist() == sorted(recommendations["composite_score"], reverse=True)
    assert recommendations["market_rank"].tolist() == [1, 2, 3, 4, 5]


def test_ui_does_not_use_invalid_percent_sprintf_format():
    ui_source = Path("src/a_share_recommender/ui.py").read_text(encoding="utf-8")
    assert 'format="%.1%"' not in ui_source


def test_snapshot_round_trip_preserves_public_result(tmp_path):
    config = StrategyConfig(horizon_days=20, top_n=5, min_amount=1_000_000)
    result = run_pipeline(config, data_request=DataRequest(force_sample=True))
    destination = write_snapshot(result, tmp_path / "snapshot", config)

    loaded = load_snapshot(destination)

    assert loaded.provider_status.mode.startswith("snapshot-")
    assert loaded.data_date == result.data_date
    assert loaded.gate_ok == result.gate_ok
    assert loaded.metrics == result.metrics
    expected_codes = result.recommendations.sort_values("composite_score", ascending=False)["code"].tolist()
    assert loaded.recommendations["code"].tolist() == expected_codes
    assert loaded.recommendations["composite_score"].is_monotonic_decreasing
    assert loaded.market["code"].nunique() == result.market["code"].nunique()
    manifest = read_manifest(destination)
    assert manifest["schema_version"] == 2
    assert manifest["market_symbol_count"] == result.market["code"].nunique()
    assert manifest["scored_symbol_count"] == result.latest_scored["code"].nunique()
    assert sum(manifest["board_counts"].values()) == result.latest_scored["code"].nunique()
    assert (destination / "provider_market.parquet").exists()


def test_public_snapshot_load_does_not_require_builder_fallback_file(tmp_path):
    config = StrategyConfig(horizon_days=20, top_n=5, min_amount=1_000_000)
    result = run_pipeline(config, data_request=DataRequest(force_sample=True))
    source = write_snapshot(result, tmp_path / "source", config)
    app_only = tmp_path / "app-only"
    app_only.mkdir()
    (app_only / "manifest.json").write_bytes((source / "manifest.json").read_bytes())
    for name in APP_SNAPSHOT_FILES:
        (app_only / name).write_bytes((source / name).read_bytes())

    loaded = load_snapshot(app_only)

    assert loaded.latest_scored["code"].nunique() == result.latest_scored["code"].nunique()


def test_snapshot_rejects_incomplete_expected_symbol_coverage(tmp_path):
    config = StrategyConfig(horizon_days=20, top_n=5, min_amount=1_000_000)
    result = run_pipeline(config, data_request=DataRequest(force_sample=True))

    with pytest.raises(RuntimeError, match="快照覆盖未达标"):
        write_snapshot(result, tmp_path / "snapshot", config, expected_symbols=3000)


def test_snapshot_rejects_modified_file(tmp_path):
    config = StrategyConfig(horizon_days=20, top_n=5, min_amount=1_000_000)
    result = run_pipeline(config, data_request=DataRequest(force_sample=True))
    destination = write_snapshot(result, tmp_path / "snapshot", config)
    with (destination / "result.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")

    with pytest.raises(RuntimeError, match="校验失败"):
        load_snapshot(destination)
