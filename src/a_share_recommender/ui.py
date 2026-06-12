from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from .config import StrategyConfig
from .data_providers import DataRequest
from .evaluator import evaluate_stock, normalize_stock_code
from .pipeline import run_pipeline


BOARD_OPTIONS = ["上证主板", "深证主板", "创业板", "科创板"]


def render_app() -> None:
    st.set_page_config(page_title="A股波段推荐", layout="wide")
    st.title("A股波段预测推荐")
    st.caption("本地研究辅助工具；不构成投资建议，不承诺收益或固定胜率。")

    with st.sidebar:
        st.header("策略设置")
        horizon = st.slider("目标持有期（交易日）", 20, 40, 30, 5)
        top_n = st.slider("推荐数量上限", 3, 20, 10)
        min_amount = st.number_input("20日平均成交额下限（万元）", 1000, 100000, 8000, 1000) * 10000
        transaction_cost = st.number_input("单轮交易成本", 0.0, 0.02, 0.003, 0.001, format="%.3f")

        st.header("个股评估")
        evaluation_code = st.text_input("股票代码", value="", placeholder="例如 000001 或 600519")
        normalized_evaluation_code = normalize_stock_code(evaluation_code)
        if evaluation_code and normalized_evaluation_code:
            st.caption(f"将评估：{normalized_evaluation_code}")

        st.header("数据设置")
        force_sample = st.checkbox("演示模式（不拉真实行情）", value=False)
        prefer_tushare = st.checkbox("优先使用 Tushare Pro", value=False, disabled=force_sample)
        tushare_token = st.text_input("Tushare token", type="password", disabled=force_sample)
        boards = st.multiselect("市场板块", BOARD_OPTIONS, default=BOARD_OPTIONS, disabled=force_sample)
        max_symbols = st.slider("真实数据股票数量", 5, 80, 30, 5, disabled=force_sample)
        history_years = st.slider("历史数据年限", 2, 6, 4, 1, disabled=force_sample)
        use_finance = st.checkbox("启用财务增强（较慢）", value=True, disabled=force_sample)
        force_refresh = st.checkbox("忽略今日缓存并重新拉取", value=False, disabled=force_sample)
        run = st.button("生成今日推荐", type="primary")

    config = StrategyConfig(
        horizon_days=horizon,
        top_n=top_n,
        min_amount=float(min_amount),
        transaction_cost=float(transaction_cost),
    )
    extra_symbols = (normalized_evaluation_code,) if normalized_evaluation_code and not force_sample else ()
    data_request = DataRequest(
        max_symbols=max_symbols,
        history_years=history_years,
        use_finance=use_finance,
        force_sample=force_sample,
        force_refresh=force_refresh,
        extra_symbols=extra_symbols,
        boards=tuple(boards or BOARD_OPTIONS),
    )

    cache_key = (
        horizon,
        top_n,
        min_amount,
        transaction_cost,
        evaluation_code.strip(),
        prefer_tushare,
        bool(tushare_token),
        tuple(boards or BOARD_OPTIONS),
        max_symbols,
        history_years,
        use_finance,
        force_sample,
        force_refresh,
    )
    if st.session_state.get("cache_key") != cache_key:
        st.session_state.pop("result", None)
        st.session_state["cache_key"] = cache_key

    if "result" not in st.session_state or run:
        with st.spinner("正在更新数据、训练模型并回测。真实数据首次拉取可能需要数分钟..."):
            st.session_state["result"] = run_pipeline(
                config,
                prefer_tushare=prefer_tushare,
                tushare_token=tushare_token or None,
                data_request=data_request,
            )

    result = st.session_state["result"]
    st.info(result.provider_status.message)

    cols = st.columns(5)
    cols[0].metric("数据日期", result.data_date.strftime("%Y-%m-%d"))
    cols[1].metric("样本外胜率", f"{result.metrics['sample_win_rate']:.1%}")
    cols[2].metric("总收益", f"{result.metrics['total_return']:.1%}")
    cols[3].metric("最大回撤", f"{result.metrics['max_drawdown']:.1%}")
    cols[4].metric("收益回撤比", f"{result.metrics['return_drawdown_ratio']:.2f}")

    if result.gate_ok:
        st.success("回测门槛通过：今日推荐允许作为买入观察清单。")
    else:
        st.warning("回测门槛未通过：今日清单仅用于观察。原因：" + "；".join(result.gate_reasons))

    tab_rec, tab_eval, tab_bt, tab_data = st.tabs(["今日推荐", "个股评估", "回测证据", "数据源状态"])

    with tab_rec:
        st.subheader("推荐列表")
        st.dataframe(
            result.recommendations,
            use_container_width=True,
            column_config={
                "score": st.column_config.NumberColumn("模型分", format="%.4f"),
                "board": st.column_config.TextColumn("市场板块"),
                "score_rank": st.column_config.ProgressColumn("分位", min_value=0, max_value=1),
                "close": st.column_config.NumberColumn("收盘价", format="%.2f"),
                "position_limit": st.column_config.NumberColumn("仓位上限", format="%.1%"),
                "stop_loss": st.column_config.NumberColumn("止损价", format="%.2f"),
            },
        )
        st.download_button(
            "导出推荐 CSV",
            result.recommendations.to_csv(index=False).encode("utf-8-sig"),
            file_name="a_share_recommendations.csv",
            mime="text/csv",
        )

    with tab_eval:
        _render_stock_evaluation(evaluation_code, result, config)

    with tab_bt:
        st.subheader("样本外资金曲线")
        if result.equity_curve.empty:
            st.write("没有足够回测记录。")
        else:
            fig = px.line(
                result.equity_curve,
                x="date",
                y=["equity", "benchmark_equity"],
                labels={"value": "净值", "variable": "序列"},
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(result.equity_curve.tail(20), use_container_width=True)

    with tab_data:
        st.subheader("可用数据源")
        st.table(result.availability)
        st.write(
            {
                "数据模式": result.provider_status.mode,
                "数据行数": result.provider_status.rows,
                "训练行数": result.model_result.train_rows,
                "测试行数": result.model_result.test_rows,
                "训练截止": result.model_result.train_end.strftime("%Y-%m-%d"),
                "测试开始": result.model_result.test_start.strftime("%Y-%m-%d"),
            }
        )


def _render_stock_evaluation(raw_code: str, result, config: StrategyConfig) -> None:
    st.subheader("个股评估")
    if not raw_code.strip():
        st.write("在左侧输入股票代码后，这里会显示模型评分、技术面、基本面、风险标签和近一年走势。")
        return

    evaluation = evaluate_stock(raw_code, result.market, result.latest_scored, config, result.gate_ok)
    if not evaluation.found:
        st.error(evaluation.explanation)
        return

    summary = evaluation.summary
    cols = st.columns(6)
    cols[0].metric("结论", evaluation.conclusion)
    cols[1].metric("股票", f"{summary['名称']} {summary['代码']}")
    cols[2].metric("模型分位", f"{summary['股票池分位']:.1%}")
    cols[3].metric("收盘价", f"{summary['最新收盘价']:.2f}")
    cols[4].metric("市场板块", str(summary["市场板块"]))
    cols[5].metric("行业", str(summary["行业"]))

    if evaluation.conclusion == "买入观察":
        st.success(evaluation.explanation)
    elif evaluation.conclusion == "仅观察":
        st.warning(evaluation.explanation)
    else:
        st.error(evaluation.explanation)

    signal_frame = _signals_to_frame(evaluation.signals)
    st.dataframe(signal_frame, use_container_width=True, hide_index=True)

    if evaluation.risks:
        st.write("风险标签：" + "；".join(evaluation.risks))
    else:
        st.write("风险标签：常规")

    price = evaluation.price_history[["date", "close", "ma20", "ma60"]].copy()
    price = price.rename(columns={"close": "收盘价", "ma20": "20日均线", "ma60": "60日均线"})
    fig = px.line(price, x="date", y=["收盘价", "20日均线", "60日均线"], labels={"value": "价格", "variable": "序列"})
    st.plotly_chart(fig, use_container_width=True)


def _signals_to_frame(signals: dict[str, float]) -> pd.DataFrame:
    percent_keys = {"20日收益", "60日收益", "120日收益", "20日均线偏离", "60日均线偏离", "20日波动", "ROE", "净利润增长率", "行业20日强度"}
    rows = []
    for key, value in signals.items():
        if pd.isna(value):
            display = "缺失"
        elif key in percent_keys:
            display = f"{value:.2%}"
        elif key == "20日平均成交额":
            display = f"{value / 10000:.0f} 万元"
        elif key == "PE TTM":
            display = f"{value:.2f}"
        else:
            display = f"{value:.4f}"
        rows.append({"指标": key, "数值": display})
    return pd.DataFrame(rows)
