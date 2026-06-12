from __future__ import annotations

import plotly.express as px
import streamlit as st

from .config import StrategyConfig
from .pipeline import run_pipeline


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
        prefer_tushare = st.checkbox("优先使用 Tushare Pro", value=False)
        tushare_token = st.text_input("Tushare token", type="password")
        run = st.button("生成今日推荐", type="primary")

    config = StrategyConfig(
        horizon_days=horizon,
        top_n=top_n,
        min_amount=float(min_amount),
        transaction_cost=float(transaction_cost),
    )

    if "result" not in st.session_state or run:
        with st.spinner("正在更新数据、训练模型并回测..."):
            st.session_state["result"] = run_pipeline(
                config,
                prefer_tushare=prefer_tushare,
                tushare_token=tushare_token or None,
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

    tab_rec, tab_bt, tab_data = st.tabs(["今日推荐", "回测证据", "数据源状态"])

    with tab_rec:
        st.subheader("推荐列表")
        st.dataframe(
            result.recommendations,
            use_container_width=True,
            column_config={
                "score": st.column_config.NumberColumn("模型分", format="%.4f"),
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
                "训练行数": result.model_result.train_rows,
                "测试行数": result.model_result.test_rows,
                "训练截止": result.model_result.train_end.strftime("%Y-%m-%d"),
                "测试开始": result.model_result.test_start.strftime("%Y-%m-%d"),
            }
        )

