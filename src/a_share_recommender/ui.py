from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from .config import StrategyConfig
from .data_providers import DataRequest
from .evaluator import evaluate_stock, normalize_stock_code
from .pipeline import run_pipeline
from .snapshot import load_configured_snapshot, public_snapshot_mode


BOARD_OPTIONS = ["上证主板", "深证主板", "创业板", "科创板"]
APP_STATE_VERSION = "large-pool-ranking-v1"


def render_app() -> None:
    st.set_page_config(page_title="A股波段推荐", layout="wide")
    _apply_commercial_theme()
    if st.session_state.get("app_state_version") != APP_STATE_VERSION:
        st.session_state.pop("result", None)
        st.session_state.pop("run_error", None)
        st.session_state.pop("cache_key", None)
        st.session_state["app_state_version"] = APP_STATE_VERSION
    st.markdown(
        """
        <div class="terminal-hero">
          <div>
            <div class="eyebrow">A-SHARE ALPHA RESEARCH PLATFORM</div>
            <h1>华策 Alpha 投研平台</h1>
          </div>
          <div class="risk-note">研究辅助工具 · 不构成投资建议 · 不承诺收益</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    public_mode = public_snapshot_mode()
    with st.sidebar:
        st.header("策略设置")
        horizon = 30 if public_mode else st.slider("目标持有期（交易日）", 20, 40, 30, 5)
        top_n = st.slider("推荐数量上限", 3, 20, 10)
        min_amount = 80_000_000 if public_mode else st.number_input("20日平均成交额下限（万元）", 1000, 100000, 8000, 1000) * 10000
        transaction_cost = 0.003 if public_mode else st.number_input("单轮交易成本", 0.0, 0.02, 0.003, 0.001, format="%.3f")
        if public_mode:
            st.caption("公网版使用每日收盘后生成的共享模型快照。")

        st.header("个股评估")
        evaluation_code = st.text_input("股票代码", value="", placeholder="例如 000001 或 600519")
        normalized_evaluation_code = normalize_stock_code(evaluation_code)
        if evaluation_code and normalized_evaluation_code:
            st.caption(f"将评估：{normalized_evaluation_code}")

        if public_mode:
            prefer_tushare = False
            tushare_token = ""
            boards = BOARD_OPTIONS
            max_symbols = 800
            history_years = 4
            use_finance = True
            force_refresh = False
            run = st.button("查看最新推荐", type="primary", use_container_width=True)
        else:
            st.header("数据设置")
            prefer_tushare = st.checkbox("优先使用 Tushare Pro", value=False)
            tushare_token = st.text_input("Tushare token", type="password")
            boards = st.multiselect("市场板块", BOARD_OPTIONS, default=BOARD_OPTIONS)
            st.caption("推荐列表使用真实数据大池排名：最多扫描 800 只股票，并按综合评估从高到低输出。")
            max_symbols = st.slider("真实数据股票数量", 20, 800, 800, 20)
            history_years = st.slider("历史数据年限", 2, 6, 4, 1)
            use_finance = st.checkbox("启用财务增强（较慢）", value=True)
            force_refresh = st.checkbox("忽略今日缓存并重新拉取", value=False)
            run = st.button("生成今日推荐", type="primary", use_container_width=True)
        evaluate_only = st.button("评估输入股票", use_container_width=True)

    config = StrategyConfig(
        horizon_days=horizon,
        top_n=top_n,
        min_amount=float(min_amount),
        transaction_cost=float(transaction_cost),
    )
    extra_symbols = (normalized_evaluation_code,) if normalized_evaluation_code else ()
    data_request = DataRequest(
        max_symbols=max_symbols,
        history_years=history_years,
        use_finance=use_finance,
        force_sample=False,
        force_refresh=force_refresh,
        allow_sample_fallback=False,
        extra_symbols=extra_symbols,
        boards=tuple(boards or BOARD_OPTIONS),
        full_market_scan=True,
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
        force_refresh,
    )
    if st.session_state.get("cache_key") != cache_key:
        st.session_state.pop("result", None)
        st.session_state.pop("run_error", None)
        st.session_state["cache_key"] = cache_key

    requested_run = run or evaluate_only
    if requested_run:
        spinner_text = "正在读取最新共享快照..." if public_mode else "正在更新数据、训练模型并回测。真实数据首次拉取可能需要数分钟..."
        with st.spinner(spinner_text):
            try:
                if public_mode:
                    st.session_state["result"] = _cached_public_snapshot()
                else:
                    st.session_state["result"] = run_pipeline(
                        config,
                        prefer_tushare=prefer_tushare,
                        tushare_token=tushare_token or None,
                        data_request=data_request,
                    )
                st.session_state.pop("run_error", None)
            except Exception as exc:
                st.session_state.pop("result", None)
                st.session_state["run_error"] = str(exc)

    if "result" not in st.session_state:
        _render_landing(evaluation_code, public_mode)
        if "run_error" in st.session_state:
            st.error(st.session_state["run_error"])
            st.info("真实数据失败时不会生成推荐结果。可以点击“忽略今日缓存并重新拉取”，或稍后等待 AKShare/Tushare 接口恢复。")
        return

    result = st.session_state["result"]
    market_regime = _latest_market_regime(result)
    displayed_recommendations = result.recommendations.head(top_n)
    buy_count = int((displayed_recommendations["action"] == "买入观察").sum()) if not displayed_recommendations.empty else 0
    _render_health_panel(result, market_regime, buy_count)

    cols = st.columns(6)
    cols[0].metric("数据日期", result.data_date.strftime("%Y-%m-%d"))
    cols[1].metric("市场状态", _market_label(market_regime), f"{market_regime:.0%}")
    cols[2].metric("买入观察", buy_count)
    cols[3].metric("样本外胜率", f"{result.metrics['sample_win_rate']:.1%}")
    cols[4].metric("最大回撤", f"{result.metrics['max_drawdown']:.1%}")
    cols[5].metric("收益回撤比", f"{result.metrics['return_drawdown_ratio']:.2f}")

    tab_rec, tab_eval, tab_bt, tab_data = st.tabs(["今日推荐", "个股评估", "回测证据", "数据源状态"])

    with tab_rec:
        st.subheader("推荐列表")
        st.caption("当前为真实数据大池排名口径：候选股来自所选板块最多 800 只股票，并严格按综合评估从高到低排列。")
        st.dataframe(
            displayed_recommendations,
            use_container_width=True,
            column_config={
                "market_rank": st.column_config.NumberColumn("大池排名", format="%d"),
                "rating": st.column_config.TextColumn("候选等级"),
                "action": st.column_config.TextColumn("动作"),
                "win_probability": st.column_config.ProgressColumn("胜率评分", min_value=0, max_value=1, format="percent"),
                "composite_score": st.column_config.ProgressColumn("综合评分", min_value=0, max_value=1, format="percent"),
                "trend_score": st.column_config.ProgressColumn("趋势", min_value=0, max_value=1, format="percent"),
                "risk_score": st.column_config.ProgressColumn("风险质量", min_value=0, max_value=1, format="percent"),
                "fundamental_score": st.column_config.ProgressColumn("基本面", min_value=0, max_value=1, format="percent"),
                "industry_score": st.column_config.ProgressColumn("行业", min_value=0, max_value=1, format="percent"),
                "market_regime_score": st.column_config.ProgressColumn("市场", min_value=0, max_value=1, format="percent"),
                "score": st.column_config.NumberColumn("综合分", format="%.4f"),
                "board": st.column_config.TextColumn("市场板块"),
                "score_rank": st.column_config.ProgressColumn("分位", min_value=0, max_value=1),
                "close": st.column_config.NumberColumn("收盘价", format="%.2f"),
                "position_limit": st.column_config.NumberColumn("仓位上限", format="percent"),
                "stop_loss": st.column_config.NumberColumn("止损价", format="%.2f"),
            },
        )
        st.download_button(
            "导出推荐 CSV",
            displayed_recommendations.to_csv(index=False).encode("utf-8-sig"),
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
    cols = st.columns(7)
    cols[0].metric("结论", evaluation.conclusion)
    cols[1].metric("股票", f"{summary['名称']} {summary['代码']}")
    cols[2].metric("评级", str(summary["综合评级"]))
    cols[3].metric("胜率评分", f"{summary['胜率评分']:.1%}")
    cols[4].metric("模型分位", f"{summary['股票池分位']:.1%}")
    cols[5].metric("收盘价", f"{summary['最新收盘价']:.2f}")
    cols[6].metric("板块", str(summary["市场板块"]))

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


def _render_health_panel(result, market_regime: float, buy_count: int) -> None:
    snapshot_age = (pd.Timestamp.now().normalize() - result.data_date.normalize()).days
    if result.provider_status.mode.startswith("snapshot-") and snapshot_age > 3:
        data_title = "数据状态：共享快照已过期"
        data_body = f"当前快照数据日期为 {result.data_date:%Y-%m-%d}，距今 {snapshot_age} 天。系统已保留最近一次成功结果，请等待自动更新。"
        data_class = "health-warn"
    elif result.provider_status.mode.startswith("snapshot-"):
        data_title = "数据状态：共享快照可用"
        data_body = f"当前使用每日预计算的真实数据快照，数据日期：{result.data_date:%Y-%m-%d}。"
        data_class = "health-ok"
    elif result.provider_status.mode == "akshare-stale-cache":
        data_title = "数据状态：使用最近交易日缓存"
        data_body = f"在线接口暂不可用，系统已使用真实历史缓存继续运行。当前数据日期：{result.data_date:%Y-%m-%d}。如需刷新，稍后勾选“忽略今日缓存并重新拉取”。"
        data_class = "health-warn"
    elif "sample" in result.provider_status.mode:
        data_title = "数据状态：演示数据"
        data_body = "当前结果来自样例数据，仅用于查看流程和界面，不用于现实判断。"
        data_class = "health-warn"
    else:
        data_title = "数据状态：真实数据在线"
        data_body = result.provider_status.message
        data_class = "health-ok"

    if result.gate_ok and buy_count > 0:
        model_title = "模型状态：允许进攻"
        model_body = f"回测门槛通过，当前有 {buy_count} 只股票进入买入观察。"
        model_class = "health-ok"
    elif result.gate_ok:
        model_title = "模型状态：通过但无买入"
        model_body = "回测门槛通过，但当前个股风控或市场状态未触发买入观察。"
        model_class = "health-warn"
    else:
        model_title = "模型状态：防守"
        model_body = "今日清单仅用于观察。原因：" + "；".join(result.gate_reasons)
        model_class = "health-bad"

    market_title = "市场状态：" + _market_label(market_regime)
    market_body = f"市场状态评分 {market_regime:.0%}。低于 45% 时系统倾向防守，只给观察或等回调动作。"
    market_class = "health-ok" if market_regime >= 0.62 else "health-warn" if market_regime >= 0.45 else "health-bad"

    cards = [
        (data_class, data_title, data_body),
        (model_class, model_title, model_body),
        (market_class, market_title, market_body),
    ]
    html = '<div class="health-grid">' + "".join(
        f'<div class="health-card {klass}"><div class="health-title">{title}</div><div class="health-copy">{body}</div></div>'
        for klass, title, body in cards
    ) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _render_landing(evaluation_code: str, public_mode: bool = False) -> None:
    recommendation_copy = (
        "点击左侧“查看最新推荐”。系统直接读取每日收盘后生成的 800 股共享快照。"
        if public_mode
        else "选择市场板块和数据源后，点击左侧“生成今日推荐”。系统会拉取真实行情、训练集成模型并输出经过风控过滤的候选清单。"
    )
    evaluation_copy = (
        "输入股票代码后点击“评估输入股票”。系统使用共享快照展示评级、胜率评分、趋势、基本面和风险解释。"
        if public_mode
        else "输入股票代码后点击“评估输入股票”。系统会强制拉取该股票，展示评级、胜率评分、趋势、基本面和风险解释。"
    )
    st.markdown(
        f"""
        <div class="landing-grid">
          <div class="landing-card">
            <div class="card-title">今日工作台</div>
            <div class="card-copy">{recommendation_copy}</div>
          </div>
          <div class="landing-card">
            <div class="card-title">个股评估</div>
            <div class="card-copy">{evaluation_copy}</div>
          </div>
          <div class="landing-card">
            <div class="card-title">数据纪律</div>
            <div class="card-copy">系统只展示真实数据结果。公开接口失败时不生成推荐结果；请等待接口恢复、使用 Tushare token，或重新拉取缓存。</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    next_step = "已输入代码，点击左侧“评估输入股票”开始。" if evaluation_code.strip() else "可先输入股票代码，或直接生成今日推荐。"
    mode = "当前为真实数据模式。"
    st.markdown(f'<div class="data-banner">{mode} {next_step}</div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner=False, ttl=900)
def _cached_public_snapshot():
    return load_configured_snapshot()


def _signals_to_frame(signals: dict[str, float]) -> pd.DataFrame:
    percent_keys = {
        "20日收益",
        "60日收益",
        "120日收益",
        "20日均线偏离",
        "60日均线偏离",
        "20日波动",
        "ROE",
        "净利润增长率",
        "行业20日强度",
        "趋势评分",
        "风险评分",
        "基本面评分",
        "行业评分",
        "市场状态评分",
    }
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


def _latest_market_regime(result) -> float:
    if result.latest_scored.empty or "market_regime_score" not in result.latest_scored.columns:
        return 0.5
    return float(result.latest_scored["market_regime_score"].mean())


def _market_label(score: float) -> str:
    if score >= 0.62:
        return "允许进攻"
    if score >= 0.45:
        return "中性观察"
    return "防守"


def _apply_commercial_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
          --bg: #07111f;
          --panel: #0f1a2b;
          --panel-2: #132238;
          --line: #24344f;
          --text: #e6edf7;
          --muted: #94a3b8;
          --blue: #2f81f7;
          --green: #2fbf71;
          --red: #f05252;
          --gold: #d6a84f;
        }
        .stApp { background: var(--bg); color: var(--text); }
        [data-testid="stSidebar"] { background: #081321; border-right: 1px solid var(--line); }
        [data-testid="stMetric"] {
          background: linear-gradient(180deg, var(--panel), #0b1727);
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 12px 14px;
          min-height: 96px;
        }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        [data-testid="stMetricValue"] { color: var(--text); font-size: 1.35rem; }
        .terminal-hero {
          display:flex;
          align-items:flex-end;
          justify-content:space-between;
          gap:16px;
          padding: 18px 20px;
          margin: 0 0 14px 0;
          border: 1px solid var(--line);
          border-radius: 8px;
          background: linear-gradient(135deg, #0d1d33 0%, #07111f 58%, #111827 100%);
        }
        .terminal-hero h1 {
          margin: 2px 0 0 0;
          font-size: 45px;
          line-height: 1.1;
          letter-spacing: 0;
        }
        .eyebrow { color: var(--gold); font-size: 12px; font-weight: 700; }
        .risk-note { color: var(--muted); font-size: 13px; white-space: nowrap; }
        .data-banner {
          border: 1px solid var(--line);
          background: var(--panel);
          border-left: 4px solid var(--blue);
          border-radius: 6px;
          padding: 10px 12px;
          margin: 8px 0 14px 0;
          color: var(--text);
        }
        .landing-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 14px;
          margin: 12px 0 16px 0;
        }
        .landing-card {
          border: 1px solid var(--line);
          background: linear-gradient(180deg, var(--panel), #0a1524);
          border-radius: 8px;
          padding: 18px;
          min-height: 150px;
        }
        .card-title {
          color: var(--text);
          font-size: 18px;
          font-weight: 800;
          margin-bottom: 10px;
        }
        .card-copy {
          color: var(--muted);
          font-size: 14px;
          line-height: 1.65;
        }
        .health-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 12px;
          margin: 10px 0 14px 0;
        }
        .health-card {
          border: 1px solid var(--line);
          background: var(--panel);
          border-radius: 8px;
          padding: 14px 16px;
          min-height: 116px;
        }
        .health-ok { border-left: 4px solid var(--green); }
        .health-warn { border-left: 4px solid var(--gold); }
        .health-bad { border-left: 4px solid var(--red); }
        .health-title {
          font-size: 15px;
          font-weight: 800;
          margin-bottom: 8px;
          color: var(--text);
        }
        .health-copy {
          color: var(--muted);
          font-size: 13px;
          line-height: 1.55;
        }
        div[data-testid="stDataFrame"] {
          border: 1px solid var(--line);
          border-radius: 8px;
          overflow: hidden;
        }
        .stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
        .stTabs [data-baseweb="tab"] {
          background: var(--panel);
          border: 1px solid var(--line);
          border-bottom: 0;
          border-radius: 8px 8px 0 0;
          padding: 8px 14px;
        }
        .stButton > button {
          border-radius: 6px;
          border: 1px solid #3b82f6;
          background: #1d4ed8;
          color: white;
          font-weight: 700;
        }
        h2, h3 { letter-spacing: 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )
