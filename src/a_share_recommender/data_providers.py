from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.util import find_spec
from pathlib import Path
import hashlib
import os
import time

import numpy as np
import pandas as pd

from .sample_data import make_sample_market


CORE_A_SHARE_POOL = [
    "000001",
    "000002",
    "000063",
    "000100",
    "000333",
    "000338",
    "000651",
    "000725",
    "000858",
    "000977",
    "002027",
    "002049",
    "002129",
    "002230",
    "002236",
    "002241",
    "002271",
    "002304",
    "002352",
    "002415",
    "002475",
    "002594",
    "002714",
    "300014",
    "300015",
    "300031",
    "300059",
    "300122",
    "300124",
    "300274",
    "300308",
    "300347",
    "300408",
    "300450",
    "300750",
    "300760",
    "600000",
    "600009",
    "600010",
    "600019",
    "600028",
    "600030",
    "600031",
    "600036",
    "600048",
    "600050",
    "600104",
    "600196",
    "600276",
    "600309",
    "600406",
    "600438",
    "600519",
    "600570",
    "600585",
    "600690",
    "600703",
    "600745",
    "600887",
    "601012",
    "601088",
    "601166",
    "601318",
    "601398",
    "601601",
    "601633",
    "601668",
    "601688",
    "601857",
    "601888",
    "601899",
    "603259",
    "603501",
    "603799",
    "688008",
    "688012",
    "688036",
    "688111",
    "688126",
    "688169",
    "688256",
]

CORE_A_SHARE_NAMES = {
    "000001": "平安银行",
    "000002": "万科A",
    "000063": "中兴通讯",
    "000100": "TCL科技",
    "000333": "美的集团",
    "000338": "潍柴动力",
    "000651": "格力电器",
    "000725": "京东方A",
    "000858": "五粮液",
    "000977": "浪潮信息",
    "002027": "分众传媒",
    "002049": "紫光国微",
    "002129": "TCL中环",
    "002230": "科大讯飞",
    "002236": "大华股份",
    "002241": "歌尔股份",
    "002271": "东方雨虹",
    "002304": "洋河股份",
    "002352": "顺丰控股",
    "002415": "海康威视",
    "002475": "立讯精密",
    "002594": "比亚迪",
    "002714": "牧原股份",
    "300014": "亿纬锂能",
    "300015": "爱尔眼科",
    "300031": "宝通科技",
    "300059": "东方财富",
    "300122": "智飞生物",
    "300124": "汇川技术",
    "300274": "阳光电源",
    "300308": "中际旭创",
    "300347": "泰格医药",
    "300408": "三环集团",
    "300450": "先导智能",
    "300750": "宁德时代",
    "300760": "迈瑞医疗",
    "688008": "澜起科技",
    "688012": "中微公司",
    "688036": "传音控股",
    "688111": "金山办公",
    "688126": "沪硅产业",
    "688169": "石头科技",
    "688256": "寒武纪",
}

KNOWN_A_SHARE_NAMES = CORE_A_SHARE_NAMES


@dataclass(frozen=True)
class ProviderStatus:
    mode: str
    message: str
    rows: int


@dataclass(frozen=True)
class DataRequest:
    max_symbols: int = 800
    history_years: int = 4
    use_finance: bool = True
    force_sample: bool = False
    force_refresh: bool = False
    allow_sample_fallback: bool = False
    extra_symbols: tuple[str, ...] = ()
    full_market_scan: bool = False
    boards: tuple[str, ...] = ("上证主板", "深证主板", "创业板", "科创板")


class SampleProvider:
    name = "sample"

    def load_market(self, request: DataRequest | None = None) -> tuple[pd.DataFrame, ProviderStatus]:
        data = make_sample_market()
        if request:
            data = _inject_sample_extras(data, request.extra_symbols)
        return data, ProviderStatus("sample", "使用内置样例数据；结果不是现实市场推荐。", len(data))


class AkshareProvider:
    name = "akshare"

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir

    def load_market(self, request: DataRequest) -> tuple[pd.DataFrame, ProviderStatus]:
        cache_path = _cache_path(self.cache_dir, "akshare", request)
        if cache_path.exists() and not request.force_refresh:
            data = pd.read_parquet(cache_path)
            if _cache_satisfies_request(data, request):
                return data, ProviderStatus("akshare-cache", f"使用 AKShare 本地缓存：{cache_path}", len(data))

        if not find_spec("akshare"):
            raise RuntimeError("未安装 akshare，请先运行 python -m pip install akshare")

        import akshare as ak

        full_universe = _akshare_universe(ak, self.cache_dir, request.force_refresh)
        _assert_full_market_universe(full_universe, request, "code")
        universe = _filter_boards(full_universe, request.boards)
        symbols = _select_request_symbols(universe, request)
        if not symbols:
            symbols = CORE_A_SHARE_POOL[: request.max_symbols]

        start_date = (date.today() - timedelta(days=365 * request.history_years + 90)).strftime("%Y%m%d")
        end_date = date.today().strftime("%Y%m%d")

        frames = []
        errors: list[str] = []
        workers = max(1, min(int(os.getenv("DATA_FETCH_WORKERS", "8")), 12, len(symbols)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="akshare") as executor:
            futures = {
                executor.submit(
                    _load_one_akshare_symbol,
                    ak,
                    self.cache_dir,
                    full_universe,
                    symbol,
                    start_date,
                    end_date,
                    request.force_refresh,
                ): symbol
                for symbol in symbols
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    frame = future.result()
                    if frame is None or frame.empty:
                        errors.append(f"{symbol}:Empty")
                    else:
                        frames.append(frame)
                except Exception as exc:
                    errors.append(f"{symbol}:{type(exc).__name__}")

        if not frames:
            stale = _load_latest_provider_cache(self.cache_dir, "akshare", request)
            if stale is not None:
                message = "AKShare 当前连接失败，已使用最近一次真实数据缓存；失败样本：" + "；".join(errors[:5])
                return stale, ProviderStatus("akshare-stale-cache", message, len(stale))
            raise RuntimeError("AKShare 日线接口未返回可用数据：" + "；".join(errors[:5]))

        data = pd.concat(frames, ignore_index=True)
        stale_fill_count = 0
        if errors:
            stale = _load_latest_provider_cache(self.cache_dir, "akshare", request)
            if stale is not None:
                data, stale_fill_count = _merge_missing_symbol_history(data, stale, symbols)
        if request.use_finance:
            data = _attach_akshare_finance(ak, data, symbols[: min(len(symbols), 30)])

        data = data.sort_values(["date", "code"]).reset_index(drop=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(cache_path, index=False)
        suffix = f"；失败 {len(errors)} 只" if errors else ""
        if stale_fill_count:
            suffix += f"；其中 {stale_fill_count} 只使用最近真实历史补齐"
        return data, ProviderStatus("akshare", f"使用 AKShare 真实日线/行业/财务数据{suffix}", len(data))


def _load_one_akshare_symbol(
    ak,
    cache_dir: Path,
    full_universe: pd.DataFrame,
    symbol: str,
    start_date: str,
    end_date: str,
    force_refresh: bool,
) -> pd.DataFrame | None:
    meta = full_universe.loc[full_universe["code"] == symbol].head(1)
    symbol_cache = _symbol_cache_path(cache_dir, "akshare", symbol, start_date, end_date)
    cached = _load_symbol_cache(symbol_cache, force_refresh)
    if cached is not None:
        return cached

    raw, source = _load_akshare_hist_with_retry(ak, symbol, start_date, end_date)
    if raw.empty:
        return None
    normalized = (
        _normalize_akshare_tx_hist(raw, symbol, meta)
        if source == "tx"
        else _normalize_akshare_hist(raw, meta)
    )
    if normalized.empty:
        return None
    _save_symbol_cache(symbol_cache, normalized)
    return normalized


def _merge_missing_symbol_history(
    current: pd.DataFrame,
    stale: pd.DataFrame,
    symbols: list[str],
) -> tuple[pd.DataFrame, int]:
    requested_codes = {_suffix_code(symbol) for symbol in symbols}
    current_codes = set(current["code"].astype(str))
    missing_codes = requested_codes - current_codes
    fallback = stale[stale["code"].astype(str).isin(missing_codes)].copy()
    if fallback.empty:
        return current, 0
    merged = pd.concat([current, fallback], ignore_index=True)
    merged = merged.drop_duplicates(["date", "code"], keep="first")
    return merged, int(fallback["code"].nunique())


class TushareProvider:
    name = "tushare"

    def __init__(self, token: str, cache_dir: Path):
        self.token = token
        self.cache_dir = cache_dir

    def load_market(self, request: DataRequest) -> tuple[pd.DataFrame, ProviderStatus]:
        cache_path = _cache_path(self.cache_dir, "tushare", request)
        if cache_path.exists() and not request.force_refresh:
            data = pd.read_parquet(cache_path)
            if _cache_satisfies_request(data, request):
                return data, ProviderStatus("tushare-cache", f"使用 Tushare 本地缓存：{cache_path}", len(data))

        if not find_spec("tushare"):
            raise RuntimeError("未安装 tushare，请先运行 python -m pip install tushare")

        import tushare as ts

        ts.set_token(self.token)
        pro = ts.pro_api(self.token)
        full_universe = _tushare_universe(pro)
        _assert_full_market_universe(full_universe, request, "ts_code")
        universe = _filter_boards(full_universe, request.boards)
        symbols = [
            _suffix_code(symbol)
            for symbol in _select_request_symbols(universe.rename(columns={"ts_code": "code"}), request)
        ]
        start_date = (date.today() - timedelta(days=365 * request.history_years + 90)).strftime("%Y%m%d")
        end_date = date.today().strftime("%Y%m%d")

        frames = []
        errors: list[str] = []
        for ts_code in symbols:
            try:
                raw = ts.pro_bar(ts_code=ts_code, adj="qfq", start_date=start_date, end_date=end_date)
                if raw is None or raw.empty:
                    continue
                meta = full_universe.loc[full_universe["ts_code"] == ts_code].head(1)
                daily_basic = _safe_tushare_daily_basic(pro, ts_code, start_date, end_date)
                fina = _safe_tushare_fina_indicator(pro, ts_code) if request.use_finance else pd.DataFrame()
                frames.append(_normalize_tushare_hist(raw, meta, daily_basic, fina))
                time.sleep(0.15)
            except Exception as exc:
                errors.append(f"{ts_code}:{type(exc).__name__}")

        if not frames:
            raise RuntimeError("Tushare 日线接口未返回可用数据：" + "；".join(errors[:5]))

        data = pd.concat(frames, ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(cache_path, index=False)
        suffix = f"；失败 {len(errors)} 只" if errors else ""
        return data, ProviderStatus("tushare", f"使用 Tushare Pro 真实日线/行业/财务数据{suffix}", len(data))


class ProviderRouter:
    def __init__(
        self,
        prefer_tushare: bool = False,
        tushare_token: str | None = None,
        cache_dir: Path | str = "data/cache",
    ):
        self.prefer_tushare = prefer_tushare
        self.tushare_token = tushare_token
        self.cache_dir = Path(cache_dir)
        self.last_errors: list[str] = []

    def load_market(self, request: DataRequest) -> tuple[pd.DataFrame, ProviderStatus]:
        if request.force_sample:
            return SampleProvider().load_market(request)

        providers = []
        if self.prefer_tushare and self.tushare_token:
            providers.append(TushareProvider(self.tushare_token, self.cache_dir))
        providers.append(AkshareProvider(self.cache_dir))
        if self.tushare_token and not self.prefer_tushare:
            providers.append(TushareProvider(self.tushare_token, self.cache_dir))

        self.last_errors = []
        for provider in providers:
            try:
                return provider.load_market(request)
            except Exception as exc:
                self.last_errors.append(f"{provider.name}: {exc}")

        if request.allow_sample_fallback:
            data, status = SampleProvider().load_market(request)
            message = status.message + " 真实数据源不可用：" + " | ".join(self.last_errors[:3])
            return data, ProviderStatus("sample-fallback", message, len(data))
        raise RuntimeError("真实数据源不可用：" + " | ".join(self.last_errors[:3]))

    def availability(self) -> dict[str, str]:
        return {
            "AKShare": "已安装" if find_spec("akshare") else "未安装",
            "Tushare": "已安装" if find_spec("tushare") else "未安装",
            "最近错误": " | ".join(self.last_errors) if self.last_errors else "无",
        }


def _cache_path(cache_dir: Path, provider: str, request: DataRequest) -> Path:
    extras = ",".join(sorted(_plain_symbol(symbol) for symbol in request.extra_symbols))
    boards = ",".join(request.boards)
    key = f"{provider}-{request.max_symbols}-{request.history_years}-{request.use_finance}-{request.full_market_scan}-{extras}-{boards}-{date.today():%Y%m%d}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return cache_dir / f"{provider}_{digest}.parquet"


def _symbol_cache_path(cache_dir: Path, provider: str, symbol: str, start_date: str, end_date: str) -> Path:
    return cache_dir / "symbols" / provider / f"{_plain_symbol(symbol)}_{start_date}_{end_date}.parquet"


def _load_symbol_cache(path: Path, force_refresh: bool) -> pd.DataFrame | None:
    if force_refresh or not path.exists():
        return None
    try:
        data = pd.read_parquet(path)
    except Exception:
        return None
    required = {"date", "code", "close", "name"}
    if data.empty or not required.issubset(data.columns):
        return None
    return data


def _save_symbol_cache(path: Path, data: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(path, index=False)


def _load_akshare_hist_with_retry(ak, symbol: str, start_date: str, end_date: str, attempts: int = 3) -> tuple[pd.DataFrame, str]:
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            ), "em"
        except Exception as exc:
            last_exc = exc
            time.sleep(0.8 * (attempt + 1))
    try:
        return ak.stock_zh_a_hist_tx(
            symbol=_tx_symbol(symbol),
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        ), "tx"
    except Exception as exc:
        last_exc = exc
    if last_exc:
        raise last_exc
    return pd.DataFrame(), "em"


def _load_latest_provider_cache(cache_dir: Path, provider: str, request: DataRequest | None = None) -> pd.DataFrame | None:
    if not cache_dir.exists():
        return None
    candidates = sorted(
        [path for path in cache_dir.glob(f"{provider}_*.parquet") if path.name != f"{provider}_universe.parquet"],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    frames: list[pd.DataFrame] = []
    for path in candidates:
        try:
            data = pd.read_parquet(path)
            required = {"date", "code", "close", "name"}
            if not data.empty and required.issubset(data.columns):
                frames.append(data)
        except Exception:
            continue
    if not frames:
        return None

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values("date", ascending=False).drop_duplicates(["date", "code"], keep="first")
    if request is None or "board" not in merged.columns:
        return merged.sort_values(["date", "code"]).reset_index(drop=True)

    filtered = merged[merged["board"].isin(request.boards)].copy()
    if filtered.empty:
        return None

    latest_by_code = filtered.sort_values("date").groupby("code").tail(1)
    selected_plain = _select_request_symbols(
        latest_by_code.assign(code=latest_by_code["code"].map(_plain_symbol)),
        request,
    )
    selected_codes = {_suffix_code(symbol) for symbol in selected_plain}
    selected = filtered[filtered["code"].isin(selected_codes)].copy()
    if selected["code"].nunique() < _min_symbols_for_request(request):
        return None
    return selected.sort_values(["date", "code"]).reset_index(drop=True)


def _cache_satisfies_request(data: pd.DataFrame, request: DataRequest) -> bool:
    if data.empty or "code" not in data.columns:
        return False
    required_extras = {_suffix_code(symbol) for symbol in request.extra_symbols if _plain_symbol(symbol)}
    if required_extras and not required_extras.issubset(set(data["code"].astype(str))):
        return False
    min_symbols = _min_symbols_for_request(request)
    if data["code"].nunique() < min_symbols:
        return False
    if "board" in data.columns:
        requested_boards = {board for board in request.boards if board != "全市场"}
        available_requested = set(data["board"].dropna().astype(str)) & requested_boards
        if len(requested_boards) >= 2 and request.max_symbols >= 8 and len(available_requested) < 2:
            return False
    return True


def _min_symbols_for_request(request: DataRequest) -> int:
    if request.full_market_scan:
        return max(8, min(request.max_symbols, 100))
    return max(3, min(request.max_symbols, 8))


def _assert_full_market_universe(universe: pd.DataFrame, request: DataRequest, code_column: str) -> None:
    if not request.full_market_scan:
        return
    if universe.empty or code_column not in universe.columns or universe[code_column].nunique() < 1000:
        raise RuntimeError("大池排名需要完整 A 股股票列表；当前数据源只返回了过少股票，不能作为推荐依据。请稍后重试、使用 Tushare token，或降低真实数据股票数量后重新运行。")


def _akshare_universe(ak, cache_dir: Path, force_refresh: bool) -> pd.DataFrame:
    cache_path = cache_dir / "akshare_universe.parquet"
    if cache_path.exists() and not force_refresh:
        cached = pd.read_parquet(cache_path)
        if _universe_is_usable(cached):
            return cached

    rows = []
    try:
        sh = ak.stock_info_sh_name_code()
        rows.append(
            pd.DataFrame(
                {
                    "code": sh["证券代码"].astype(str).str.zfill(6),
                    "name": sh["证券简称"].astype(str),
                    "industry": "未知",
                    "board": sh["证券代码"].astype(str).str.zfill(6).map(_board_from_symbol),
                    "listing_date": pd.to_datetime(sh["上市日期"], errors="coerce"),
                }
            )
        )
    except Exception:
        pass

    try:
        sz = ak.stock_info_sz_name_code()
        rows.append(
            pd.DataFrame(
                {
                    "code": sz["A股代码"].astype(str).str.zfill(6),
                    "name": sz["A股简称"].astype(str),
                    "industry": sz.get("所属行业", "未知").astype(str),
                    "board": sz.get("板块", "").astype(str).map(_normalize_sz_board),
                    "listing_date": pd.to_datetime(sz["A股上市日期"], errors="coerce"),
                }
            )
        )
    except Exception:
        pass

    try:
        bj = ak.stock_info_bj_name_code()
        rows.append(
            pd.DataFrame(
                {
                    "code": bj["证券代码"].astype(str).str.zfill(6),
                    "name": bj["证券简称"].astype(str),
                    "industry": "未知",
                    "board": "北交所",
                    "listing_date": pd.to_datetime(bj["上市日期"], errors="coerce"),
                }
            )
        )
    except Exception:
        pass

    rows.append(_core_fallback_universe())

    if rows:
        universe = pd.concat(rows, ignore_index=True)
        universe = universe.drop_duplicates("code")
        core_rank = {code: i for i, code in enumerate(CORE_A_SHARE_POOL)}
        universe["rank"] = universe["code"].map(core_rank)
        universe["rank"] = universe["rank"].fillna(len(core_rank) + universe.index.to_series())
        universe = universe.sort_values("rank").drop(columns=["rank"]).reset_index(drop=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        universe.to_parquet(cache_path, index=False)
        return universe

    return pd.DataFrame(
        {
            "code": CORE_A_SHARE_POOL,
            "name": CORE_A_SHARE_POOL,
            "industry": "未知",
            "board": [_board_from_symbol(code) for code in CORE_A_SHARE_POOL],
            "listing_date": pd.NaT,
        }
    )


def _normalize_akshare_hist(raw: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    symbol = str(raw["股票代码"].iloc[0]).zfill(6)
    meta_row = meta.iloc[0].to_dict() if not meta.empty else {}
    listing_date = pd.to_datetime(meta_row.get("listing_date"), errors="coerce")
    dates = pd.to_datetime(raw["日期"], errors="coerce")
    list_days = (dates - listing_date).dt.days if pd.notna(listing_date) else 9999

    return pd.DataFrame(
        {
            "date": dates,
            "code": _suffix_code(symbol),
            "name": meta_row.get("name", symbol),
            "industry": meta_row.get("industry", "未知"),
            "board": meta_row.get("board", _board_from_symbol(symbol)),
            "open": pd.to_numeric(raw["开盘"], errors="coerce"),
            "high": pd.to_numeric(raw["最高"], errors="coerce"),
            "low": pd.to_numeric(raw["最低"], errors="coerce"),
            "close": pd.to_numeric(raw["收盘"], errors="coerce"),
            "volume": pd.to_numeric(raw["成交量"], errors="coerce") * 100,
            "amount": pd.to_numeric(raw["成交额"], errors="coerce"),
            "turnover_rate": pd.to_numeric(raw["换手率"], errors="coerce"),
            "money_flow": 0.0,
            "pe_ttm": 25.0,
            "roe": 0.08,
            "net_profit_growth": 0.0,
            "market_cap": 10_000_000_000.0,
            "is_st": str(meta_row.get("name", "")).upper().find("ST") >= 0,
            "list_days": list_days,
            "suspended": False,
        }
    ).dropna(subset=["date", "close"])


def _normalize_akshare_tx_hist(raw: pd.DataFrame, symbol: str, meta: pd.DataFrame) -> pd.DataFrame:
    plain = _plain_symbol(symbol)
    meta_row = meta.iloc[0].to_dict() if not meta.empty else known_stock_identity(plain)
    listing_date = pd.to_datetime(meta_row.get("listing_date"), errors="coerce")
    dates = pd.to_datetime(raw["date"], errors="coerce")
    list_days = (dates - listing_date).dt.days if pd.notna(listing_date) else 9999
    volume = pd.to_numeric(raw.get("amount", 0.0), errors="coerce") * 100
    close = pd.to_numeric(raw["close"], errors="coerce")

    return pd.DataFrame(
        {
            "date": dates,
            "code": _suffix_code(plain),
            "name": meta_row.get("name", plain),
            "industry": meta_row.get("industry", "未知"),
            "board": meta_row.get("board", _board_from_symbol(plain)),
            "open": pd.to_numeric(raw["open"], errors="coerce"),
            "high": pd.to_numeric(raw["high"], errors="coerce"),
            "low": pd.to_numeric(raw["low"], errors="coerce"),
            "close": close,
            "volume": volume,
            "amount": volume * close,
            "turnover_rate": 0.0,
            "money_flow": 0.0,
            "pe_ttm": 25.0,
            "roe": 0.08,
            "net_profit_growth": 0.0,
            "market_cap": 10_000_000_000.0,
            "is_st": str(meta_row.get("name", "")).upper().find("ST") >= 0,
            "list_days": list_days,
            "suspended": False,
        }
    ).dropna(subset=["date", "close"])


def _attach_akshare_finance(ak, data: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    enriched = data.copy()
    for symbol in symbols:
        try:
            finance = ak.stock_financial_analysis_indicator(symbol=symbol, start_year=str(date.today().year - 3))
            if finance.empty:
                continue
            latest = finance.sort_values("日期").iloc[-1]
            mask = enriched["code"] == _suffix_code(symbol)
            if "净资产收益率(%)" in finance.columns:
                enriched.loc[mask, "roe"] = pd.to_numeric(latest["净资产收益率(%)"], errors="coerce") / 100
            if "净利润增长率(%)" in finance.columns:
                enriched.loc[mask, "net_profit_growth"] = pd.to_numeric(latest["净利润增长率(%)"], errors="coerce") / 100
            if "总资产(元)" in finance.columns:
                enriched.loc[mask, "market_cap"] = pd.to_numeric(latest["总资产(元)"], errors="coerce")
            time.sleep(0.05)
        except Exception:
            continue
    return enriched


def _tushare_universe(pro) -> pd.DataFrame:
    fields = "ts_code,symbol,name,industry,list_date"
    universe = pro.stock_basic(exchange="", list_status="L", fields=fields)
    universe = universe.copy()
    universe["board"] = universe["ts_code"].map(_board_from_symbol)
    core_rank = {_suffix_code(code): i for i, code in enumerate(CORE_A_SHARE_POOL)}
    universe["rank"] = universe["ts_code"].map(core_rank)
    universe["rank"] = universe["rank"].fillna(len(core_rank) + universe.index.to_series())
    return universe.sort_values("rank").drop(columns=["rank"]).reset_index(drop=True)


def _normalize_tushare_hist(
    raw: pd.DataFrame,
    meta: pd.DataFrame,
    daily_basic: pd.DataFrame,
    fina: pd.DataFrame,
) -> pd.DataFrame:
    df = raw.copy()
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    df = df.sort_values("date")
    meta_row = meta.iloc[0].to_dict() if not meta.empty else {}
    basic = daily_basic.copy()
    if not basic.empty:
        basic["date"] = pd.to_datetime(basic["trade_date"], format="%Y%m%d", errors="coerce")
        df = df.merge(basic, on=["ts_code", "trade_date", "date"], how="left", suffixes=("", "_basic"))

    roe = 0.08
    growth = 0.0
    if not fina.empty:
        latest = fina.sort_values("end_date").iloc[-1]
        roe = pd.to_numeric(latest.get("roe_dt", latest.get("roe", 8.0)), errors="coerce") / 100
        growth = pd.to_numeric(latest.get("netprofit_yoy", 0.0), errors="coerce") / 100

    list_date = pd.to_datetime(meta_row.get("list_date"), format="%Y%m%d", errors="coerce")
    list_days = (df["date"] - list_date).dt.days if pd.notna(list_date) else 9999
    amount = pd.to_numeric(df.get("amount", 0.0), errors="coerce") * 1000
    total_mv = pd.to_numeric(df.get("total_mv", np.nan), errors="coerce") * 10000

    return pd.DataFrame(
        {
            "date": df["date"],
            "code": df["ts_code"],
            "name": meta_row.get("name", df["ts_code"].iloc[0]),
            "industry": meta_row.get("industry", "未知"),
            "board": meta_row.get("board", _board_from_symbol(df["ts_code"].iloc[0])),
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "volume": pd.to_numeric(df.get("vol", 0.0), errors="coerce") * 100,
            "amount": amount,
            "turnover_rate": pd.to_numeric(df.get("turnover_rate", np.nan), errors="coerce").fillna(0.0),
            "money_flow": 0.0,
            "pe_ttm": pd.to_numeric(df.get("pe_ttm", np.nan), errors="coerce").fillna(25.0),
            "roe": roe if pd.notna(roe) else 0.08,
            "net_profit_growth": growth if pd.notna(growth) else 0.0,
            "market_cap": total_mv.fillna(10_000_000_000.0),
            "is_st": str(meta_row.get("name", "")).upper().find("ST") >= 0,
            "list_days": list_days,
            "suspended": False,
        }
    ).dropna(subset=["date", "close"])


def _safe_tushare_daily_basic(pro, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        return pro.daily_basic(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields="ts_code,trade_date,turnover_rate,pe_ttm,total_mv",
        )
    except Exception:
        return pd.DataFrame()


def _safe_tushare_fina_indicator(pro, ts_code: str) -> pd.DataFrame:
    try:
        return pro.fina_indicator(
            ts_code=ts_code,
            start_date=f"{date.today().year - 3}0101",
            end_date=date.today().strftime("%Y%m%d"),
            fields="ts_code,end_date,roe,roe_dt,netprofit_yoy",
        )
    except Exception:
        return pd.DataFrame()


def _suffix_code(symbol: str) -> str:
    symbol = str(symbol).split(".")[0].zfill(6)
    if symbol.startswith(("6", "9", "688")):
        return f"{symbol}.SH"
    return f"{symbol}.SZ"


def _tx_symbol(symbol: str) -> str:
    plain = _plain_symbol(symbol)
    prefix = "sh" if _suffix_code(plain).endswith(".SH") else "sz"
    return f"{prefix}{plain}"


def known_stock_identity(symbol: str) -> dict[str, str]:
    plain = _plain_symbol(symbol)
    return {
        "code": _suffix_code(plain),
        "name": KNOWN_A_SHARE_NAMES.get(plain, plain),
        "board": _board_from_symbol(plain),
    }


def _board_from_symbol(symbol: str) -> str:
    plain = _plain_symbol(symbol)
    if plain.startswith("688"):
        return "科创板"
    if plain.startswith(("600", "601", "603", "605")):
        return "上证主板"
    if plain.startswith("300"):
        return "创业板"
    if plain.startswith(("000", "001", "002", "003")):
        return "深证主板"
    return "未知"


def _normalize_sz_board(board: str) -> str:
    text = str(board).strip()
    if "创业" in text:
        return "创业板"
    if "主板" in text or text in {"中小板", "中小企业板"}:
        return "深证主板"
    return text or "深证主板"


def _filter_boards(universe: pd.DataFrame, boards: tuple[str, ...]) -> pd.DataFrame:
    if not boards or "全市场" in boards or "board" not in universe.columns:
        return universe
    filtered = universe[universe["board"].isin(boards)].copy()
    return filtered if not filtered.empty else universe


def _core_fallback_universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": CORE_A_SHARE_POOL,
            "name": [CORE_A_SHARE_NAMES.get(code, code) for code in CORE_A_SHARE_POOL],
            "industry": "核心池",
            "board": [_board_from_symbol(code) for code in CORE_A_SHARE_POOL],
            "listing_date": pd.NaT,
        }
    )


def _universe_is_usable(universe: pd.DataFrame) -> bool:
    if universe.empty or "board" not in universe.columns:
        return False
    boards = set(universe["board"].dropna().astype(str))
    return bool({"上证主板", "深证主板"}.issubset(boards) and ({"创业板", "科创板"} & boards))


def _select_symbols_by_board(universe: pd.DataFrame, max_symbols: int, boards: tuple[str, ...]) -> list[str]:
    if universe.empty:
        return []
    active_boards = [board for board in boards if board in set(universe["board"].astype(str))]
    if not active_boards:
        return universe["code"].head(max_symbols).astype(str).tolist()

    per_board = max(1, int(np.ceil(max_symbols / len(active_boards))))
    selected: list[str] = []
    for board in active_boards:
        selected.extend(universe.loc[universe["board"] == board, "code"].head(per_board).astype(str).tolist())
    if len(selected) < max_symbols:
        for code in universe["code"].astype(str):
            if code not in selected:
                selected.append(code)
            if len(selected) >= max_symbols:
                break
    return selected[:max_symbols]


def _select_request_symbols(universe: pd.DataFrame, request: DataRequest) -> list[str]:
    if universe.empty:
        return _merge_symbols([], request.extra_symbols)
    base = _select_symbols_by_board(universe, request.max_symbols, request.boards)
    return _merge_symbols(base, request.extra_symbols)


def _plain_symbol(symbol: str) -> str:
    text = str(symbol).strip().upper()
    if "." in text:
        text = text.split(".")[0]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(6) if digits else ""


def _merge_symbols(base: list[str], extra: tuple[str, ...]) -> list[str]:
    merged: list[str] = []
    for symbol in [*extra, *base]:
        plain = _plain_symbol(symbol)
        if plain and plain not in merged:
            merged.append(plain)
    return merged


def _inject_sample_extras(data: pd.DataFrame, extra: tuple[str, ...]) -> pd.DataFrame:
    if not extra:
        return data
    frames = [data]
    template_code = data["code"].iloc[0]
    template = data[data["code"] == template_code].copy()
    existing = set(data["code"].unique())
    for symbol in extra:
        code = _suffix_code(symbol)
        if code in existing:
            continue
        injected = template.copy()
        injected["code"] = code
        injected["name"] = _plain_symbol(symbol)
        injected["industry"] = "样例占位"
        injected["board"] = _board_from_symbol(symbol)
        frames.append(injected)
    return pd.concat(frames, ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True)
