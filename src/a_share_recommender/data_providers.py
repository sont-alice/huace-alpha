from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from importlib.util import find_spec
from pathlib import Path
import hashlib
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


@dataclass(frozen=True)
class ProviderStatus:
    mode: str
    message: str
    rows: int


@dataclass(frozen=True)
class DataRequest:
    max_symbols: int = 30
    history_years: int = 4
    use_finance: bool = True
    force_sample: bool = False
    force_refresh: bool = False


class SampleProvider:
    name = "sample"

    def load_market(self, request: DataRequest | None = None) -> tuple[pd.DataFrame, ProviderStatus]:
        data = make_sample_market()
        return data, ProviderStatus("sample", "使用内置样例数据；结果不是现实市场推荐。", len(data))


class AkshareProvider:
    name = "akshare"

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir

    def load_market(self, request: DataRequest) -> tuple[pd.DataFrame, ProviderStatus]:
        cache_path = _cache_path(self.cache_dir, "akshare", request)
        if cache_path.exists() and not request.force_refresh:
            data = pd.read_parquet(cache_path)
            return data, ProviderStatus("akshare-cache", f"使用 AKShare 本地缓存：{cache_path}", len(data))

        if not find_spec("akshare"):
            raise RuntimeError("未安装 akshare，请先运行 python -m pip install akshare")

        import akshare as ak

        universe = _akshare_universe(ak)
        symbols = universe["code"].head(request.max_symbols).tolist()
        if not symbols:
            symbols = CORE_A_SHARE_POOL[: request.max_symbols]

        start_date = (date.today() - timedelta(days=365 * request.history_years + 90)).strftime("%Y%m%d")
        end_date = date.today().strftime("%Y%m%d")

        frames = []
        errors: list[str] = []
        for symbol in symbols:
            try:
                raw = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq",
                )
                if raw.empty:
                    continue
                meta = universe.loc[universe["code"] == symbol].head(1)
                frames.append(_normalize_akshare_hist(raw, meta))
                time.sleep(0.08)
            except Exception as exc:
                errors.append(f"{symbol}:{type(exc).__name__}")

        if not frames:
            raise RuntimeError("AKShare 日线接口未返回可用数据：" + "；".join(errors[:5]))

        data = pd.concat(frames, ignore_index=True)
        if request.use_finance:
            data = _attach_akshare_finance(ak, data, symbols[: min(len(symbols), 30)])

        data = data.sort_values(["date", "code"]).reset_index(drop=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(cache_path, index=False)
        suffix = f"；失败 {len(errors)} 只" if errors else ""
        return data, ProviderStatus("akshare", f"使用 AKShare 真实日线/行业/财务数据{suffix}", len(data))


class TushareProvider:
    name = "tushare"

    def __init__(self, token: str, cache_dir: Path):
        self.token = token
        self.cache_dir = cache_dir

    def load_market(self, request: DataRequest) -> tuple[pd.DataFrame, ProviderStatus]:
        cache_path = _cache_path(self.cache_dir, "tushare", request)
        if cache_path.exists() and not request.force_refresh:
            data = pd.read_parquet(cache_path)
            return data, ProviderStatus("tushare-cache", f"使用 Tushare 本地缓存：{cache_path}", len(data))

        if not find_spec("tushare"):
            raise RuntimeError("未安装 tushare，请先运行 python -m pip install tushare")

        import tushare as ts

        ts.set_token(self.token)
        pro = ts.pro_api(self.token)
        universe = _tushare_universe(pro)
        symbols = universe["ts_code"].head(request.max_symbols).tolist()
        start_date = (date.today() - timedelta(days=365 * request.history_years + 90)).strftime("%Y%m%d")
        end_date = date.today().strftime("%Y%m%d")

        frames = []
        errors: list[str] = []
        for ts_code in symbols:
            try:
                raw = ts.pro_bar(ts_code=ts_code, adj="qfq", start_date=start_date, end_date=end_date)
                if raw is None or raw.empty:
                    continue
                meta = universe.loc[universe["ts_code"] == ts_code].head(1)
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

        data, status = SampleProvider().load_market(request)
        message = status.message + " 真实数据源不可用：" + " | ".join(self.last_errors[:3])
        return data, ProviderStatus("sample-fallback", message, len(data))

    def availability(self) -> dict[str, str]:
        return {
            "AKShare": "已安装" if find_spec("akshare") else "未安装",
            "Tushare": "已安装" if find_spec("tushare") else "未安装",
            "最近错误": " | ".join(self.last_errors) if self.last_errors else "无",
        }


def _cache_path(cache_dir: Path, provider: str, request: DataRequest) -> Path:
    key = f"{provider}-{request.max_symbols}-{request.history_years}-{request.use_finance}-{date.today():%Y%m%d}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return cache_dir / f"{provider}_{digest}.parquet"


def _akshare_universe(ak) -> pd.DataFrame:
    rows = []
    try:
        sh = ak.stock_info_sh_name_code()
        rows.append(
            pd.DataFrame(
                {
                    "code": sh["证券代码"].astype(str).str.zfill(6),
                    "name": sh["证券简称"].astype(str),
                    "industry": "沪市",
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
                    "industry": sz.get("所属行业", "深市").astype(str),
                    "listing_date": pd.to_datetime(sz["A股上市日期"], errors="coerce"),
                }
            )
        )
    except Exception:
        pass

    if rows:
        universe = pd.concat(rows, ignore_index=True)
        universe = universe[universe["code"].isin(CORE_A_SHARE_POOL)].drop_duplicates("code")
        core_rank = {code: i for i, code in enumerate(CORE_A_SHARE_POOL)}
        universe["rank"] = universe["code"].map(core_rank)
        return universe.sort_values("rank").drop(columns=["rank"]).reset_index(drop=True)

    return pd.DataFrame(
        {
            "code": CORE_A_SHARE_POOL,
            "name": CORE_A_SHARE_POOL,
            "industry": "未知",
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
    core_ts = {_suffix_code(code).replace(".SH", ".SH").replace(".SZ", ".SZ") for code in CORE_A_SHARE_POOL}
    universe = universe[universe["ts_code"].isin(core_ts)].copy()
    core_rank = {_suffix_code(code): i for i, code in enumerate(CORE_A_SHARE_POOL)}
    universe["rank"] = universe["ts_code"].map(core_rank)
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
