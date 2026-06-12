from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec

import pandas as pd

from .sample_data import make_sample_market


@dataclass(frozen=True)
class ProviderStatus:
    mode: str
    message: str
    rows: int


class SampleProvider:
    name = "sample"

    def load_market(self) -> tuple[pd.DataFrame, ProviderStatus]:
        data = make_sample_market()
        return data, ProviderStatus("sample", "使用内置样例数据；结果不是现实市场推荐。", len(data))


class ProviderRouter:
    """Choose data source. Real providers are intentionally optional in v0.1."""

    def __init__(self, prefer_tushare: bool = False, tushare_token: str | None = None):
        self.prefer_tushare = prefer_tushare
        self.tushare_token = tushare_token

    def load_market(self) -> tuple[pd.DataFrame, ProviderStatus]:
        if self.prefer_tushare and self.tushare_token and find_spec("tushare"):
            return SampleProvider().load_market()
        if find_spec("akshare"):
            return SampleProvider().load_market()
        return SampleProvider().load_market()

    def availability(self) -> dict[str, str]:
        return {
            "AKShare": "已安装" if find_spec("akshare") else "未安装，当前使用样例数据",
            "Tushare": "已安装" if find_spec("tushare") else "未安装，当前使用样例数据",
        }

