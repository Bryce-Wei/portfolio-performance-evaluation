"""外币指数收益换算为人民币收益：正文第一部分“基准选择”（QDII 与含境外成分的合同基准）。

基金净值以人民币计价，合同基准中的境外成分（如中证香港300，港元计价）须先换算为人民币收益，
否则基准收益含汇率差异。换算公式：

    r_CNY,t = (1 + r_local,t) × (S_t / S_{t−1}) − 1

S 为每单位外币折合的人民币（国家外汇管理局人民币汇率中间价 ÷ 100，见 etl.sources.akshare.fx_rates）。
先在日度上换算，再用 etl.returns.to_frequency 按期内复利合成周、月等低频收益。

指数交易日与中间价发布日不一致（如香港交易、内地休市）时，S_t 取不晚于 t 的最近一个中间价（asof），
不使用未来数据；t 之前没有任何中间价时 S_t 为 NaN，相应的收益为 NaN，不填补。
"""

from __future__ import annotations

import pandas as pd

from fundeval.etl import schema

#: 汇率换算的方法、数据源与限定语
FX_CONVERT = "convert"
FX_NONE = "none"
FX_MODES = (FX_CONVERT, FX_NONE)
FX_SOURCE_LABEL = "国家外汇管理局人民币汇率中间价"


def converted_caveat(currencies) -> str:
    """换算后的口径文字，如“港元指数按国家外汇管理局人民币汇率中间价换算为人民币收益”。"""
    names = "、".join(dict.fromkeys(str(c) for c in currencies))
    return f"{names}指数按{FX_SOURCE_LABEL}换算为人民币收益"


def check_mode(mode: str) -> str:
    """校验 --fx：convert 或 none。"""
    if mode not in FX_MODES:
        raise ValueError(f"--fx 须为 convert 或 none，收到 {mode!r}")
    return mode


def fx_asof(rates: pd.Series, dates) -> pd.Series:
    """每个日期取不晚于该日的最近一个中间价（asof）；该日之前没有中间价时为 NaN。不使用未来数据。"""
    if not isinstance(rates.index, pd.DatetimeIndex):
        raise schema.SchemaError("汇率序列的索引必须是 DatetimeIndex")
    s = rates.astype(float).dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    target = pd.DatetimeIndex(dates)
    if s.empty:
        return pd.Series(float("nan"), index=target)
    return pd.Series(s.asof(target).to_numpy(dtype=float), index=target)


def convert_returns(local: pd.Series, rates: pd.Series, base_date=None) -> pd.Series:
    """把外币计价的日度收益换算为人民币收益：r_CNY,t = (1 + r_local,t) × (S_t / S_{t−1}) − 1。

    ``local`` 为外币计价的单期收益（索引为指数交易日），``rates`` 为每单位外币折合人民币的中间价序列。
    S_t 与 S_{t−1} 分别取 t 与上一个交易日的 asof 中间价（fx_asof）。``base_date`` 为第一个收益的上一个
    交易日（即价格序列的首日）；不给出时第一个观测没有上一个交易日，结果为 NaN。
    缺失值不填补，原样为 NaN。``local`` 的 attrs 原样保留。
    """
    if not isinstance(local.index, pd.DatetimeIndex):
        raise schema.SchemaError("收益的索引必须是 DatetimeIndex")
    r = local.astype(float).sort_index()
    s = fx_asof(rates, r.index)
    prev = s.shift(1)
    if base_date is not None and len(r):
        base = pd.Timestamp(base_date)
        if base >= r.index[0]:
            raise ValueError(f"base_date {base:%Y-%m-%d} 须早于第一个收益日期 {r.index[0]:%Y-%m-%d}")
        prev.iloc[0] = fx_asof(rates, [base]).iloc[0]
    ratio = s / prev.where(prev > 0)  # 中间价非正（数据错误）时为 NaN，同 _utils.safe_div 的约定
    out = (1 + r) * ratio - 1
    out.name = local.name
    out.attrs = dict(local.attrs)
    return out


__all__ = [
    "FX_CONVERT", "FX_NONE", "FX_MODES", "FX_SOURCE_LABEL", "check_mode", "converted_caveat", "fx_asof",
    "convert_returns",
]
