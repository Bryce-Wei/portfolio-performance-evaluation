"""单期收益计算：正文第二部分第 1 节“先计算单期收益”。"""

from __future__ import annotations

import pandas as pd

from fundeval.etl import schema


def nav_to_returns(nav: pd.Series) -> pd.Series:
    """复权单位净值到单期收益：r_t = NAV_t / NAV_{t-1} - 1。首期无收益，被去掉。"""
    nav = schema.validate_nav(nav.astype(float))
    return (nav / nav.shift(1) - 1).iloc[1:].rename(schema.PORTFOLIO)


def price_to_returns(price: pd.Series, dividends: pd.Series | None = None) -> pd.Series:
    """未复权价格加现金分配到单期总收益：r_t = (P_t + D_t) / P_{t-1} - 1。

    ``dividends`` 按除息日记录每单位现金分配，缺省日期视为无分配；仅在未复权价格上使用，
    避免对已复权净值重复计入分红。
    """
    price = schema.validate_nav(price.astype(float))
    div = pd.Series(0.0, index=price.index)
    if dividends is not None:
        extra = dividends.index.difference(price.index)
        if len(extra):
            raise schema.SchemaError(f"分红日期不在价格序列中：{list(extra[:5])}")
        div = div.add(dividends.astype(float), fill_value=0.0)
    return ((price + div) / price.shift(1) - 1).iloc[1:].rename(schema.PORTFOLIO)


def value_to_returns(value: pd.Series, net_cashflow: pd.Series | None = None) -> pd.Series:
    """账户资产价值到单期收益，剔除外部现金流：r_t = (V_t - NCF_t) / V_{t-1} - 1。

    NCF 以流入账户为正，仅适用于现金流发生在期末的情形；期中大额现金流应按发生时点
    拆分子期，即在现金流发生日补一个估值点。
    """
    value = schema.validate_nav(value.astype(float))
    ncf = pd.Series(0.0, index=value.index)
    if net_cashflow is not None:
        flows = schema.validate_cashflows(net_cashflow.astype(float).groupby(level=0).sum())
        extra = flows.index.difference(value.index)
        if len(extra):
            raise schema.SchemaError(f"现金流日期没有对应估值，请先拆分子期：{list(extra[:5])}")
        ncf = ncf.add(flows, fill_value=0.0)
    return ((value - ncf) / value.shift(1) - 1).iloc[1:].rename(schema.PORTFOLIO)


def percent_to_decimal(obj: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """百分数转小数（2.0 → 0.02）。"""
    return obj / 100.0


#: to_frequency 支持的目标频率与 pandas Period 代码；周度按周五收盘结束
_PERIOD_CODES = {"W": "W-FRI", "M": "M", "Q": "Q", "A": "Y"}


def period_code(freq: str) -> str:
    """把 W/M/Q/A（Y）换算为 pandas Period 代码。"""
    key = str(freq).upper()[:1]
    key = "A" if key == "Y" else key
    if key not in _PERIOD_CODES:
        raise schema.SchemaError(f"to_frequency 只支持 W、M、Q、A，收到 {freq!r}")
    return _PERIOD_CODES[key]


def period_end_index(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """把日期映射到所属期的日历期末（如 2025-01-15 → 2025-01-31），用于跨数据源对齐。"""
    periods = pd.DatetimeIndex(index).to_period(period_code(freq))
    return periods.to_timestamp(how="end").normalize()


def to_frequency(
    returns: pd.Series | pd.DataFrame, freq: str, min_obs: int | None = None
) -> pd.Series | pd.DataFrame:
    """把日度（或更高频）单期收益按期内复利合成为周、月、季或年收益：R = ∏(1 + r) − 1。

    ``freq`` 为 "W"（周五结束的周）、"M"、"Q" 或 "A"。结果索引为日历期末日期
    （如 2025-01-31），便于基金、指数与无风险收益按同一标签对齐。

    缺失值不跳过：期内任一观测为 NaN，该期结果为 NaN，避免把缺失日当作零收益。
    ``min_obs`` 给出时，期内有效观测少于 min_obs 的期（如只含几天的首末不完整月份）
    同样为 NaN。输入为 DataFrame 时按列分别处理。
    """
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise schema.SchemaError("收益的索引必须是 DatetimeIndex")
    if min_obs is not None and (isinstance(min_obs, bool) or int(min_obs) != min_obs or min_obs < 1):
        raise ValueError(f"min_obs 须为正整数或 None，收到 {min_obs}")
    obj = returns.astype(float).sort_index()
    labels = period_end_index(obj.index, freq)
    grouped = (1 + obj).groupby(labels)
    out = grouped.prod(min_count=1) - 1
    has_missing = obj.isna().groupby(labels).any()
    out = out.mask(has_missing)
    if min_obs is not None:
        out = out.mask(obj.notna().groupby(labels).sum() < int(min_obs))
    out.index = pd.DatetimeIndex(out.index).rename(schema.DATE)
    return out
