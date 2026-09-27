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
