"""标准数据结构与校验。

记号约定沿用正文：p 组合，b 基准，m 市场，f 无风险资产；K 为一年期数。
收益表以日期为索引，列名使用下列常量；收益一律为小数。
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

DATE = "date"
PORTFOLIO = "portfolio"
BENCHMARK = "benchmark"
MARKET = "market"
RISK_FREE = "risk_free"
NAV = "nav"
PRICE = "price"
DIVIDEND = "dividend"
VALUE = "value"
CASHFLOW = "cashflow"

RETURN_COLUMNS = (PORTFOLIO, BENCHMARK, MARKET, RISK_FREE)

#: 一年期数 K。日度按 252 个交易日约定，如另有约定应显式传入。
PERIODS_PER_YEAR = {"D": 252, "W": 52, "M": 12, "Q": 4, "A": 1}


class SchemaError(ValueError):
    """数据不符合标准结构。"""


def periods_per_year(freq: str | int) -> int:
    """把频率代码（D/W/M/Q/A）或整数转换为 K。"""
    if isinstance(freq, (int, np.integer)):
        if freq <= 0:
            raise SchemaError(f"K 必须为正整数，收到 {freq}")
        return int(freq)
    key = str(freq).upper()[:1]
    if key == "Y":
        key = "A"
    if key not in PERIODS_PER_YEAR:
        raise SchemaError(f"无法识别的收益频率：{freq!r}")
    return PERIODS_PER_YEAR[key]


def _check_index(obj: pd.Series | pd.DataFrame, what: str) -> None:
    if not isinstance(obj.index, pd.DatetimeIndex):
        raise SchemaError(f"{what} 的索引必须是 DatetimeIndex")
    if obj.index.has_duplicates:
        dup = obj.index[obj.index.duplicated()].unique()
        raise SchemaError(f"{what} 存在重复日期：{list(dup[:5])}")
    if not obj.index.is_monotonic_increasing:
        raise SchemaError(f"{what} 的日期未按升序排列")


def validate_returns(
    df: pd.DataFrame,
    required: Iterable[str] = (PORTFOLIO,),
    allow_missing: bool = False,
) -> pd.DataFrame:
    """校验收益表：日期索引唯一升序、所需列存在、数值为有限小数且大于 -100%。"""
    _check_index(df, "收益表")
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise SchemaError(f"收益表缺少列：{missing_cols}")
    cols = [c for c in df.columns if c in RETURN_COLUMNS]
    values = df[cols].apply(pd.to_numeric, errors="raise")
    if not allow_missing and values[list(required)].isna().any().any():
        raise SchemaError("收益表必需列存在缺失值；缺失不应机械填零，请先复核")
    arr = values.to_numpy(dtype=float)
    finite = arr[~np.isnan(arr)]
    if np.isinf(finite).any():
        raise SchemaError("收益表存在无穷值")
    if (finite <= -1).any():
        raise SchemaError("存在小于等于 -100% 的收益，请确认是否误用了百分数")
    return df


def validate_nav(nav: pd.Series) -> pd.Series:
    """校验净值序列：日期索引唯一升序、严格为正、无缺失。"""
    _check_index(nav, "净值序列")
    if nav.isna().any():
        raise SchemaError("净值序列存在缺失值")
    if (nav <= 0).any():
        raise SchemaError("净值必须为正")
    return nav


def validate_cashflows(flows: pd.Series) -> pd.Series:
    """校验外部现金流序列：日期索引升序、数值有限。正值表示流入账户。"""
    if not isinstance(flows.index, pd.DatetimeIndex):
        raise SchemaError("现金流序列的索引必须是 DatetimeIndex")
    if not flows.index.is_monotonic_increasing:
        raise SchemaError("现金流序列的日期未按升序排列")
    if not np.isfinite(flows.to_numpy(dtype=float)).all():
        raise SchemaError("现金流序列存在缺失或无穷值")
    return flows
