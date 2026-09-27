"""数据清洗：日期对齐、币种统一、缺失与异常复核。

正文要求：停牌或估值滞后不能直接当作低风险证据；组合和基准不同交易日的缺失值
也不应机械填零。因此这里只对齐、标记和报告，不自动填补。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fundeval.etl import schema


def drop_duplicate_dates(obj: pd.Series | pd.DataFrame, keep: str = "last") -> pd.Series | pd.DataFrame:
    """按日期排序并去除重复日期（默认保留最后一条，视为更正值）。"""
    obj = obj.sort_index()
    return obj[~obj.index.duplicated(keep=keep)]


def align(*series: pd.Series, how: str = "inner", names: list[str] | None = None) -> pd.DataFrame:
    """把多条序列按日期对齐成一张表。

    默认取交集，只保留所有序列都有观测的日期；``how="outer"`` 保留并集，
    缺失处保持 NaN 以便复核，不填零。
    """
    if not series:
        raise ValueError("至少需要一条序列")
    if names is not None:
        if len(names) != len(series):
            raise ValueError("names 与序列数量不一致")
        series = tuple(s.rename(n) for s, n in zip(series, names))
    return pd.concat(series, axis=1, join=how).sort_index()


def align_returns(
    portfolio: pd.Series,
    benchmark: pd.Series | None = None,
    risk_free: pd.Series | float | None = None,
    market: pd.Series | None = None,
) -> pd.DataFrame:
    """对齐组合、基准、市场与无风险收益为标准收益表。

    组合与基准（及市场）取共同日期；无风险收益可为常数，或须覆盖全部共同日期。
    """
    parts = {schema.PORTFOLIO: portfolio}
    if benchmark is not None:
        parts[schema.BENCHMARK] = benchmark
    if market is not None:
        parts[schema.MARKET] = market
    df = align(*parts.values(), names=list(parts))
    if risk_free is not None:
        if np.isscalar(risk_free):
            df[schema.RISK_FREE] = float(risk_free)
        else:
            rf = risk_free.reindex(df.index)
            if rf.isna().any():
                missing = df.index[rf.isna()]
                raise schema.SchemaError(f"无风险收益未覆盖 {len(missing)} 个观测日：{list(missing[:5])}")
            df[schema.RISK_FREE] = rf
    return schema.validate_returns(df, required=list(parts))


def convert_currency(values: pd.Series, fx_rate: pd.Series | float) -> pd.Series:
    """把以原币计价的金额或净值换算为计价币种：values × fx_rate（1 单位原币折合计价币种）。

    换算须在计算收益之前完成；汇率须覆盖全部日期。
    """
    if np.isscalar(fx_rate):
        return values * float(fx_rate)
    fx = fx_rate.reindex(values.index)
    if fx.isna().any():
        missing = values.index[fx.isna()]
        raise schema.SchemaError(f"汇率未覆盖 {len(missing)} 个日期：{list(missing[:5])}")
    return values * fx


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """按列统计缺失：观测数、缺失数、缺失比例、首个缺失日期。"""
    rows = {}
    for col in df.columns:
        na = df[col].isna()
        rows[col] = {
            "observations": int(len(na)),
            "missing": int(na.sum()),
            "missing_ratio": float(na.mean()) if len(na) else float("nan"),
            "first_missing": df.index[na][0] if na.any() else pd.NaT,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def flag_outliers(returns: pd.Series, threshold: float = 5.0) -> pd.Series:
    """用稳健 z 值（中位数与 MAD）标记异常收益，返回布尔序列，供人工复核。

    MAD 为零（多数观测相同，例如停牌期间收益为零）时，凡偏离中位数的值都被标记。
    """
    r = returns.astype(float)
    med = r.median()
    mad = (r - med).abs().median() * 1.4826
    if mad == 0 or np.isnan(mad):
        return (r - med).abs() > 0
    return ((r - med).abs() / mad) > threshold


def flag_stale(values: pd.Series, min_run: int = 3) -> pd.Series:
    """标记连续 min_run 期及以上数值不变的区间（疑似停牌或估值滞后）。"""
    same = values.diff().eq(0)
    run_id = (~same).cumsum()
    run_len = values.groupby(run_id).transform("size")
    return run_len >= min_run
