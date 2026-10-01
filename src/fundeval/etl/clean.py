"""数据清洗：日期对齐、币种统一、现金流识别、缺失与异常复核（正文第一部分第 2 节的清洗顺序）。

正文要求：停牌或估值滞后不能直接当作低风险证据；组合和基准不同交易日的缺失值
也不应机械填零。因此这里只对齐、标记和报告，不自动填补。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fundeval._utils import safe_div
from fundeval.etl import schema

#: 期中大额现金流的默认阈值：净申赎超过资产规模的 5%
LARGE_CASHFLOW_THRESHOLD = 0.05


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
    return pd.concat(series, axis=1, join=how, sort=True).sort_index()


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


def infer_cashflows(units: pd.Series, nav: pd.Series) -> pd.Series:
    """由份额变动推算净申赎现金流：NCF_t = (Units_t − Units_{t−1}) × NAV_t，流入（净申购）为正。

    ``units`` 为各期末总份额，``nav`` 为同日单位净值，两者须有相同的日期索引。首期没有上期份额，被去掉
    （与 etl.returns.nav_to_returns 一致）；任一侧缺失的期结果为 NaN，不填零。

    这是按期末净值的近似：申赎实际按申请日净值成交，期内多笔申赎会被合并；红利再投资、份额拆分或折算
    也会改变份额，会被误认为申赎，须结合分红与拆分记录复核。
    """
    if not isinstance(units.index, pd.DatetimeIndex) or not isinstance(nav.index, pd.DatetimeIndex):
        raise schema.SchemaError("units 与 nav 的索引必须是 DatetimeIndex")
    if not units.index.equals(nav.index):
        raise schema.SchemaError("units 与 nav 的日期须一致，请先用 align 对齐")
    if not units.index.is_monotonic_increasing or units.index.has_duplicates:
        raise schema.SchemaError("units 与 nav 的日期须唯一且升序")
    u = units.astype(float)
    flows = u.diff() * nav.astype(float)
    return flows.iloc[1:].rename(schema.CASHFLOW)


def flag_large_cashflows(
    flows: pd.Series, assets: pd.Series | float, threshold: float = LARGE_CASHFLOW_THRESHOLD
) -> pd.DataFrame:
    """标出超过资产规模一定比例的期中大额现金流，供拆分子期或人工复核。

    正文第二部分第 1 节：“NCF 以流入账户为正，仅适用于现金流发生在期末的情形；期中大额现金流应按现金流
    发生时点拆分子期。”超过阈值的现金流若发生在期中，用期末现金流公式 r = (V_t − NCF_t) / V_{t−1} − 1
    会明显失真，应在现金流发生日补一个估值点，把该期拆成两个子期（etl.returns.value_to_returns）。

    ``assets`` 为资产规模（宜用期初，即上一期末的资产净值；标量或按 flows 日期对齐的序列），
    ``threshold`` 为比例阈值（默认 5%）。返回列 ``cashflow``、``assets``、``ratio``（|NCF| / 资产）与
    ``flagged``；现金流或资产缺失、资产为零时 ratio 为 NaN、flagged 为 False（缺失另行复核，不填零）。
    """
    if threshold <= 0:
        raise ValueError(f"threshold 须为正数，收到 {threshold}")
    f = flows.astype(float)
    a = pd.Series(float(assets), index=f.index) if np.isscalar(assets) else assets.astype(float).reindex(f.index)
    if (a.dropna() < 0).any():
        raise ValueError("资产规模不能为负")
    ratio = pd.Series([safe_div(abs(x), y) for x, y in zip(f.to_numpy(), a.to_numpy())], index=f.index, dtype=float)
    out = pd.DataFrame({"cashflow": f, "assets": a, "ratio": ratio})
    out["flagged"] = (out["ratio"] > threshold).fillna(False).astype(bool)
    out.attrs["threshold"] = threshold
    return out


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
