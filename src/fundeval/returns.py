"""收益衡量：正文第二部分“收益衡量与计算口径”。

- TWR：各无外部现金流子期收益几何连接，衡量组合管理表现
- MWR：内部收益率 IRR / XIRR，衡量投资者的实际资金体验
- 两种超额收益：累计收益差额（百分点）与几何相对收益
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from fundeval._utils import as_series, paired


def wealth_index(returns, start: float = 1.0) -> pd.Series:
    """从 start 开始累计连乘得到的财富指数 W_t。"""
    r = as_series(returns)
    return start * (1 + r).cumprod()


def cumulative_return(returns) -> float:
    """累计收益（TWR）：∏(1 + r_j) - 1。"""
    r = as_series(returns)
    return float(np.prod(1 + r.to_numpy()) - 1)


twr = cumulative_return


def annualize_return(total_return: float, years: float) -> float:
    """区间累计收益年化：(1 + R)^(1/T) - 1，T 为年数。"""
    if years <= 0:
        raise ValueError("年数 T 必须为正")
    return float((1 + total_return) ** (1 / years) - 1)


def annualized_return(returns, periods_per_year: int = 12) -> float:
    """n 个等长观测期的年化几何收益，T = n / K。

    按现金流拆成的不等长子期不能用本函数，应以实际年数调用 annualize_return。
    """
    r = as_series(returns)
    if len(r) == 0:
        return float("nan")
    return annualize_return(cumulative_return(r), len(r) / periods_per_year)


def active_returns(portfolio, benchmark) -> pd.Series:
    """单期主动收益 a_t = r_p,t - r_b,t。"""
    p, b = paired(portfolio, benchmark)
    return (p - b).rename("active")


def cumulative_difference(portfolio, benchmark) -> float:
    """累计收益差额：R_p - R_b，单位为百分点（以小数表示）。"""
    p, b = paired(portfolio, benchmark)
    return cumulative_return(p) - cumulative_return(b)


def relative_return(portfolio, benchmark) -> float:
    """几何相对收益：(1 + R_p) / (1 + R_b) - 1。"""
    p, b = paired(portfolio, benchmark)
    return float((1 + cumulative_return(p)) / (1 + cumulative_return(b)) - 1)


def _solve_rate(npv, lower: float = -0.9999, upper: float = 10.0) -> float:
    """在 (lower, upper) 内求净现值为零的收益率；区间内无变号时返回 NaN。"""
    grid = np.concatenate([np.linspace(lower, 0, 200, endpoint=False), np.geomspace(1e-6, upper, 200)])
    vals = np.array([npv(x) for x in grid])
    roots = []
    for i in range(len(grid) - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] <= 0:
            roots.append(brentq(npv, grid[i], grid[i + 1], xtol=1e-14, maxiter=500))
    if not roots:
        return float("nan")
    # 多个根时取最接近零的一个，并由调用方复核现金流方向
    return float(min(roots, key=abs))


def irr(cashflows: Iterable[float]) -> float:
    """等间隔现金流的每期内部收益率：0 = Σ CF_j / (1 + i)^j。

    按投资者视角，投入为负、收回为正；期末剩余资产作为最后一笔正现金流。
    """
    cf = np.asarray(list(cashflows), dtype=float)
    if not ((cf > 0).any() and (cf < 0).any()):
        raise ValueError("现金流须同时包含正负值")
    t = np.arange(len(cf))
    return _solve_rate(lambda i: float(np.sum(cf / (1 + i) ** t)))


def xirr(cashflows: pd.Series | Iterable[tuple], day_count: float = 365.0) -> float:
    """不规则日期的资金加权年化收益：0 = Σ CF_j / (1 + i)^((d_j - d_0)/365)。

    ``cashflows`` 为以日期为索引的 Series，或 (日期, 金额) 序列。同日多笔会合并。
    """
    if not isinstance(cashflows, pd.Series):
        pairs = list(cashflows)
        cashflows = pd.Series([a for _, a in pairs], index=pd.to_datetime([d for d, _ in pairs]))
    cf = cashflows.astype(float).groupby(pd.to_datetime(cashflows.index)).sum().sort_index()
    if not ((cf > 0).any() and (cf < 0).any()):
        raise ValueError("现金流须同时包含正负值")
    years = ((cf.index - cf.index[0]).days / day_count).to_numpy(dtype=float)
    amounts = cf.to_numpy()
    return _solve_rate(lambda i: float(np.sum(amounts / (1 + i) ** years)))


def mwr(cashflows, dates: Iterable | None = None) -> float:
    """资金加权收益。给出日期时用 XIRR（年化），否则按等间隔 IRR（每期）。"""
    if dates is None and not isinstance(cashflows, pd.Series):
        return irr(cashflows)
    if dates is not None:
        cashflows = pd.Series(list(cashflows), index=pd.to_datetime(list(dates)))
    return xirr(cashflows)
