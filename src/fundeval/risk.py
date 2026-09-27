"""风险调整后的绩效衡量：正文第三部分。

口径：比率使用同频收益的算术均值与样本标准差（n-1 分母，对应 Excel STDEV.S），
乘以 √K 年化；年化累计收益使用几何口径（见 fundeval.returns）。
比率分母为零时返回 NaN，表示报告不适用。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fundeval._utils import as_series, broadcast_like, paired, safe_div
from fundeval.returns import wealth_index


def _std(x: pd.Series) -> float:
    return float(x.std(ddof=1)) if len(x) > 1 else float("nan")


def excess_returns(returns, risk_free=0.0) -> pd.Series:
    """超额无风险收益 e_t = r_p,t - r_f,t。risk_free 可为常数或同期序列。"""
    r = as_series(returns)
    return (r - broadcast_like(risk_free, r)).rename("excess")


def volatility(returns, periods_per_year: int = 12) -> float:
    """年化波动率 σ_ann = s(r) √K。"""
    r = as_series(returns)
    return _std(r) * np.sqrt(periods_per_year)


def drawdown(returns) -> pd.Series:
    """回撤序列 W_t / max_{s≤t} W_s - 1（非正），历史峰值包含初始值 1。"""
    w = wealth_index(returns)
    peak = w.cummax().clip(lower=1.0)
    return (w / peak - 1).rename("drawdown")


def max_drawdown(returns) -> float:
    """最大回撤，以正的损失幅度报告。须同时披露观察频率。"""
    dd = drawdown(returns)
    return float(-dd.min()) if len(dd) else float("nan")


def sharpe_ratio(returns, risk_free=0.0, periods_per_year: int = 12) -> float:
    """年化 Sharpe = ē / s(e) × √K，分母为超额收益的样本标准差。"""
    e = excess_returns(returns, risk_free)
    return safe_div(float(e.mean()), _std(e)) * np.sqrt(periods_per_year)


def tracking_error(portfolio, benchmark, periods_per_year: int = 12) -> float:
    """年化跟踪误差 TE = s(a) √K。"""
    p, b = paired(portfolio, benchmark)
    return _std(p - b) * np.sqrt(periods_per_year)


def information_ratio(portfolio, benchmark, periods_per_year: int = 12) -> float:
    """年化信息比率 IR = ā / s(a) × √K。"""
    p, b = paired(portfolio, benchmark)
    a = p - b
    return safe_div(float(a.mean()), _std(a)) * np.sqrt(periods_per_year)


def beta(portfolio, market, risk_free=0.0) -> float:
    """市场 Beta：超额收益的样本协方差除以市场超额收益的样本方差（即 OLS 斜率）。"""
    p, m = paired(portfolio, market)
    ep, em = excess_returns(p, risk_free), excess_returns(m, risk_free)
    return safe_div(float(np.cov(ep, em, ddof=1)[0, 1]), float(em.var(ddof=1)))


def treynor_ratio(portfolio, market, risk_free=0.0, periods_per_year: int = 12) -> float:
    """年化 Treynor = K ē / β_p。Beta 接近零或为负时不宜机械排名。"""
    p, m = paired(portfolio, market)
    e = excess_returns(p, risk_free)
    return safe_div(periods_per_year * float(e.mean()), beta(p, m, risk_free))


def jensen_alpha(portfolio, market, risk_free=0.0) -> float:
    """每期 Jensen Alpha：mean(r_p - r_f) - β mean(r_m - r_f)。

    这是均值口径的点估计；显著性与稳健标准误由后续 alpha 模块的回归给出。
    """
    p, m = paired(portfolio, market)
    b = beta(p, m, risk_free)
    return float(excess_returns(p, risk_free).mean() - b * excess_returns(m, risk_free).mean())


def m_squared(portfolio, benchmark, risk_free=0.0, periods_per_year: int = 12) -> float:
    """M² = μ_f + (σ_b / σ_p)(μ_p - μ_f)。

    μ 为年化算术均值（每期均值 × K），σ 为年化波动率；无风险收益视为稳定。
    """
    p, b = paired(portfolio, benchmark)
    k = periods_per_year
    mu_p = float(p.mean()) * k
    mu_f = float(broadcast_like(risk_free, p).mean()) * k
    ratio = safe_div(volatility(b, k), volatility(p, k))
    return mu_f + ratio * (mu_p - mu_f)


def m_squared_difference(portfolio, benchmark, risk_free=0.0, periods_per_year: int = 12) -> float:
    """M² 相对差额 = M² - μ_b，μ_b 为基准年化算术均值（不是累计收益）。"""
    p, b = paired(portfolio, benchmark)
    return m_squared(p, b, risk_free, periods_per_year) - float(b.mean()) * periods_per_year


def summary(portfolio, benchmark=None, risk_free=0.0, periods_per_year: int = 12) -> pd.Series:
    """汇总第九部分表格中的收益与风险指标。"""
    from fundeval import returns as ret

    p = as_series(portfolio)
    k = periods_per_year
    out = {
        "cumulative_return": ret.cumulative_return(p),
        "annualized_return": ret.annualized_return(p, k),
        "volatility": volatility(p, k),
        "sharpe": sharpe_ratio(p, risk_free, k),
        "max_drawdown": max_drawdown(p),
    }
    if benchmark is not None:
        p, b = paired(p, benchmark)
        out.update(
            {
                "benchmark_cumulative_return": ret.cumulative_return(b),
                "cumulative_difference": ret.cumulative_difference(p, b),
                "relative_return": ret.relative_return(p, b),
                "tracking_error": tracking_error(p, b, k),
                "information_ratio": information_ratio(p, b, k),
                "m_squared": m_squared(p, b, risk_free, k),
                "m_squared_difference": m_squared_difference(p, b, risk_free, k),
            }
        )
    return pd.Series(out, name="value")
