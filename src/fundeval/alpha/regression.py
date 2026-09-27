"""通过回归估计 Alpha 与统计不确定性：正文第四部分第 2 节。

    r_p,t - r_f,t = α + Σ β_k F_k,t + ε_t,    t_α = α̂ / SE(α̂)

报告 Alpha、各因子系数、标准误、t 值、p 值、95% 置信区间、R² 与样本期数 n。
收益存在异方差或自相关时，给出 ``hac_lags`` 改用 Newey–West（HAC）稳健标准误，
结果中记录滞后阶数。

正且显著的 Alpha 只支持“模型下存在正截距”，不足以排除模型遗漏、样本偏差或运气；
|t| ≈ 1.96 只是大样本双侧 5% 的参考，实际阈值须结合自由度与多重比较调整。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from fundeval._utils import as_series, broadcast_like

ALPHA = "alpha"


@dataclass(frozen=True)
class RegressionResult:
    """回归结果。系数按 ``alpha`` 在前、各因子在后排列，收益单位与输入同频。

    ``cov_type`` 为 ``"nonrobust"``（普通 OLS 标准误）或 ``"HAC"``（Newey–West），
    ``hac_lags`` 为 HAC 滞后阶数，普通 OLS 时为 None。``use_t`` 记录 p 值与置信区间实际
    采用的分布：True 为自由度 n - k - 1 的 t 分布，False 为标准正态近似。默认情况下
    普通 OLS 用 t 分布，HAC 沿用 statsmodels 默认的正态近似。
    """

    params: pd.Series
    bse: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    conf_int: pd.DataFrame
    rsquared: float
    rsquared_adj: float
    n: int
    cov_type: str
    hac_lags: int | None
    use_t: bool
    resid: pd.Series = field(repr=False)
    model_result: Any = field(repr=False, compare=False)

    @property
    def alpha(self) -> float:
        """每期 Alpha（截距），与输入收益同频。"""
        return float(self.params[ALPHA])

    @property
    def alpha_se(self) -> float:
        return float(self.bse[ALPHA])

    @property
    def alpha_t(self) -> float:
        return float(self.tvalues[ALPHA])

    @property
    def alpha_p(self) -> float:
        return float(self.pvalues[ALPHA])

    @property
    def betas(self) -> pd.Series:
        """各因子系数（不含截距）。"""
        return self.params.drop(ALPHA)

    def annualized_alpha(self, periods_per_year: int) -> float:
        """算术年化 Alpha = α × K。

        这是按期数线性放大的算术口径，不是几何年化 (1 + α)^K - 1；
        报告时须标明口径与数据频率。
        """
        return self.alpha * periods_per_year

    def table(self) -> pd.DataFrame:
        """系数表：coef、se、t、p 与 95% 置信区间上下限。"""
        return pd.DataFrame(
            {
                "coef": self.params,
                "se": self.bse,
                "t": self.tvalues,
                "p": self.pvalues,
                "ci_lower": self.conf_int["lower"],
                "ci_upper": self.conf_int["upper"],
            }
        )


def _factor_frame(factors) -> pd.DataFrame:
    if isinstance(factors, pd.DataFrame):
        f = factors.astype(float)
    elif isinstance(factors, pd.Series):
        f = factors.astype(float).to_frame(factors.name if factors.name is not None else "factor")
    else:
        arr = np.asarray(factors, dtype=float)
        if arr.ndim == 1:
            f = pd.DataFrame({"factor": arr})
        elif arr.ndim == 2:
            f = pd.DataFrame(arr, columns=[f"f{i + 1}" for i in range(arr.shape[1])])
        else:
            raise ValueError("因子须为一维或二维数据")
    f.columns = [str(c) for c in f.columns]
    if ALPHA in f.columns:
        raise ValueError(f"因子名不能为 {ALPHA!r}，该名称保留给截距")
    if f.columns.duplicated().any():
        raise ValueError("因子名重复")
    return f


def _align(returns, factors) -> tuple[pd.Series, pd.DataFrame]:
    """收益与因子对齐：两者都带 pandas 索引时要求观测期一致，否则按位置配对。"""
    f = _factor_frame(factors)
    if isinstance(returns, pd.Series) and isinstance(factors, (pd.Series, pd.DataFrame)):
        r = returns.astype(float)
        if len(r.index.intersection(f.index, sort=False)) != len(r) or len(r) != len(f):
            raise ValueError("收益与因子的观测期不一致，请先用 fundeval.etl.clean.align 对齐")
        return r, f.reindex(r.index)
    r = as_series(returns)
    if len(r) != len(f):
        raise ValueError(f"长度不一致：收益 {len(r)} 期，因子 {len(f)} 期")
    if isinstance(returns, pd.Series):
        f.index = r.index
    else:
        r.index = f.index
    return r, f


def _check_lags(hac_lags) -> int | None:
    if hac_lags is None:
        return None
    if isinstance(hac_lags, bool) or int(hac_lags) != hac_lags or hac_lags < 0:
        raise ValueError(f"hac_lags 须为非负整数或 None，收到 {hac_lags}")
    return int(hac_lags)


def _check_use_t(use_t) -> bool | None:
    if use_t is None or isinstance(use_t, (bool, np.bool_)):
        return None if use_t is None else bool(use_t)
    raise ValueError(f"use_t 须为 True、False 或 None，收到 {use_t!r}")


def fit_ols(
    y: pd.Series, x: pd.DataFrame, hac_lags: int | None = None, use_t: bool | None = None
) -> RegressionResult:
    """对 y = α + x·β + ε 做 OLS，返回 RegressionResult。

    y 与 x 须已对齐；任一变量缺失的期整行剔除（不补零），n 为实际参与回归的期数。
    供 factor_regression 与 attribution.timing 复用。

    ``use_t`` 决定 p 值与置信区间所用的分布：None 保持 statsmodels 默认（普通 OLS 用
    t 分布，HAC 用标准正态近似）；True 时 HAC 也改用自由度 n - k - 1 的 t 分布；
    False 时两者都用正态近似。小样本下正态近似的尾部偏薄，会低估 p 值、收窄置信区间，
    即高估显著性；月度样本只有几十期时，HAC 建议设 ``use_t=True``。
    """
    lags = _check_lags(hac_lags)
    use_t = _check_use_t(use_t)
    fit_kwargs = {} if use_t is None else {"use_t": use_t}
    data = pd.concat([y.rename("__y__"), x], axis=1).dropna()
    k = x.shape[1]
    if len(data) <= k + 1:
        raise ValueError(f"有效样本 {len(data)} 期，不足以估计 {k + 1} 个系数的标准误")
    exog = sm.add_constant(data[x.columns], prepend=True, has_constant="add").rename(
        columns={"const": ALPHA}
    )
    model = sm.OLS(data["__y__"], exog)
    if lags is None:
        res = model.fit(**fit_kwargs)
        cov_type = "nonrobust"
    else:
        res = model.fit(cov_type="HAC", cov_kwds={"maxlags": lags}, **fit_kwargs)
        cov_type = "HAC"
    ci = res.conf_int(alpha=0.05)
    ci.columns = ["lower", "upper"]
    return RegressionResult(
        params=res.params.rename("coef"),
        bse=res.bse.rename("se"),
        tvalues=res.tvalues.rename("t"),
        pvalues=res.pvalues.rename("p"),
        conf_int=ci,
        rsquared=float(res.rsquared),
        rsquared_adj=float(res.rsquared_adj),
        n=int(res.nobs),
        cov_type=cov_type,
        hac_lags=lags,
        use_t=bool(res.use_t),
        resid=res.resid.rename("resid"),
        model_result=res,
    )


def excess_frame(returns, factors, risk_free=0.0) -> tuple[pd.Series, pd.DataFrame]:
    """对齐收益与因子并返回 (r_p - r_f, 因子)。risk_free 可为常数或同期序列。"""
    r, f = _align(returns, factors)
    return (r - broadcast_like(risk_free, r)).rename("excess"), f


def factor_regression(
    returns, factors, risk_free=0.0, hac_lags: int | None = None, use_t: bool | None = None
) -> RegressionResult:
    """多因子时间序列回归 r_p,t - r_f,t = α + Σ β_k F_k,t + ε_t。

    ``factors`` 为 DataFrame（每列一个因子，列名即系数名）、Series 或数组。
    因子按原样进入回归：市场因子应已是超额收益（如 MKT = r_m - r_f），SMB、HML 等
    多空因子无需再减无风险收益。只有组合收益减去 ``risk_free``。

    ``hac_lags`` 为 None 时用普通 OLS 标准误；为整数 L 时用 Newey–West（HAC，
    Bartlett 核，最大滞后 L）稳健标准误，结果的 ``hac_lags`` 记录 L。

    ``use_t`` 为 None 时保持默认（OLS 用 t 分布，HAC 用正态近似）；为 True 时 HAC 的
    p 值与置信区间也用 t 分布。小样本下正态近似会高估显著性，详见 fit_ols。

    任一变量缺失的期整行剔除，结果的 ``n`` 为实际样本期数。
    """
    y, f = excess_frame(returns, factors, risk_free)
    return fit_ols(y, f, hac_lags, use_t)


def capm_regression(
    returns, market, risk_free=0.0, hac_lags: int | None = None, use_t: bool | None = None
) -> RegressionResult:
    """CAPM 单因子回归 r_p,t - r_f,t = α + β (r_m,t - r_f,t) + ε_t。

    ``market`` 为市场（或基准代理）的原始收益，函数内部同样减去 ``risk_free``。
    截距与斜率与 fundeval.risk.jensen_alpha、fundeval.risk.beta 在同一数据上完全一致
    （OLS 恒等式），回归额外给出标准误与置信区间。

    ``hac_lags`` 与 ``use_t`` 的含义同 factor_regression；小样本 HAC 建议 ``use_t=True``，
    否则正态近似会高估显著性。
    """
    r, m = _align(returns, market)
    m_col = m.iloc[:, 0]
    rf = broadcast_like(risk_free, r)
    y = (r - rf).rename("excess")
    x = pd.DataFrame({"market": m_col - rf})
    return fit_ols(y, x, hac_lags, use_t)
