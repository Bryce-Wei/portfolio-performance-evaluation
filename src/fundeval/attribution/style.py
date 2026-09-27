"""Sharpe 收益型风格分析：正文第五部分第 1 节。

用组合收益对一组风格指数（资产类别）做约束拟合，回答“组合的收益变化更像哪些市场或风格”：

    r_p,t = Σ w_k r_k,t + ε_t,    w_k ≥ 0,    Σ w_k = 1

- ``objective="variance"``（默认）：最小化残差方差 Var(ε)，与正文“以解释收益波动为目标”一致；
  此时残差均值相当于正文公式中的 α，不参与权重估计
- ``objective="sse"``：最小化残差平方和 Σ ε_t²（无截距的约束最小二乘）

拟合优度 R² = 1 − Var(ε) / Var(r_p)。风格权重是统计估计，不等于实际持仓；残差均值也不能直接
视为扣除一切风险后的选股能力（风格基准不完整、共线性与估计窗口都会影响它）。风格指数高度
重叠（如正文 SPY、SPYV、SPYG）时权重解释不稳定，本模块对两两相关系数高于 0.95 的风格发出警告。
组合与风格指数须采用一致的总收益口径（宜用全收益指数）。
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize

from fundeval._utils import broadcast_like, safe_div
from fundeval.alpha.regression import _align
from fundeval.etl.benchmark import CASH

OBJECTIVES = ("variance", "sse")

#: 两两相关系数高于该值视为共线（正文：风格基准高度重叠会使权重解释不稳定）
COLLINEARITY_THRESHOLD = 0.95

#: 约束的容差：权重下限与加总
_CONSTRAINT_TOL = 1e-6


class StyleNotConverged(RuntimeError):
    """约束优化未收敛；不返回半成品权重。"""


@dataclass(frozen=True)
class StyleResult:
    """风格分析结果。

    - ``weights``：风格权重（Series，非负，和为 1），按输入列顺序
    - ``r_squared``：1 − Var(ε) / Var(r_p)
    - ``residual_mean`` / ``residual_volatility``：残差 ε = r_p − Σ w_k r_k 的每期均值与每期样本标准差（n−1）
    - ``n``：实际使用的期数（任一序列缺失的期整行剔除，不补零）
    - ``objective``：``"variance"`` 或 ``"sse"``
    - ``diagnostics``：求解器信息（``converged``、``iterations``、``message``）、风格指数两两相关矩阵
      ``correlation``、相关系数高于阈值的组合 ``collinear_pairs``（[(列 A, 列 B, 相关系数), ...]）

    权重是统计估计，不等于实际持仓；残差均值不能直接视为扣除一切风险后的选股能力。
    """

    weights: pd.Series
    r_squared: float
    residual_mean: float
    residual_volatility: float
    n: int
    objective: str
    resid: pd.Series = field(repr=False)
    fitted: pd.Series = field(repr=False)
    diagnostics: dict[str, Any] = field(default_factory=dict, repr=False)

    def annualized_residual_mean(self, periods_per_year: int) -> float:
        """残差均值的算术年化 = 每期均值 × K（算术口径，不是几何年化）。"""
        return self.residual_mean * periods_per_year

    def annualized_residual_volatility(self, periods_per_year: int) -> float:
        """残差年化波动 = 每期样本标准差 × √K。"""
        return self.residual_volatility * math.sqrt(periods_per_year)

    @property
    def collinear_pairs(self) -> list[tuple[str, str, float]]:
        return list(self.diagnostics.get("collinear_pairs", []))

    def top(self, n: int = 2) -> pd.Series:
        """权重最大的前 n 项（只含正权重），按权重降序。"""
        w = self.weights[self.weights > _CONSTRAINT_TOL]
        return w.sort_values(ascending=False, kind="stable").head(n)

    def table(self) -> pd.DataFrame:
        """风格权重表：风格、权重。"""
        return pd.DataFrame({"风格": self.weights.index, "权重": self.weights.to_numpy()})


def _style_frame(returns, style_returns, risk_free) -> pd.DataFrame:
    """对齐组合与风格收益，给出 risk_free 时加入 cash 列；返回首列为组合、其余为风格的表，已剔除缺失行。"""
    r, x = _align(returns, style_returns)
    if x.shape[1] == 0:
        raise ValueError("至少需要一个风格指数")
    if risk_free is not None:
        if CASH in x.columns:
            raise ValueError(f"风格收益已含 {CASH!r} 列，不能再传入 risk_free")
        x = x.assign(**{CASH: broadcast_like(risk_free, r)})
    if "__portfolio__" in x.columns:
        raise ValueError("风格列名不能为 '__portfolio__'")
    data = pd.concat([r.rename("__portfolio__"), x], axis=1).dropna()
    return data


def _correlation_check(x: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, list[tuple[str, str, float]]]:
    """风格指数两两相关系数；常数列（如常数现金收益）相关系数为 NaN，不参与判断。"""
    corr = x.corr()
    pairs = []
    cols = list(x.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            c = corr.loc[a, b]
            if np.isfinite(c) and c > threshold:
                pairs.append((a, b, float(c)))
    return corr, pairs


def _collinearity_message(pairs) -> str:
    text = "；".join(f"{a} 与 {b} 相关系数 {c:.3f}" for a, b, c in pairs)
    return f"风格指数高度相关（{text}，高于 {COLLINEARITY_THRESHOLD}），权重解释不稳定，宜换用低冗余的风格基准"


def _solve(y: np.ndarray, x: np.ndarray, objective: str, maxiter: int) -> optimize.OptimizeResult:
    """二次规划：min f(w)，w ≥ 0，Σw = 1。目标函数按 Var(r_p) 或 Σ r_p² 归一，避免收益量级过小时提前停止。"""
    k = x.shape[1]
    if objective == "variance":
        xc = x - x.mean(axis=0)
        yc = y - y.mean()
    else:
        xc, yc = x, y
    scale = float(yc @ yc)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    q = xc.T @ xc / scale
    c = xc.T @ yc / scale
    const = float(yc @ yc) / scale

    def fun(w):
        return float(w @ q @ w - 2 * w @ c + const)

    def jac(w):
        return 2 * (q @ w - c)

    return optimize.minimize(
        fun,
        np.full(k, 1.0 / k),
        jac=jac,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0, "jac": lambda w: np.ones(k)}],
        options={"maxiter": maxiter, "ftol": 1e-12},
    )


def style_analysis(
    returns,
    style_returns,
    *,
    objective: str = "variance",
    risk_free=None,
    maxiter: int = 1000,
    warn: bool = True,
) -> StyleResult:
    """Sharpe 收益型风格分析：约束回归 r_p,t = Σ w_k r_k,t + ε_t，w_k ≥ 0，Σ w_k = 1。

    参数
    ----
    returns : 组合单期收益（Series 或数组）
    style_returns : 风格指数单期收益（DataFrame，每列一个风格，列名即风格名）；两者都带 pandas 索引时
        观测期须一致（先用 fundeval.etl.clean.align 对齐）
    objective : ``"variance"``（默认，最小化残差方差）或 ``"sse"``（最小化残差平方和）
    risk_free : 给出时把“现金”作为一个风格资产加入，列名为 ``cash``；常数或同期序列
    maxiter : SLSQP 最大迭代次数
    warn : 风格指数两两相关系数高于 0.95 时是否发出 RuntimeWarning（诊断中总会记录）

    用 scipy.optimize 的 SLSQP 求解该二次规划，未收敛时抛出 StyleNotConverged，不返回半成品。
    任一序列缺失的期整行剔除，``n`` 为实际使用的期数；期数须不少于风格资产数 + 2。

    风格权重是统计估计，不等于实际持仓；残差均值不能直接视为扣除一切风险后的选股能力。
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective 须为 {OBJECTIVES} 之一，收到 {objective!r}")
    data = _style_frame(returns, style_returns, risk_free)
    y = data["__portfolio__"]
    x = data.drop(columns="__portfolio__")
    k = x.shape[1]
    if len(data) < k + 2:
        raise ValueError(f"有效样本 {len(data)} 期，少于风格资产数 {k} + 2，无法估计风格权重")

    corr, pairs = _correlation_check(x, COLLINEARITY_THRESHOLD)
    if pairs and warn:
        warnings.warn(_collinearity_message(pairs), RuntimeWarning, stacklevel=2)

    res = _solve(y.to_numpy(), x.to_numpy(), objective, maxiter)
    w = np.asarray(res.x, dtype=float)
    if not res.success:
        raise StyleNotConverged(f"风格权重的约束优化未收敛（{res.message}，迭代 {res.nit} 次）")
    if w.min() < -_CONSTRAINT_TOL or abs(w.sum() - 1) > _CONSTRAINT_TOL or not np.isfinite(w).all():
        raise StyleNotConverged(f"风格权重不满足约束（最小值 {w.min():.2e}，加总 {w.sum():.8f}）")
    # 去掉浮点误差带来的微小负数，再归一
    w = np.clip(w, 0.0, None)
    w = w / w.sum()

    weights = pd.Series(w, index=x.columns, name="weight")
    fitted = (x @ weights).rename("fitted")
    resid = (y - fitted).rename("resid")
    var_p = float(y.var(ddof=1))
    var_e = float(resid.var(ddof=1))
    diagnostics = {
        "converged": True,
        "iterations": int(res.nit),
        "message": str(res.message),
        "objective_value": float(res.fun),
        "correlation": corr,
        "collinear_pairs": pairs,
    }
    if pairs:
        diagnostics["collinearity_warning"] = _collinearity_message(pairs)
    return StyleResult(
        weights=weights,
        r_squared=1 - safe_div(var_e, var_p),
        residual_mean=float(resid.mean()),
        residual_volatility=float(resid.std(ddof=1)),
        n=len(data),
        objective=objective,
        resid=resid,
        fitted=fitted,
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class RollingStyleResult:
    """滚动风格分析结果：``weights`` 为按窗口末期排列的权重表，``r_squared`` 为逐期 R²。"""

    weights: pd.DataFrame
    r_squared: pd.Series
    window: int
    objective: str
    collinear_pairs: list[tuple[str, str, float]] = field(default_factory=list)

    def table(self) -> pd.DataFrame:
        """权重与 R² 合并的表，索引为窗口末期。"""
        return self.weights.assign(R2=self.r_squared)


def rolling_style(returns, style_returns, window: int, **kwargs) -> RollingStyleResult:
    """滚动风格分析：每个长度为 ``window`` 的窗口估计一次，结果按窗口末期排列。

    ``kwargs`` 传给 style_analysis（objective、risk_free、maxiter）。窗口须为整数，不小于风格资产数
    （含 cash）+ 2，且不超过有效样本期数。共线性警告按全样本判断，只发一次。
    滚动权重的变化只提示进一步检查持仓、投资授权与估计窗口，不能据此断言经理主动切换了风格。
    """
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
        raise ValueError(f"window 须为整数，收到 {window!r}")
    window = int(window)
    kwargs.pop("warn", None)
    risk_free = kwargs.pop("risk_free", None)
    data = _style_frame(returns, style_returns, risk_free)
    y = data["__portfolio__"]
    x = data.drop(columns="__portfolio__")
    k = x.shape[1]
    if window < k + 2:
        raise ValueError(f"window = {window} 小于风格资产数 {k} + 2")
    if window > len(data):
        raise ValueError(f"window = {window} 超过有效样本 {len(data)} 期")
    _, pairs = _correlation_check(x, COLLINEARITY_THRESHOLD)
    if pairs:
        warnings.warn(_collinearity_message(pairs), RuntimeWarning, stacklevel=2)
    rows, r2, index = [], [], []
    for end in range(window, len(data) + 1):
        res = style_analysis(y.iloc[end - window : end], x.iloc[end - window : end], warn=False, **kwargs)
        rows.append(res.weights.to_numpy())
        r2.append(res.r_squared)
        index.append(data.index[end - 1])
    idx = pd.Index(index, name=data.index.name)
    return RollingStyleResult(
        weights=pd.DataFrame(rows, index=idx, columns=x.columns),
        r_squared=pd.Series(r2, index=idx, name="R2"),
        window=window,
        objective=kwargs.get("objective", "variance"),
        collinear_pairs=pairs,
    )
