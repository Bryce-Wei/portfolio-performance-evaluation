"""多因子分解：正文第五部分第 3 节。

    r_p,t − r_f,t = α + Σ β_k F_k,t + ε_t

回归沿用 alpha.regression.factor_regression（OLS，可选 Newey–West HAC 与 t 分布）。在系数之外，
按样本均值把组合的平均超额收益拆成各因子的贡献、Alpha 与残差：

    mean(r_p − r_f) = α + Σ β_k × mean(F_k) + mean(ε)

其中 β_k × mean(F_k) 为因子 k 的收益贡献（每期；乘以 K 为算术年化）。OLS 含截距时 mean(ε) 在数值
误差内为零，保留这一项是为了逐项对账：三部分之和与平均超额收益不一致时报错。

A 股指数代理因子（``factor_preset("cn_index_proxy")``）用中证指数官网的全收益指数构造：

- MKT = 沪深300全收益 H00300 − 无风险收益
- SMB = 中证1000全收益 H00852 − 沪深300全收益 H00300
- HML = 沪深300价值全收益 H00919 − 沪深300成长全收益 H00918

这是“指数代理因子”：由只做多的指数收益相减得到，与 Fama–French 按市值、账面市值比分组、
多空组合的构造不同（成分、加权、再平衡与行业结构都不同），系数不能与学术因子的结果直接比较。
没有经过核实的动量指数，默认不构造 UMD；调用方可以在 ``extra`` 中传入自定义因子列。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fundeval._utils import broadcast_like
from fundeval.alpha.regression import ALPHA, RegressionResult, excess_frame, factor_regression

#: 大样本双侧 5% 的 t 值参考阈值
T_THRESHOLD = 1.96

#: 对账容差：因子贡献、Alpha 与残差之和与平均超额收益之差（每期，小数）
RECONCILE_TOL = 1e-10

#: 无风险收益在预设中的占位名
RISK_FREE = "rf"

#: 指数代理因子的限定语（报告口径与结论使用）
PROXY_CAVEAT = (
    "因子为指数代理因子，由只做多的指数收益相减得到，与 Fama–French 的分组构造不同，"
    "系数不能与学术因子结果直接比较"
)

#: 因子预设：{因子名: (多头指数代码, 空头代码)}，空头为 ``rf`` 时减无风险收益。指数一律为全收益代码。
FACTOR_PRESETS: dict[str, dict[str, tuple[str, str]]] = {
    "cn_index_proxy": {
        "MKT": ("H00300", RISK_FREE),
        "SMB": ("H00852", "H00300"),
        "HML": ("H00919", "H00918"),
    },
}

#: 预设的说明文字
FACTOR_DESCRIPTIONS = {
    "cn_index_proxy": {
        "MKT": "沪深300全收益 H00300 − 无风险收益",
        "SMB": "中证1000全收益 H00852 − 沪深300全收益 H00300",
        "HML": "沪深300价值全收益 H00919 − 沪深300成长全收益 H00918",
    },
}

#: 常见因子显著时的方向解读：{因子名: (为正, 为负)}
EXPOSURE_HINTS = {
    "MKT": ("市场暴露", "市场暴露为负"),
    "SMB": ("小盘暴露", "大盘暴露"),
    "HML": ("价值暴露", "成长暴露"),
    "UMD": ("动量暴露", "反转暴露"),
}

#: DataFrame.attrs 中标记因子类型的键；值为 "index_proxy" 时报告写明代理因子的限定语
FACTOR_TYPE_ATTR = "factor_type"
INDEX_PROXY = "index_proxy"


def factor_preset(name: str) -> dict[str, tuple[str, str]]:
    """因子预设，返回 {因子名: (多头指数代码, 空头代码)} 的副本；空头为 ``rf`` 时减无风险收益。

    - ``cn_index_proxy``：MKT = H00300 − rf，SMB = H00852 − H00300，HML = H00919 − H00918
      （中证指数官网全收益指数，2026-09-27 实测可取）。不含 UMD（没有经过核实的动量指数）。
    """
    if name not in FACTOR_PRESETS:
        raise KeyError(f"未知的因子预设 {name!r}，可选：{sorted(FACTOR_PRESETS)}")
    return dict(FACTOR_PRESETS[name])


def preset_codes(name: str) -> list[str]:
    """预设用到的指数代码（去重、保持顺序，不含 rf）。"""
    codes: list[str] = []
    for long, short in factor_preset(name).values():
        for code in (long, short):
            if code != RISK_FREE and code not in codes:
                codes.append(code)
    return codes


def index_proxy_factors(
    index_returns: Mapping[str, pd.Series] | pd.DataFrame,
    risk_free=0.0,
    preset: str = "cn_index_proxy",
    extra: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """用指数收益构造指数代理因子（每期，小数）。

    ``index_returns`` 为 {指数代码: 单期收益} 或以代码为列的 DataFrame，须包含预设用到的全部代码且
    日期一致；``risk_free`` 为常数或同期序列；``extra`` 为调用方自定义的因子列（如经核实的动量因子），
    按日期并入，须覆盖全部日期。任一期缺失时报错，不填零。

    返回的 DataFrame 在 ``attrs["factor_type"]`` 中标为 ``"index_proxy"``，evaluate 据此在口径与结论中
    写明代理因子的限定语。
    """
    spec = factor_preset(preset)
    frame = index_returns if isinstance(index_returns, pd.DataFrame) else pd.concat(dict(index_returns), axis=1)
    frame = frame.astype(float)
    missing = [c for c in preset_codes(preset) if c not in frame.columns]
    if missing:
        raise ValueError(f"构造 {preset} 因子缺少指数：{'、'.join(missing)}")
    gaps = frame[preset_codes(preset)].isna().any(axis=1)
    if gaps.any():
        raise ValueError(
            f"指数收益有 {int(gaps.sum())} 期缺失（首个 {frame.index[gaps.to_numpy()][0]}），请先对齐日期，不能填零"
        )
    rf = broadcast_like(risk_free, frame.iloc[:, 0])
    out = pd.DataFrame(index=frame.index)
    for name, (long, short) in spec.items():
        out[name] = frame[long] - (rf if short == RISK_FREE else frame[short])
    if extra is not None:
        ex = pd.DataFrame(extra).astype(float)
        dup = [c for c in ex.columns if c in out.columns]
        if dup:
            raise ValueError(f"自定义因子与预设因子重名：{'、'.join(map(str, dup))}")
        ex = ex.reindex(out.index)
        if ex.isna().any().any():
            raise ValueError("自定义因子未覆盖全部日期，不能填零")
        out = pd.concat([out, ex], axis=1)
    out.attrs[FACTOR_TYPE_ATTR] = INDEX_PROXY
    out.attrs["preset"] = preset
    return out


@dataclass(frozen=True)
class FactorDecomposition:
    """多因子分解结果。

    - ``regression``：factor_regression 的回归结果（系数、标准误、t、p、置信区间、R²、n）
    - ``excess_mean``：参与回归各期的组合平均超额收益（每期）
    - ``factor_means``：同一样本期的因子均值（每期）
    - ``contributions``：各因子的收益贡献 β_k × mean(F_k)（每期）
    - ``residual_mean``：残差均值（每期，OLS 含截距时约为零）

    每期口径满足 excess_mean = alpha + Σ contributions + residual_mean；算术年化各项乘以 K，
    由 ``annualized_*`` 与 ``table(periods_per_year)`` 给出。
    """

    regression: RegressionResult
    excess_mean: float
    factor_means: pd.Series
    contributions: pd.Series
    residual_mean: float
    factor_type: str | None = None

    @property
    def alpha(self) -> float:
        """每期多因子 Alpha（截距）。"""
        return self.regression.alpha

    @property
    def alpha_t(self) -> float:
        return self.regression.alpha_t

    @property
    def alpha_p(self) -> float:
        return self.regression.alpha_p

    @property
    def betas(self) -> pd.Series:
        return self.regression.betas

    @property
    def n(self) -> int:
        return self.regression.n

    @property
    def rsquared(self) -> float:
        return self.regression.rsquared

    @property
    def index_proxy(self) -> bool:
        return self.factor_type == INDEX_PROXY

    def annualized_alpha(self, periods_per_year: int) -> float:
        """算术年化多因子 Alpha = α × K（不是几何年化）。"""
        return self.regression.annualized_alpha(periods_per_year)

    def annualized_contributions(self, periods_per_year: int) -> pd.Series:
        """各因子收益贡献的算术年化值 = β_k × mean(F_k) × K。"""
        return self.contributions * periods_per_year

    def reconciliation(self, periods_per_year: int | None = None) -> pd.Series:
        """对账表：各因子贡献、Alpha、残差、合计与组合平均超额收益。

        ``periods_per_year`` 为 None 时为每期值，给出时为算术年化值（各项乘以 K）。
        """
        k = 1 if periods_per_year is None else periods_per_year
        items = {f"{name} 贡献": float(c) * k for name, c in self.contributions.items()}
        items["Alpha"] = self.alpha * k
        items["残差"] = self.residual_mean * k
        items["合计"] = sum(items.values())
        items["平均超额收益"] = self.excess_mean * k
        return pd.Series(items, name="每期" if periods_per_year is None else "算术年化")

    def table(self, periods_per_year: int | None = None) -> pd.DataFrame:
        """因子暴露表：因子、系数、t、p、因子均值（每期）、贡献（每期），给出 K 时另附贡献（算术年化）。"""
        reg = self.regression
        rows = []
        for name in self.contributions.index:
            row = {
                "因子": name,
                "系数": float(reg.params[name]),
                "t": float(reg.tvalues[name]),
                "p": float(reg.pvalues[name]),
                "因子均值（每期）": float(self.factor_means[name]),
                "贡献（每期）": float(self.contributions[name]),
            }
            if periods_per_year is not None:
                row["贡献（算术年化）"] = float(self.contributions[name]) * periods_per_year
            rows.append(row)
        return pd.DataFrame(rows)

    def significant_exposures(self, threshold: float = T_THRESHOLD) -> list[tuple[str, float, float]]:
        """|t| ≥ threshold 的因子：[(因子名, 系数, t), ...]，按 |t| 从大到小。"""
        out = [
            (name, float(self.regression.params[name]), float(self.regression.tvalues[name]))
            for name in self.contributions.index
            if np.isfinite(self.regression.tvalues[name]) and abs(self.regression.tvalues[name]) >= threshold
        ]
        return sorted(out, key=lambda x: -abs(x[2]))

    def exposure_text(self, threshold: float = T_THRESHOLD) -> str:
        """显著暴露的文字描述，如“HML 显著为正（t = 3.10），提示价值暴露”；没有显著暴露时说明。"""
        sig = self.significant_exposures(threshold)
        if not sig:
            return f"各因子系数均未显著偏离零（|t| < {threshold}）"
        parts = []
        for name, coef, t in sig:
            direction = "为正" if coef > 0 else "为负"
            hint = EXPOSURE_HINTS.get(name)
            text = f"{name} 显著{direction}（β = {coef:.3f}，t = {t:.2f}）"
            if hint is not None:
                text += f"，提示{hint[0] if coef > 0 else hint[1]}"
            parts.append(text)
        return "；".join(parts)


def factor_decomposition(
    returns, factors, risk_free=0.0, hac_lags: int | None = None, use_t: bool | None = None
) -> FactorDecomposition:
    """多因子回归与收益分解（正文第五部分第 3 节）。

    ``returns`` 为组合单期收益，``factors`` 为因子收益 DataFrame（每列一个因子；市场因子应已是超额收益，
    多空或相减得到的因子不再减无风险收益），``risk_free`` 为常数或同期序列。回归由
    alpha.regression.factor_regression 完成，``hac_lags`` 与 ``use_t`` 含义相同。

    收益贡献按参与回归的样本期计算：β_k × mean(F_k)（每期），算术年化用结果的
    ``annualized_contributions(K)``、``annualized_alpha(K)``。各因子贡献、Alpha 与残差均值之和须与组合
    平均超额收益一致（容差 1e-10），否则报错。

    ``factors`` 的 ``attrs["factor_type"]`` 为 ``"index_proxy"``（index_proxy_factors 的输出）时，结果标为
    指数代理因子，报告据此写明：因子由只做多的指数收益相减得到，与 Fama–French 的分组构造不同，
    系数不能与学术因子结果直接比较（PROXY_CAVEAT）。
    """
    factor_type = getattr(factors, "attrs", {}).get(FACTOR_TYPE_ATTR) if isinstance(factors, pd.DataFrame) else None
    reg = factor_regression(returns, factors, risk_free, hac_lags=hac_lags, use_t=use_t)
    y, f = excess_frame(returns, factors, risk_free)
    used = reg.resid.index
    y_used = y.reindex(used)
    f_used = f.reindex(used)
    excess_mean = float(y_used.mean())
    factor_means = f_used.mean().rename("mean")
    betas = reg.params.drop(ALPHA)
    contributions = (betas * factor_means.reindex(betas.index)).rename("contribution")
    residual_mean = float(reg.resid.mean())
    total = reg.alpha + float(contributions.sum()) + residual_mean
    if not np.isfinite(total) or abs(total - excess_mean) > RECONCILE_TOL:
        raise ValueError(
            f"多因子分解无法对账：Alpha + 因子贡献 + 残差 = {total:.10f}，平均超额收益 = {excess_mean:.10f}"
        )
    return FactorDecomposition(
        regression=reg,
        excess_mean=excess_mean,
        factor_means=factor_means,
        contributions=contributions,
        residual_mean=residual_mean,
        factor_type=factor_type,
    )
