"""稳健性检验：剔除异常期后重新估计，正文第四部分第 3 节“稳健性”。

正文要求检查“结论是否依赖单一设定”。少数极端期（如单月大涨大跌）可能主导回归系数，
尤其是择时回归中的 x² 与 max(x, 0) 项。本模块在全样本之外，剔除指定的期（通常为数据质量
报告标为异常收益的期）重新估计 CAPM、Treynor–Mazuy 与 Henriksson–Merton 回归，比较关键系数：

- CAPM：Alpha（截距）
- TM、HM：择时 γ

剔除后关键系数变号，或 |t| 跨过 1.96（由显著变为不显著或反之），即视为“结论对异常期敏感”。
剔除只用于敏感性分析，不修改数据；异常期是否为数据错误须另行复核。剔除后序列不再连续，
HAC 标准误按剩余观测的先后顺序计算，只作近似。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fundeval.alpha.regression import RegressionResult, capm_regression
from fundeval.attribution.timing import henriksson_merton, treynor_mazuy

#: 大样本双侧 5% 的 t 值参考阈值
T_THRESHOLD = 1.96

#: 各模型的关键系数与显示名
KEY_COEFFICIENTS = {"CAPM": ("alpha", "CAPM Alpha"), "TM": ("gamma", "TM γ"), "HM": ("gamma", "HM γ")}

_MODELS = {"CAPM": capm_regression, "TM": treynor_mazuy, "HM": henriksson_merton}


def _changed(coef_a: float, t_a: float, coef_b: float, t_b: float) -> bool:
    if not all(np.isfinite(v) for v in (coef_a, t_a, coef_b, t_b)):
        return False
    return bool(np.sign(coef_a) != np.sign(coef_b) or (abs(t_a) >= T_THRESHOLD) != (abs(t_b) >= T_THRESHOLD))


def _fmt_dates(dates: pd.DatetimeIndex) -> str:
    return "、".join(f"{d:%Y-%m-%d}" for d in dates)


@dataclass(frozen=True)
class ExclusionSensitivity:
    """剔除异常期的敏感性分析结果。

    - ``excluded``：实际剔除的期（按日期排序）；为空时未做剔除
    - ``full`` / ``trimmed``：全样本与剔除后的回归结果（{"CAPM", "TM", "HM"} → 结果）
    - ``error``：剔除后样本不足等原因导致无法重新估计时的说明
    """

    excluded: pd.DatetimeIndex
    full: dict[str, RegressionResult]
    trimmed: dict[str, RegressionResult] = field(default_factory=dict)
    error: str | None = None

    @property
    def done(self) -> bool:
        return len(self.excluded) > 0 and self.error is None

    def key_table(self) -> pd.DataFrame:
        """关键系数对比：模型、系数、全样本估计与 t、剔除后估计与 t、是否改变显著性。"""
        rows = []
        for model, (coef, label) in KEY_COEFFICIENTS.items():
            full = self.full.get(model)
            if full is None:
                continue
            trimmed = self.trimmed.get(model)
            a, ta = float(full.params[coef]), float(full.tvalues[coef])
            b = float(trimmed.params[coef]) if trimmed is not None else float("nan")
            tb = float(trimmed.tvalues[coef]) if trimmed is not None else float("nan")
            rows.append(
                {
                    "模型": model,
                    "系数": label,
                    "全样本估计": a,
                    "全样本 t": ta,
                    "全样本 n": full.n,
                    "剔除后估计": b,
                    "剔除后 t": tb,
                    "剔除后 n": trimmed.n if trimmed is not None else np.nan,
                    "显著性或符号改变": "是" if _changed(a, ta, b, tb) else "否",
                }
            )
        return pd.DataFrame(rows)

    def table(self) -> pd.DataFrame:
        """全部系数的对比表：模型、系数、全样本与剔除后的估计值与 t 值。"""
        rows = []
        for model, full in self.full.items():
            trimmed = self.trimmed.get(model)
            for coef in full.params.index:
                rows.append(
                    {
                        "模型": model,
                        "系数": coef,
                        "全样本估计": float(full.params[coef]),
                        "全样本 t": float(full.tvalues[coef]),
                        "剔除后估计": float(trimmed.params[coef]) if trimmed is not None else np.nan,
                        "剔除后 t": float(trimmed.tvalues[coef]) if trimmed is not None else np.nan,
                    }
                )
        return pd.DataFrame(rows)

    @property
    def changed(self) -> list[str]:
        """关键系数变号或 |t| 跨过 1.96 的模型。"""
        if not self.done:
            return []
        tbl = self.key_table()
        return list(tbl.loc[tbl["显著性或符号改变"] == "是", "模型"])

    @property
    def sensitive(self) -> bool:
        return bool(self.changed)

    def summary(self) -> str:
        """一句话结论，供报告使用。"""
        n = len(self.excluded)
        if n == 0:
            return "无异常期，未做剔除。"
        dates = _fmt_dates(self.excluded)
        if self.error is not None:
            return f"拟剔除 {n} 个异常期（{dates}），但剔除后无法重新估计：{self.error}。"
        tbl = self.key_table()

        def t_text(row) -> str:
            return (
                f"{row['系数']} t 值 {row['全样本 t']:.2f} → {row['剔除后 t']:.2f}"
                f"（系数 {row['全样本估计']:.4f} → {row['剔除后估计']:.4f}）"
            )

        if self.sensitive:
            changed = tbl[tbl["显著性或符号改变"] == "是"]
            return (
                f"结论对 {n} 个异常期敏感（剔除 {dates}）："
                + "；".join(t_text(row) for _, row in changed.iterrows())
                + "。相关结论依赖少数极端期，须复核这些期的数据与成因。"
            )
        return (
            f"剔除 {n} 个异常期（{dates}）后，关键系数的符号与显著性不变："
            + "；".join(t_text(row) for _, row in tbl.iterrows())
            + "。"
        )


def exclusion_sensitivity(
    returns,
    market,
    risk_free=0.0,
    exclude: Iterable = (),
    *,
    hac_lags: int | None = None,
    use_t: bool | None = None,
    full: dict[str, RegressionResult] | None = None,
) -> ExclusionSensitivity:
    """剔除 ``exclude`` 中的期后重新估计 CAPM、TM、HM 回归，与全样本比较。

    ``returns``、``market`` 为带 DatetimeIndex 的 Series（观测期一致），``risk_free`` 为常数或同期
    序列；``exclude`` 为要剔除的日期，须在样本内，否则报错。``full`` 可传入已算好的全样本结果，
    避免重复估计。``hac_lags`` 与 ``use_t`` 同 alpha.regression.capm_regression。
    """
    if not isinstance(returns, pd.Series) or not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns 须为以 DatetimeIndex 为索引的 Series")
    dates = pd.DatetimeIndex(sorted({pd.Timestamp(d) for d in exclude}))
    outside = dates.difference(returns.index)
    if len(outside):
        raise ValueError(f"要剔除的日期不在样本内：{_fmt_dates(outside)}")
    rf = risk_free if np.isscalar(risk_free) else pd.Series(risk_free, copy=False).reindex(returns.index)
    kw = {"hac_lags": hac_lags, "use_t": use_t}
    if full is None:
        full = {name: fn(returns, market, rf, **kw) for name, fn in _MODELS.items()}
    models = {name: _MODELS[name] for name in full}
    if len(dates) == 0:
        return ExclusionSensitivity(excluded=dates, full=full)
    keep = ~returns.index.isin(dates)
    r = returns[keep]
    m = market.reindex(r.index)
    rf_kept = rf if np.isscalar(rf) else rf[keep]
    try:
        trimmed = {name: fn(r, m, rf_kept, **kw) for name, fn in models.items()}
    except ValueError as exc:
        return ExclusionSensitivity(excluded=dates, full=full, error=str(exc))
    return ExclusionSensitivity(excluded=dates, full=full, trimmed=trimmed)
