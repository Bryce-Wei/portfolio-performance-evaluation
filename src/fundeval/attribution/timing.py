"""择时模型检验市场敞口是否随行情变化：正文第五部分第 5 节。

x 为市场超额收益，y 为组合超额收益：

    TM：y_t = α + β x_t + γ x_t² + ε_t
    HM：y_t = α + β x_t + γ max(x_t, 0) + ε_t

TM 中正且显著的 γ 表示正向凸性；HM 中下行 Beta 为 β，上行 Beta 为 β + γ。
γ 为正且显著可支持择时特征，但期权、动态风险控制及其他非线性策略同样可能产生
这一结果；反过来，仅凭线性模型的恒定 Beta 也不能证明不存在择时能力。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fundeval._utils import broadcast_like
from fundeval.alpha.regression import RegressionResult, _align, fit_ols


@dataclass(frozen=True)
class TimingResult(RegressionResult):
    """择时回归结果：在 RegressionResult 基础上增加 ``model``（"TM" 或 "HM"）。

    系数名为 ``alpha``、``beta``、``gamma``；HM 另提供 ``downside_beta`` 与 ``upside_beta``。
    """

    model: str = "TM"

    @property
    def beta(self) -> float:
        return float(self.params["beta"])

    @property
    def gamma(self) -> float:
        return float(self.params["gamma"])

    @property
    def gamma_t(self) -> float:
        return float(self.tvalues["gamma"])

    @property
    def gamma_p(self) -> float:
        return float(self.pvalues["gamma"])

    def _require_hm(self) -> None:
        if self.model != "HM":
            raise AttributeError("上行 / 下行 Beta 只对 Henriksson–Merton 模型定义")

    @property
    def downside_beta(self) -> float:
        """HM 下行 Beta = β。"""
        self._require_hm()
        return self.beta

    @property
    def upside_beta(self) -> float:
        """HM 上行 Beta = β + γ。"""
        self._require_hm()
        return self.beta + self.gamma


def _timing(returns, market, risk_free, hac_lags, model: str) -> TimingResult:
    r, m = _align(returns, market)
    rf = broadcast_like(risk_free, r)
    y = (r - rf).rename("excess")
    x = m.iloc[:, 0] - rf
    extra = x**2 if model == "TM" else np.maximum(x, 0.0)
    design = pd.DataFrame({"beta": x, "gamma": extra})
    res = fit_ols(y, design, hac_lags)
    return TimingResult(**res.__dict__, model=model)


def treynor_mazuy(returns, market, risk_free=0.0, hac_lags: int | None = None) -> TimingResult:
    """Treynor–Mazuy 择时回归 y = α + βx + γx² + ε。

    ``market`` 为市场原始收益，函数内部与组合收益一并减去 ``risk_free``。
    ``hac_lags`` 的含义与 alpha.regression.factor_regression 相同。
    γ 为正且显著可支持择时特征，但期权、动态风险控制等非线性策略同样可能产生
    这一结果，不能单凭 γ 断言经理具备择时能力。
    """
    return _timing(returns, market, risk_free, hac_lags, "TM")


def henriksson_merton(returns, market, risk_free=0.0, hac_lags: int | None = None) -> TimingResult:
    """Henriksson–Merton 择时回归 y = α + βx + γ·max(x, 0) + ε。

    结果给出下行 Beta ``downside_beta`` = β 与上行 Beta ``upside_beta`` = β + γ。
    γ 为正且显著可支持择时特征，但期权、动态风险控制等非线性策略同样可能产生
    这一结果，不能单凭 γ 断言经理具备择时能力。
    """
    return _timing(returns, market, risk_free, hac_lags, "HM")
