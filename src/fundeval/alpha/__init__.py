"""Alpha 来源与能力判断：正文第四部分。

- ``regression``：因子回归与 CAPM 回归，报告 Alpha、Beta、标准误、t 值、p 值、置信区间与 R²，
  可选 Newey–West（HAC）稳健标准误
- ``rolling``：滚动 Alpha / Beta、滚动 IR 与按时间顺序切分的样本内外检验
- ``robustness``：剔除异常期后重新估计 CAPM 与择时回归，检查结论是否依赖少数极端期（第 3 节）
- ``fundamental``：主动管理基本定律 IR ≈ TC × IC × √BR
"""

from fundeval.alpha import fundamental, regression, rolling
from fundeval.alpha.fundamental import expected_ir
from fundeval.alpha.regression import RegressionResult, capm_regression, factor_regression
from fundeval.alpha.rolling import rolling_information_ratio, rolling_regression, split_in_out_of_sample

# robustness 依赖 attribution.timing（后者又依赖 alpha.regression），放在最后导入
from fundeval.alpha import robustness  # noqa: E402
from fundeval.alpha.robustness import ExclusionSensitivity, exclusion_sensitivity  # noqa: E402

__all__ = [
    "regression",
    "rolling",
    "fundamental",
    "robustness",
    "ExclusionSensitivity",
    "exclusion_sensitivity",
    "RegressionResult",
    "factor_regression",
    "capm_regression",
    "rolling_regression",
    "rolling_information_ratio",
    "split_in_out_of_sample",
    "expected_ir",
]
