"""Alpha 来源与能力判断：正文第四部分。

- ``regression``：因子回归与 CAPM 回归，报告 Alpha、Beta、标准误、t 值、p 值、置信区间与 R²，
  可选 Newey–West（HAC）稳健标准误
- ``rolling``：滚动 Alpha / Beta、滚动 IR 与按时间顺序切分的样本内外检验
- ``fundamental``：主动管理基本定律 IR ≈ TC × IC × √BR
"""

from fundeval.alpha import fundamental, regression, rolling
from fundeval.alpha.fundamental import expected_ir
from fundeval.alpha.regression import RegressionResult, capm_regression, factor_regression
from fundeval.alpha.rolling import rolling_information_ratio, rolling_regression, split_in_out_of_sample

__all__ = [
    "regression",
    "rolling",
    "fundamental",
    "RegressionResult",
    "factor_regression",
    "capm_regression",
    "rolling_regression",
    "rolling_information_ratio",
    "split_in_out_of_sample",
    "expected_ir",
]
