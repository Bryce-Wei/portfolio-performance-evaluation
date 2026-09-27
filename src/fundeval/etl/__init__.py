"""ETL：对应正文第一部分“评价框架与数据准备”。

清洗顺序：日期对齐、币种统一、现金流识别、缺失与异常复核、收益计算。

- ``schema``：标准列名、频率与校验
- ``sources.files``：从 CSV/Excel 读取为标准结构
- ``clean``：对齐、去重、币种换算、缺失与异常复核
- ``returns``：净值、价格加分红、账户资产加现金流到单期收益
"""

from fundeval.etl import clean, returns, schema
from fundeval.etl.sources import files

__all__ = ["clean", "files", "returns", "schema"]
