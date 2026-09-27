"""ETL：对应正文第一部分“评价框架与数据准备”。

清洗顺序：日期对齐、币种统一、现金流识别、缺失与异常复核、收益计算。

- ``schema``：标准列名、频率与校验
- ``sources.files``：从 CSV/Excel 读取为标准结构
- ``sources.akshare``：国内公募基金净值与分红、指数行情、无风险利率（可选依赖 akshare，按需导入）
- ``clean``：对齐、去重、币种换算、缺失与异常复核
- ``returns``：净值、价格加分红、账户资产加现金流到单期收益；日度收益合成为周、月、季收益
- ``benchmark``：复合基准合成与合同基准文字解析
- ``quality``：数据质量报告与净值 / 日增长率交叉核对
"""

from fundeval.etl import benchmark, clean, quality, returns, schema
from fundeval.etl.sources import files

__all__ = ["benchmark", "clean", "files", "quality", "returns", "schema"]
