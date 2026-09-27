"""fundeval：基金与投资组合绩效评估工具包。

模块按 docs/投资组合绩效评估.md 的章节组织：

- ``fundeval.etl``：第一部分，数据口径、读取、清洗与单期收益计算
- ``fundeval.returns``：第二部分，TWR、年化、MWR/XIRR 与超额收益
- ``fundeval.risk``：第三部分，波动、回撤、Sharpe、IR、Treynor 与 M²

约定：所有收益率以小数输入输出（0.02 表示 2%）；K 为一年期数，月度取 12。
"""

from fundeval import etl, returns, risk

__all__ = ["etl", "returns", "risk"]
__version__ = "0.1.0"
