"""fundeval：基金与投资组合绩效评估工具包。

模块按 docs/投资组合绩效评估.md 的章节组织：

- ``fundeval.etl``：第一部分，数据口径、读取、清洗与单期收益计算
- ``fundeval.returns``：第二部分，TWR、年化、MWR/XIRR 与超额收益
- ``fundeval.risk``：第三部分，波动、回撤、Sharpe、IR、Treynor 与 M²
- ``fundeval.monitor``：第七部分，实现 TE、风险倍数、z 值与 Green/Yellow/Red 分区
- ``fundeval.tail``：第八部分，下行偏差、Sortino、Calmar、历史模拟 VaR 与 ES

约定：所有收益率以小数输入输出（0.02 表示 2%）；K 为一年期数，月度取 12。
"""

from fundeval import etl, monitor, returns, risk, tail

__all__ = ["etl", "returns", "risk", "tail", "monitor"]
__version__ = "0.1.0"
