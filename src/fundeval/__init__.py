"""fundeval：基金与投资组合绩效评估工具包。

模块按 docs/投资组合绩效评估.md 的章节组织：

- ``fundeval.config``：第一部分第 1 节，评价口径配置 EvaluationConfig 与 TOML 读取 load_config
- ``fundeval.etl``：第一部分，数据口径、读取（本地文件与 akshare）、清洗、单期收益计算、
  频率转换、复合基准与数据质量报告
- ``fundeval.returns``：第二部分，TWR、年化、MWR/XIRR 与超额收益
- ``fundeval.risk``：第三部分，波动、回撤、Sharpe、IR、Treynor 与 M²
- ``fundeval.alpha``：第四部分，因子 / CAPM 回归（含 HAC 标准误）、滚动 Alpha 与 IR、
  样本内外切分、主动管理基本定律
- ``fundeval.attribution``：第五部分，Sharpe 风格分析、Brinson 单期（BHB / BF）与多期 Cariño 归因、
  多因子分解（含 A 股指数代理因子）、Campisi 固收归因、Treynor–Mazuy 与 Henriksson–Merton 择时回归
- ``fundeval.costs``：第六部分，换手率、线性交易成本、近似净 Alpha、容量检查与规模敏感性
- ``fundeval.monitor``：第七部分，实现 TE、风险倍数、z 值与 Green/Yellow/Red 分区
- ``fundeval.tail``：第八部分，下行偏差、Sortino、Calmar、历史模拟 VaR 与 ES
- ``fundeval.report``：第九、十部分，一键评价报告、三段结论、Markdown / Excel 导出与多基金横向对比
- ``fundeval.cli``：命令行 ``fundeval report`` 与 ``fundeval compare``

约定：所有收益率以小数输入输出（0.02 表示 2%）；K 为一年期数，月度取 12。
"""

from fundeval._version import __version__
from fundeval import alpha, attribution, costs, etl, monitor, report, returns, risk, tail
from fundeval import config

__all__ = ["__version__", "config", "etl", "returns", "risk", "alpha", "attribution", "costs", "tail", "monitor", "report"]
