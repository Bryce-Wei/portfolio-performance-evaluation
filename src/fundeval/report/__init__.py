"""评价报告：正文第九部分（从原始数据到绩效结果）与第十部分（形成结论）。

- ``summary``：evaluate 生成按六个评价维度组织的 EvaluationReport，conclusion 写出三段结论
- ``export``：to_markdown 与 to_excel
"""

from fundeval.report import export, summary
from fundeval.report.export import to_excel, to_markdown
from fundeval.report.summary import EvaluationReport, conclusion, evaluate

__all__ = ["summary", "export", "EvaluationReport", "evaluate", "conclusion", "to_markdown", "to_excel"]
