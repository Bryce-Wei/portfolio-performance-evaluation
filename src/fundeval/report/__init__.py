"""评价报告：正文第九部分（从原始数据到绩效结果）与第十部分（形成结论）。

- ``summary``：evaluate 生成按六个评价维度组织的 EvaluationReport，conclusion 写出三段结论
- ``export``：to_markdown 与 to_excel
- ``inputs``：按基金代码或本地文件组装 evaluate 的参数（含合同基准解析）
- ``compare``：多基金横向对比，``fundeval.report.compare.compare`` 返回 ComparisonReport
"""

from fundeval.report import export, summary
from fundeval.report.export import to_excel, to_markdown
from fundeval.report.summary import EvaluationReport, conclusion, evaluate
from fundeval.report.compare import ComparisonReport

__all__ = ["summary", "export", "EvaluationReport", "evaluate", "conclusion", "to_markdown", "to_excel", "ComparisonReport"]
