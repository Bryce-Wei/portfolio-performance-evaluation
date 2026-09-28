"""快速开始：用正文第九部分“从原始数据到绩效结果”的 12 个月演示数据生成完整评价报告，不联网。

运行::

    python examples/quickstart.py                  # 输出到当前目录下的 quickstart_output/
    python examples/quickstart.py --out docs/examples

输出：

- ``report.md``：Markdown 报告（口径、各维度指标、回归系数、数据质量、结论）；装了 matplotlib 时
  图表 PNG 存到 ``report_files/`` 并在“图表”一节用相对路径嵌入
- ``report.xlsx``：Excel 报告（需 openpyxl；装了 matplotlib 时另有“图表”sheet）

并在标准输出打印关键指标，可与正文第九部分核对：累计收益 10.2058%、Sharpe 1.1254、IR 2.2327、
最大回撤 3.00%、M² 8.4668%。

未安装 fundeval（如直接解压源码运行）时，脚本把仓库的 src/ 加入搜索路径。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data" / "worked_example.csv"

try:
    import fundeval  # noqa: F401
except ImportError:  # 未安装时直接使用仓库源码
    sys.path.insert(0, str(ROOT / "src"))

from fundeval.etl.sources import files  # noqa: E402
from fundeval.report import evaluate, to_excel, to_markdown  # noqa: E402

#: 演示数据的列名（正文表格的中文列名）→ 标准列名
COLUMNS = {"月份": "date", "组合收益": "portfolio", "基准收益": "benchmark", "无风险收益": "risk_free"}

#: 打印的关键指标：(指标键, 显示格式)
KEY_METRICS = (
    ("cumulative_return", "pct"),
    ("benchmark_cumulative_return", "pct"),
    ("volatility", "pct"),
    ("max_drawdown", "pct"),
    ("sharpe", "ratio"),
    ("tracking_error", "pct"),
    ("information_ratio", "ratio"),
    ("m_squared", "pct"),
    ("alpha_annualized", "pct"),
    ("alpha_t", "ratio"),
)


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def build_report():
    """读取演示数据并生成评价报告（K = 12，无风险收益取表中每月 0.15%）。"""
    data = files.load_returns(DATA, columns=COLUMNS)
    return evaluate(
        data,
        periods_per_year=12,
        labels={
            "title": "正文第九部分演示组合评价报告",
            "portfolio": "演示组合（正文第九部分）",
            "benchmark": "演示基准（正文第九部分）",
            "risk_free": "每月 0.15%（正文第九部分）",
        },
    )


def key_metrics(report) -> list[tuple[str, str]]:
    """关键指标的 (名称, 显示值)。"""
    rows = []
    for key, kind in KEY_METRICS:
        row = report.metrics.loc[key]
        value = float(row["value"])
        rows.append((str(row["label"]), f"{value:.4%}" if kind == "pct" else f"{value:.4f}"))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="用正文第九部分演示数据生成评价报告（不联网）")
    parser.add_argument("--out", default="quickstart_output", help="输出目录（默认 quickstart_output）")
    parser.add_argument("--no-charts", action="store_true", help="不生成图表（默认装了 matplotlib 时生成）")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    charts = not args.no_charts and has_module("matplotlib")
    report = build_report()

    to_markdown(report, out / "report.md", charts=charts)
    written = [out / "report.md"]
    if has_module("openpyxl"):
        to_excel(report, out / "report.xlsx", charts=charts)
        written.append(out / "report.xlsx")
    else:
        print('未安装 openpyxl，跳过 Excel 报告（pip install "fundeval[excel]"）', file=sys.stderr)
    if not charts and not args.no_charts:
        print('未安装 matplotlib，报告不含图表（pip install "fundeval[plot]"）', file=sys.stderr)

    print("关键指标（正文第九部分演示数据，月度，K = 12）：")
    rows = key_metrics(report)
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"  {name:<{width}}  {value}")
    print("已生成：" + "、".join(str(p) for p in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
