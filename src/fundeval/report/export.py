"""报告导出：Markdown 与 Excel（正文第九、十部分）。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fundeval.report.summary import SECTIONS, EvaluationReport, conclusion, format_value


def _md_escape(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "（无）"
    header = "| " + " | ".join(_md_escape(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(_md_escape(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def _regression_display(report: EvaluationReport) -> pd.DataFrame:
    tbl = report.regression_table()
    if tbl.empty:
        return tbl
    out = tbl.copy()
    for col in ("估计值", "标准误", "95% 下限", "95% 上限"):
        out[col] = out[col].map(lambda v: f"{v:.6f}")
    for col in ("t", "R²"):
        out[col] = out[col].map(lambda v: f"{v:.4f}")
    out["p"] = out["p"].map(lambda v: f"{v:.4f}")
    return out


def to_markdown(report: EvaluationReport, path: str | Path | None = None) -> str:
    """生成 Markdown 报告：标题、口径、各维度指标表、回归系数表、数据质量、结论与未完成检验。

    ``path`` 给出时同时写入文件（UTF-8）。返回 Markdown 文本。
    回归系数为每期值（与输入收益同频），未换算为百分数。
    """
    parts = [f"# {report.title}", "", "## 口径", "", _md_table(pd.DataFrame(list(report.scope.items()), columns=["项目", "内容"]))]
    for name in SECTIONS:
        section = report.section(name)
        if section.empty:
            continue
        parts += ["", f"## {name}", "", _md_table(section)]
    reg = _regression_display(report)
    if not reg.empty:
        parts += ["", "## 回归系数", "", "系数为每期值，与输入收益同频。", "", _md_table(reg)]
    parts += ["", "## 数据质量", "", _md_table(report.quality.summary())]
    issues = report.quality.issues()
    if issues:
        parts += ["", *[f"- {i}" for i in issues]]
    if report.notes:
        parts += ["", "## 附注", "", *[f"- {n}" for n in report.notes]]
    parts += ["", "## 结论", "", conclusion(report)]
    parts += ["", "## 未完成的检验", "", _md_table(report.not_done), ""]
    text = "\n".join(parts)
    if path is not None:
        Path(path).write_text(text, encoding="utf-8")
    return text


def to_excel(report: EvaluationReport, path: str | Path) -> Path:
    """写入 Excel（openpyxl），分 sheet：口径、指标、回归、期间收益、数据质量。

    指标 sheet 保存原始数值（小数）与单位，另附格式化后的显示值；期间收益 sheet 为对齐后的
    单期收益、主动收益与财富指数。需要安装 ``fundeval[excel]``。
    """
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise ImportError('导出 Excel 需要 openpyxl：pip install "fundeval[excel]"') from exc
    path = Path(path)
    scope = pd.DataFrame(list(report.scope.items()), columns=["项目", "内容"])
    metrics = report.metrics.reset_index().rename(
        columns={"section": "维度", "key": "键", "label": "指标", "value": "数值", "unit": "单位", "note": "口径说明"}
    )
    metrics["显示值"] = [format_value(v, u, 4) if u in ("pct", "pp", "ratio") else format_value(v, u)
                       for v, u in zip(report.metrics["value"], report.metrics["unit"])]
    metrics["数值"] = [
        v if isinstance(v, (int, float)) or v is None else (f"{v:%Y-%m-%d}" if isinstance(v, pd.Timestamp) else str(v))
        for v in metrics["数值"]
    ]
    regression = report.regression_table()
    data = report.data.copy()
    data.index = data.index.strftime("%Y-%m-%d")
    data.index.name = "日期"
    quality = report.quality.summary()
    issues = pd.DataFrame({"需复核的问题": report.quality.issues()})
    conclusion_df = pd.DataFrame({"结论": conclusion(report).split("\n\n")})
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        scope.to_excel(writer, sheet_name="口径", index=False)
        metrics.to_excel(writer, sheet_name="指标", index=False)
        regression.to_excel(writer, sheet_name="回归", index=False)
        data.to_excel(writer, sheet_name="期间收益")
        quality.to_excel(writer, sheet_name="数据质量", index=False)
        issues.to_excel(writer, sheet_name="数据质量", index=False, startrow=len(quality) + 2)
        conclusion_df.to_excel(writer, sheet_name="结论", index=False)
        report.not_done.to_excel(writer, sheet_name="未完成的检验", index=False)
    return path
