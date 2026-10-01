"""多基金横向对比：同一区间、同一无风险收益，逐只评价后汇总成对比表（正文第十部分“形成结论”）。

``compare`` 对每只基金按各自的合同业绩比较基准（或统一给出的基准）运行 evaluate，单只失败时记录原因
并继续。对比表不做综合打分，默认按输入顺序排列；``sort`` 只按单一指标排序，并在表下注明限定语：
不同类型的基金不宜直接比较，只比较同类基金；样本少于 36 个月的基金单独标注；β 接近 0 或为负时
Treynor 不参与排序。
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from fundeval.report.inputs import CONTRACT, ReportOptions, build_report_inputs, fund_profile_or_none
from fundeval.report.summary import SHORT_SAMPLE_YEARS, EvaluationReport, evaluate, format_value

#: 可排序的指标：键 → (对比表列名, 是否降序)。只按单一指标排序，不做综合打分
SORT_KEYS: dict[str, tuple[str, bool]] = {
    "annualized_return": ("年化收益", True),
    "volatility": ("年化波动", False),
    "max_drawdown": ("最大回撤", False),
    "sharpe": ("Sharpe", True),
    "excess": ("累计超额", True),
    "tracking_error": ("TE", False),
    "ir": ("IR", True),
    "alpha": ("CAPM Alpha（算术年化）", True),
    "factor_alpha": ("多因子 Alpha（算术年化）", True),
    "treynor": ("Treynor", True),
}

#: β 不超过该值（接近 0 或为负）时 Treynor 不参与排序
TREYNOR_MIN_BETA = 0.1

#: 排序时写在表下的限定语
SORT_CAVEATS = (
    "不同类型的基金不宜直接比较，排序只在同类基金之间有意义；只比较同类基金。",
    f"样本少于 {SHORT_SAMPLE_YEARS * 12} 个月的基金已在“标注”列单独标注，其指标不稳定，排序仅供参考。",
    f"β 接近 0 或为负（β ≤ {TREYNOR_MIN_BETA}）时 Treynor 比率没有意义，不参与 Treynor 排序。",
)

# 对比表各列的单位：pct 百分比，pp 百分点，ratio 比率，count 期数，text 文字
_UNITS = {
    "期数": "count",
    "年化收益": "pct",
    "年化波动": "pct",
    "最大回撤": "pct",
    "Sharpe": "ratio",
    "累计超额": "pp",
    "TE": "pct",
    "IR": "ratio",
    "CAPM Alpha（算术年化）": "pct",
    "CAPM Alpha t": "ratio",
    "多因子 Alpha（算术年化）": "pct",
    "多因子 Alpha t": "ratio",
    "Treynor": "pct",
    "β": "ratio",
    "数据质量问题数": "count",
}


def _one_line(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _metric(rep: EvaluationReport, key: str) -> float:
    return float(rep.metrics.loc[key, "value"]) if key in rep.metrics.index else float("nan")


@dataclass
class ComparisonReport:
    """多基金对比结果。

    - ``table``：对比表（原始数值，小数口径），按输入顺序或 ``sort`` 排列
    - ``reports``：{基金代码: EvaluationReport}（成功的基金）；``failures``：{基金代码: 失败原因}
    - ``profiles``：{基金代码: FundProfile}（取到概况的基金）
    - ``scope``：共同口径（区间、频率、无风险收益、基准）；``notes``：附注
    """

    table: pd.DataFrame
    reports: dict[str, EvaluationReport]
    failures: dict[str, str]
    profiles: dict[str, Any]
    scope: dict[str, str]
    sort: str | None = None
    notes: list[str] = field(default_factory=list)

    def footnotes(self) -> list[str]:
        """表下说明：默认顺序或排序依据；给出 sort 时附限定语；基金类型不同时另行提醒。"""
        out = []
        if self.sort is None:
            out.append("按输入顺序排列，不做综合打分。")
        else:
            col, desc = SORT_KEYS[self.sort]
            out.append(f"按“{col}”{'从高到低' if desc else '从低到高'}排序（只按单一指标，不做综合打分）；失败或指标不适用的基金排在最后。")
            out += list(SORT_CAVEATS)
        types = sorted({t for t in self.table["类型"] if isinstance(t, str) and t})
        if len(types) > 1:
            out.append(f"本表含不同类型的基金（{'、'.join(types)}），各项指标不宜直接横向比较。")
        return out

    def display_table(self) -> pd.DataFrame:
        """格式化后的对比表：百分比带 %，NaN 显示为“—”。"""
        out = self.table.copy()
        for col, unit in _UNITS.items():
            if col not in out.columns:
                continue
            digits = 2 if unit in ("pct", "pp", "ratio") else None
            out[col] = [
                "—" if v is None or (isinstance(v, float) and math.isnan(v)) else format_value(v, unit, digits)
                for v in out[col]
            ]
        return out.fillna("")

    def fund_metrics(self, code: str) -> pd.DataFrame:
        """单只基金的关键指标（各维度指标表合并）；失败的基金返回失败原因。"""
        if code in self.failures:
            return pd.DataFrame({"项目": ["基金代码", "失败原因"], "内容": [code, self.failures[code]]})
        rep = self.reports[code]
        frames = []
        for name in dict.fromkeys(rep.metrics["section"]):
            sec = rep.section(name)
            sec.insert(0, "维度", name)
            frames.append(sec)
        return pd.concat(frames, ignore_index=True)

    def to_markdown(self, path: str | Path | None = None, *, charts: bool = False) -> str:
        """Markdown：口径、对比表、表下说明、附注与失败原因。

        ``charts=True`` 时（需 ``fundeval[plot]`` 与 ``path``）把多基金财富指数图保存到“<报告名>_files/”，
        在“图表”一节用相对路径嵌入。
        """
        from fundeval.report.export import _md_table, charts_dir, markdown_images

        if charts and path is None:
            raise ValueError("charts=True 须给出 path：图片保存到报告旁的“<报告名>_files/”目录")
        chart_lines: list[str] = []
        if charts:
            from fundeval.report.charts import comparison_charts

            chart_lines = markdown_images(comparison_charts(self, charts_dir(path)), path)

        parts = [
            "# 基金横向对比", "", "## 口径", "",
            _md_table(pd.DataFrame(list(self.scope.items()), columns=["项目", "内容"])),
            "", "## 对比表", "", _md_table(self.display_table()), "",
            *[f"- {n}" for n in self.footnotes()],
        ]
        if chart_lines:
            parts += ["", "## 图表", "", *chart_lines]
        if self.failures:
            parts += ["", "## 失败原因", "", *[f"- {code}：{reason}" for code, reason in self.failures.items()]]
        if self.notes:
            parts += ["", "## 附注", "", *[f"- {n}" for n in self.notes]]
        text = "\n".join(parts) + "\n"
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def to_excel(self, path: str | Path, *, charts: bool = False) -> Path:
        """Excel：“对比”sheet 为对比表（原始数值）及表下说明；每只基金一个 sheet（关键指标，失败时为失败原因）；
        另有“失败原因”与“口径”sheet。``charts=True`` 时另加“图表”sheet（多基金财富指数图，需 ``fundeval[plot]``）。"""
        import tempfile

        path = Path(path)
        with tempfile.TemporaryDirectory() as tmp:
            chart_set = None
            if charts:
                from fundeval.report.charts import comparison_charts

                chart_set = comparison_charts(self, tmp)
            self._write_excel(path, chart_set)
        return path

    def _write_excel(self, path: Path, chart_set) -> None:
        from fundeval.report.export import insert_chart_sheet

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            self.table.to_excel(writer, sheet_name="对比", index=False)
            notes = pd.DataFrame({"说明": self.footnotes()})
            notes.to_excel(writer, sheet_name="对比", index=False, startrow=len(self.table) + 2)
            pd.DataFrame(list(self.scope.items()), columns=["项目", "内容"]).to_excel(writer, sheet_name="口径", index=False)
            for code in self.table["基金代码"]:
                self.fund_metrics(code).to_excel(writer, sheet_name=str(code)[:31], index=False)
            pd.DataFrame(
                [(c, r) for c, r in self.failures.items()], columns=["基金代码", "失败原因"]
            ).to_excel(writer, sheet_name="失败原因", index=False)
            if chart_set is not None:
                insert_chart_sheet(writer.book, chart_set)


def _row(code: str, rep: EvaluationReport | None, profile, error: str | None, k: int, *, factors: bool,
         style: bool, treynor: bool, style_auto: bool = False) -> dict[str, Any]:
    nan = float("nan")
    row: dict[str, Any] = {
        "基金代码": code,
        "简称": getattr(profile, "short_name", None) or "",
        "类型": getattr(profile, "fund_type", None) or "",
        "基准收益类型": (rep.benchmark_return_type or "") if rep is not None else "",
        "期数": rep.n if rep is not None else nan,
    }
    keys = {
        "年化收益": "annualized_return",
        "年化波动": "volatility",
        "最大回撤": "max_drawdown",
        "Sharpe": "sharpe",
        "累计超额": "cumulative_difference",
        "TE": "tracking_error",
        "IR": "information_ratio",
        "CAPM Alpha（算术年化）": "alpha_annualized",
        "CAPM Alpha t": "alpha_t",
    }
    if factors:
        keys.update({"多因子 Alpha（算术年化）": "factor_alpha_annualized", "多因子 Alpha t": "factor_alpha_t"})
    for col, key in keys.items():
        row[col] = _metric(rep, key) if rep is not None else nan
    if treynor:
        row["Treynor"] = _metric(rep, "treynor") if rep is not None else nan
        row["β"] = _metric(rep, "beta") if rep is not None else nan
    if style:
        if style_auto:
            row["风格预设"] = (rep.labels.get("style_preset", "") if rep is not None else "")
        row["风格前两项"] = (
            "；".join(f"{name} {w:.1%}" for name, w in rep.style.top(2).items())
            if rep is not None and rep.style is not None else ""
        )
    row["数据质量问题数"] = len(rep.quality.issues()) if rep is not None else nan
    marks = []
    if rep is not None and rep.short_sample:
        marks.append(f"样本不足 {SHORT_SAMPLE_YEARS * 12} 个月（{rep.n} 期）")
    if treynor and rep is not None and not (row["β"] > TREYNOR_MIN_BETA):
        marks.append("β 接近 0 或为负，Treynor 不参与排序")
    row["标注"] = "；".join(marks)
    row["失败原因"] = error or ""
    return row


def _sort(table: pd.DataFrame, sort: str) -> pd.DataFrame:
    col, desc = SORT_KEYS[sort]
    if col not in table.columns:
        raise ValueError(f"对比表没有“{col}”列，不能按 {sort} 排序（多因子 Alpha 需给出 factors）")
    values = pd.to_numeric(table[col], errors="coerce")
    eligible = values.notna() & (table["失败原因"] == "")
    if sort == "treynor":
        eligible &= pd.to_numeric(table["β"], errors="coerce") > TREYNOR_MIN_BETA
    ranked = table[eligible].assign(_v=values[eligible]).sort_values("_v", ascending=not desc, kind="stable")
    return pd.concat([ranked.drop(columns="_v"), table[~eligible]], ignore_index=True)


class _Unset:
    """compare 参数的“未给出”标记：与 config 同时使用时区分调用方是否显式给出（显式给出者优先）。"""

    def __init__(self, default):
        self.default = default

    def __repr__(self) -> str:
        return repr(self.default)


def compare(
    codes: Sequence[str],
    start=_Unset(None),
    end=_Unset(None),
    freq: str = _Unset("M"),
    benchmark: str = _Unset(CONTRACT),
    *,
    sort: str | None = None,
    options: ReportOptions | None = None,
    config=None,
    **kwargs,
) -> ComparisonReport:
    """多基金横向对比：每只基金同一区间、同一无风险收益，逐只 evaluate，汇总成对比表。

    参数
    ----
    codes : 基金代码列表，对比表默认按此顺序
    start, end, freq : 共同区间与频率
    benchmark : ``"contract"``（默认，每只基金各自的合同业绩比较基准，见 etl.benchmark.resolve_benchmark），
        或统一的基准写法（"000300:0.8,H11001:0.2"、指数代码、合同文字）
    sort : 按单一指标排序（SORT_KEYS 的键，如 "sharpe"）；缺省按输入顺序。给出时表下写明限定语
    options : ReportOptions（取数选项，如 rf、factors、style、benchmark_map、缓存与超时）
    kwargs : ReportOptions 的字段（如 ``rf="cgb2y"``、``factors="cn_index_proxy"``、``benchmark_map={...}``、
        ``hac_lags=3``）并入 options；其余参数原样传给 evaluate（如 ``costs``、``robustness``）
    config : fundeval.config.EvaluationConfig（评价口径）；给出时取数选项（区间、频率、基准、无风险收益、风格、
        因子、缓存与超时等）与 evaluate 的 confidence、robustness、K 取自 config，显式给出的 start、end、freq、
        benchmark 与 kwargs 优先；不能与 options 同时给出。对比表口径写明“配置来源”

    无风险收益只取一次（常数或按 rf 的来源），各基金按所属期对齐，保证口径一致；取不到时整体报错。
    单只基金取数或评价失败时记录原因（写入对比表“失败原因”列）并继续，不中断整体。
    """
    from fundeval.etl import schema
    from fundeval.etl.sources import akshare as aks

    codes = [str(c).strip() for c in codes if str(c).strip()]
    if not codes:
        raise ValueError("至少需要一只基金")
    if len(set(codes)) != len(codes):
        raise ValueError(f"基金代码重复：{codes}")
    if sort is not None and sort not in SORT_KEYS:
        raise ValueError(f"sort 须为 {'、'.join(SORT_KEYS)} 之一，收到 {sort!r}")
    if config is not None and options is not None:
        raise ValueError("config 与 options 不能同时给出：取数选项写在 config 中，或用关键字参数覆盖")
    option_names = {f.name for f in dataclasses.fields(ReportOptions)}
    opt_kw = {k: v for k, v in kwargs.items() if k in option_names}
    evaluate_kwargs = {k: v for k, v in kwargs.items() if k not in option_names}
    if config is not None:
        base = config.report_options()
        evaluate_kwargs = {**config.evaluate_kwargs(), **evaluate_kwargs}
        cfg_defaults = {"start": config.start, "end": config.end, "freq": config.freq,
                        "benchmark": config.benchmark or CONTRACT}
    else:
        base = options or ReportOptions()
        cfg_defaults = {}

    def pick(name, value):
        if not isinstance(value, _Unset):
            return value
        return cfg_defaults.get(name, value.default)

    opts = dataclasses.replace(
        base, **opt_kw, start=pick("start", start), end=pick("end", end), freq=pick("freq", freq),
        benchmark=pick("benchmark", benchmark), fund=None, input=None,
    )
    if sort == "factor_alpha" and not opts.factors:
        raise ValueError("按多因子 Alpha 排序须给出 factors（如 cn_index_proxy）")
    freq_u = opts.freq.upper()
    k = int(evaluate_kwargs.get("periods_per_year", schema.periods_per_year(freq_u)))
    notes: list[str] = []

    # 同一无风险收益：只取一次
    rf_desc = None
    if opts.risk_free is None:
        rf_arg = opts.rf or "auto"
        try:
            rate = float(rf_arg)
        except (TypeError, ValueError):
            with aks.collect_notes() as rf_notes:
                series = aks.risk_free_returns(opts.start, opts.end, freq_u, source=rf_arg, **opts.cache())
            notes += rf_notes
            opts = dataclasses.replace(opts, risk_free=series)
            rf_desc = series.attrs.get("description")
        else:
            opts = dataclasses.replace(opts, rf=str(rate))
            rf_desc = aks.describe_rate_source(rate, freq_u)
    else:
        rf_desc = opts.risk_free.attrs.get("description") or "调用方给出的每期无风险收益"

    reports: dict[str, EvaluationReport] = {}
    failures: dict[str, str] = {}
    profiles: dict[str, Any] = {}
    rows = []
    for code in codes:
        rep = None
        error = None
        profile = None
        try:
            profile = fund_profile_or_none(code, opts.cache(), required=False)
            with aks.collect_notes() as source_notes:
                inputs = build_report_inputs(dataclasses.replace(opts, fund=code, profile=profile))
            inputs["notes"] = source_notes + inputs["notes"]
            inputs.update(evaluate_kwargs)
            rep = evaluate(**inputs)
        except Exception as exc:  # 单只失败不中断整体
            error = _one_line(exc)
            failures[code] = error
            rep = None
        else:
            reports[code] = rep
        if profile is not None:
            profiles[code] = profile
        rows.append(_row(code, rep, profile, error, k, factors=bool(opts.factors), style=bool(opts.style),
                         treynor=sort == "treynor", style_auto=bool(opts.style) and opts.style.strip().lower() == "auto"))
    table = pd.DataFrame(rows)
    if sort is not None:
        table = _sort(table, sort)

    scope = {
        "区间": f"{opts.start or '最早'} 至 {opts.end or '最新'}",
        "频率": f"{freq_u}，K = {k}",
        "基准": "各基金合同业绩比较基准（逐只解析，全收益优先）" if opts.benchmark == CONTRACT else str(opts.benchmark),
        "无风险收益": rf_desc or "—",
        "基金数": f"{len(codes)}（成功 {len(reports)}，失败 {len(failures)}）",
    }
    if opts.factors:
        scope["因子"] = opts.factors
    scope["汇率换算"] = (
        "convert：非人民币基准成分按国家外汇管理局人民币汇率中间价换算为人民币收益" if opts.fx == "convert"
        else "none：未做汇率换算，基准收益含汇率差异"
    )
    if opts.style:
        scope["风格指数"] = "auto（按各基金的基金类型选择预设，见“风格预设”列）" if opts.style.strip().lower() == "auto" else opts.style
    if config is not None:
        scope["配置来源"] = config.describe()
    return ComparisonReport(table=table, reports=reports, failures=failures, profiles=profiles, scope=scope,
                            sort=sort, notes=notes)


__all__ = ["ComparisonReport", "compare", "SORT_KEYS", "SORT_CAVEATS", "TREYNOR_MIN_BETA"]
