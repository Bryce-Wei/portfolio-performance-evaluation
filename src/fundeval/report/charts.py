"""报告图表（PNG）：正文第九部分“从原始数据到绩效结果”的图形展示。

- ``wealth``：组合与基准的财富指数（期初 = 1）
- ``drawdown``：回撤曲线，标出最大回撤的峰值、谷底与修复日期
- ``rolling``：滚动 12 期的超额收益（组合与基准滚动累计收益之差）与跟踪误差（年化），上下两幅
- ``style``：滚动风格权重的堆积面积图（有滚动风格分析时）
- ``compare``：多基金财富指数（期初 = 1）画在同一张图上

matplotlib 是可选依赖（``pip install "fundeval[plot]"``），按需导入；未安装时 require_matplotlib 报错并提示安装，
不影响其他功能。一律使用 Agg 后端，不弹窗。

中文字体按 CJK_FONTS 的顺序查找（Microsoft YaHei、SimHei、PingFang SC、Noto Sans CJK SC、WenQuanYi 等）；
都没有时改用英文标签并发出 RuntimeWarning，避免中文显示为方框。英文模式下风格资产与基金用代码标注。
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fundeval import returns as ret
from fundeval.etl import schema

PLOT_INSTALL_HINT = 'pip install "fundeval[plot]"'

#: 依次查找的中文字体
CJK_FONTS = (
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "Heiti SC",
    "STHeiti",
    "Arial Unicode MS",
)

#: 滚动超额收益与跟踪误差的默认窗口（期）
ROLLING_WINDOW = 12

#: 系列颜色：按固定顺序分配，不循环（超过时合并或截断并提示）
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
DPI = 144

_TEXT = {
    "zh": {
        "wealth_title": "财富指数（期初 = 1）",
        "portfolio": "组合",
        "benchmark": "基准",
        "drawdown_title": "回撤",
        "drawdown": "回撤",
        "peak": "峰值",
        "trough": "谷底",
        "recovery": "修复",
        "unrecovered": "未修复",
        "start": "期初",
        "rolling_title": "滚动 {w} 期超额收益与跟踪误差",
        "excess": "超额收益（{w} 期累计，组合 - 基准）",
        "te": "跟踪误差（年化）",
        "style_title": "滚动风格权重（窗口 {w} 期）",
        "weight": "权重",
        "compare_title": "基金财富指数对比（期初 = 1）",
        "date": "日期",
    },
    "en": {
        "wealth_title": "Wealth index (start = 1)",
        "portfolio": "Portfolio",
        "benchmark": "Benchmark",
        "drawdown_title": "Drawdown",
        "drawdown": "Drawdown",
        "peak": "Peak",
        "trough": "Trough",
        "recovery": "Recovery",
        "unrecovered": "Not recovered",
        "start": "Start",
        "rolling_title": "Rolling {w}-period excess return and tracking error",
        "excess": "Excess return ({w}-period cumulative, portfolio - benchmark)",
        "te": "Tracking error (annualized)",
        "style_title": "Rolling style weights ({w}-period window)",
        "weight": "Weight",
        "compare_title": "Fund wealth index comparison (start = 1)",
        "date": "Date",
    },
}

#: 图表文件名与中文标题（Markdown 与 Excel 的图注）
CHART_TITLES = {
    "wealth": "财富指数（期初 = 1）",
    "drawdown": "回撤曲线",
    "rolling": f"滚动 {ROLLING_WINDOW} 期超额收益与跟踪误差",
    "style": "滚动风格权重",
    "compare": "基金财富指数对比（期初 = 1）",
}


def require_matplotlib():
    """导入 matplotlib 并切换到 Agg 后端；未安装时报错并提示 ``pip install "fundeval[plot]"``。"""
    try:
        import matplotlib
    except ImportError as exc:
        raise ImportError(f"生成图表需要 matplotlib：{PLOT_INSTALL_HINT}") from exc
    matplotlib.use("Agg", force=True)
    return matplotlib


def find_cjk_font() -> str | None:
    """按 CJK_FONTS 的顺序返回第一个已安装的中文字体名；都没有时返回 None。"""
    require_matplotlib()
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    return next((name for name in CJK_FONTS if name in installed), None)


def chart_language() -> tuple[str, str | None]:
    """返回 (语言, 字体)：有中文字体时为 ("zh", 字体名)，否则发出警告并返回 ("en", None)。"""
    font = find_cjk_font()
    if font is None:
        warnings.warn(
            "未找到中文字体（已查找：" + "、".join(CJK_FONTS) + "），图表改用英文标签，避免中文显示为方框",
            RuntimeWarning, stacklevel=3,
        )
        return "en", None
    return "zh", font


@dataclass
class ChartSet:
    """生成的图表：``paths`` 为 {图表键: PNG 路径}（按生成顺序），``lang`` 为标签语言，``skipped`` 为未生成的图表与原因。"""

    paths: dict[str, Path]
    lang: str
    font: str | None = None
    skipped: dict[str, str] = field(default_factory=dict)


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _figure(nrows: int = 1, height: float = 3.6):
    require_matplotlib()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows, 1, figsize=(8, height * nrows), sharex=nrows > 1, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes[:, 0]:
        _style_axes(ax)
    return fig, list(axes[:, 0])


def _percent_axis(ax) -> None:
    from matplotlib.ticker import PercentFormatter

    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def _with_start(wealth: pd.Series) -> pd.Series:
    """在第一个日期之前补一个期初点（值为 1），使曲线从 1 起画。期初日期取首期之前一个同频间隔。"""
    if wealth.empty:
        return wealth
    idx = wealth.index
    step = idx[1] - idx[0] if len(idx) > 1 else pd.Timedelta(days=1)
    start = pd.Series([1.0], index=pd.DatetimeIndex([idx[0] - step]))
    return pd.concat([start, wealth])


def _legend(ax) -> None:
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left")
    for text in leg.get_texts():
        text.set_color(INK)


def _title(ax, text: str) -> None:
    ax.set_title(text, color=INK, fontsize=11, loc="left")


def wealth_figure(data: pd.DataFrame, lang: str = "zh"):
    """组合与基准的财富指数（期初 = 1）。``data`` 为 EvaluationReport.data（portfolio、可选 benchmark）。"""
    t = _TEXT[lang]
    fig, (ax,) = _figure()
    p = _with_start(ret.wealth_index(data[schema.PORTFOLIO]))
    ax.plot(p.index, p.to_numpy(), color=SERIES_COLORS[0], linewidth=2, label=t["portfolio"])
    if schema.BENCHMARK in data.columns:
        b = _with_start(ret.wealth_index(data[schema.BENCHMARK]))
        ax.plot(b.index, b.to_numpy(), color=SERIES_COLORS[1], linewidth=2, label=t["benchmark"])
        _legend(ax)
    ax.axhline(1.0, color=INK_SECONDARY, linewidth=0.8)
    _title(ax, t["wealth_title"])
    fig.tight_layout()
    return fig


def drawdown_figure(data: pd.DataFrame, lang: str = "zh"):
    """回撤曲线（期初财富 = 1 计入峰值），标出最大回撤的峰值、谷底与修复日期（report.summary._drawdown_details）。"""
    from fundeval.report.summary import _drawdown_details

    t = _TEXT[lang]
    p = data[schema.PORTFOLIO]
    w = _with_start(ret.wealth_index(p))
    dd = w / w.cummax() - 1
    fig, (ax,) = _figure()
    ax.fill_between(dd.index, dd.to_numpy(), 0, color=SERIES_COLORS[0], alpha=0.25, linewidth=0)
    ax.plot(dd.index, dd.to_numpy(), color=SERIES_COLORS[0], linewidth=2, label=t["drawdown"])
    ax.axhline(0, color=INK_SECONDARY, linewidth=0.8)
    details = _drawdown_details(p)
    trough = details["trough"]
    if trough is not None:
        peak = w.index[0] if details["peak"] == "期初" else details["peak"]
        peak_label = t["peak"]
        if details["peak"] == "期初":
            peak_label += f"（{t['start']}）" if lang == "zh" else f" ({t['start']})"
        points = [(peak, peak_label), (trough, t["trough"])]
        if isinstance(details["recovery"], pd.Timestamp):
            points.append((details["recovery"], t["recovery"]))
        for i, (date, label) in enumerate(points):
            y = float(dd.loc[date])
            # 期初是补出的点，没有真实日期，只标“峰值（期初）”
            text = label if i == 0 and details["peak"] == "期初" else f"{label} {date:%Y-%m-%d}"
            ax.plot([date], [y], marker="o", markersize=6, color=SERIES_COLORS[0],
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
            ax.annotate(text, (date, y), textcoords="offset points",
                        xytext=(4, -12 if label == t["trough"] else 6), fontsize=8, color=INK)
        if not isinstance(details["recovery"], pd.Timestamp):
            ax.annotate(t["unrecovered"], (dd.index[-1], float(dd.iloc[-1])), textcoords="offset points",
                        xytext=(-40, 6), fontsize=8, color=INK_SECONDARY)
    _percent_axis(ax)
    _title(ax, t["drawdown_title"])
    fig.tight_layout()
    return fig


def rolling_active(data: pd.DataFrame, periods_per_year: int, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """滚动 window 期的超额收益（组合与基准的滚动累计收益之差）与跟踪误差（主动收益样本标准差 × √K）。

    列为 excess、te；前 window − 1 期为 NaN。需要 benchmark 列。
    """
    if schema.BENCHMARK not in data.columns:
        raise ValueError("滚动超额收益与跟踪误差需要基准收益")
    p, b = data[schema.PORTFOLIO].astype(float), data[schema.BENCHMARK].astype(float)
    cum = lambda s: (1 + s).rolling(window).apply(np.prod, raw=True) - 1  # noqa: E731
    excess = cum(p) - cum(b)
    te = (p - b).rolling(window).std(ddof=1) * np.sqrt(periods_per_year)
    return pd.DataFrame({"excess": excess, "te": te})


def rolling_figure(data: pd.DataFrame, periods_per_year: int, lang: str = "zh", window: int = ROLLING_WINDOW):
    """滚动超额收益与跟踪误差：两个量含义不同，分上下两幅，各用一个纵轴。"""
    t = _TEXT[lang]
    roll = rolling_active(data, periods_per_year, window).dropna()
    fig, (ax1, ax2) = _figure(2, height=2.6)
    ax1.plot(roll.index, roll["excess"].to_numpy(), color=SERIES_COLORS[0], linewidth=2)
    ax1.axhline(0, color=INK_SECONDARY, linewidth=0.8)
    ax1.set_title(t["excess"].format(w=window), color=INK, fontsize=9, loc="left")
    ax2.plot(roll.index, roll["te"].to_numpy(), color=SERIES_COLORS[0], linewidth=2)
    ax2.set_title(t["te"], color=INK, fontsize=9, loc="left")
    ax2.set_ylim(bottom=0)
    for ax in (ax1, ax2):
        _percent_axis(ax)
    fig.suptitle(t["rolling_title"].format(w=window), color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    return fig


def style_figure(weights: pd.DataFrame, window: int, lang: str = "zh", codes: Mapping[str, str] | None = None):
    """滚动风格权重的堆积面积图（权重非负、合计为 1）。英文模式下列名换成 ``codes`` 中的代码（cash 即现金）。"""
    t = _TEXT[lang]
    w = weights.astype(float).clip(lower=0)
    names = [str(c) for c in w.columns]
    if lang == "en":
        names = [str((codes or {}).get(c, c)) for c in names]
        names = [n if n.isascii() else f"Asset {i + 1}" for i, n in enumerate(names)]
    if len(names) > len(SERIES_COLORS):
        raise ValueError(f"风格资产超过 {len(SERIES_COLORS)} 个，堆积图无法区分，请减少风格资产")
    fig, (ax,) = _figure()
    ax.stackplot(w.index, w.T.to_numpy(), labels=names, colors=SERIES_COLORS[: len(names)],
                 edgecolor=SURFACE, linewidth=1)
    ax.set_ylim(0, 1)
    ax.margins(x=0)
    _percent_axis(ax)
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for text in leg.get_texts():
        text.set_color(INK)
    _title(ax, t["style_title"].format(w=window))
    fig.tight_layout()
    return fig


def compare_figure(wealth: Mapping[str, pd.Series], lang: str = "zh"):
    """多基金财富指数（各自期初 = 1）画在同一张图上。``wealth`` 为 {图例名: 单期收益}，最多 8 只。"""
    t = _TEXT[lang]
    if len(wealth) > len(SERIES_COLORS):
        raise ValueError(f"同一张图最多画 {len(SERIES_COLORS)} 只基金，收到 {len(wealth)} 只；请分组对比")
    fig, (ax,) = _figure()
    for color, (name, r) in zip(SERIES_COLORS, wealth.items()):
        w = _with_start(ret.wealth_index(r))
        ax.plot(w.index, w.to_numpy(), color=color, linewidth=2, label=name)
    ax.axhline(1.0, color=INK_SECONDARY, linewidth=0.8)
    _legend(ax)
    _title(ax, t["compare_title"])
    fig.tight_layout()
    return fig


def _save(fig, path: Path) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _font_context(font: str | None):
    import matplotlib

    if font is None:
        return matplotlib.rc_context({"axes.unicode_minus": False})
    return matplotlib.rc_context({"font.sans-serif": [font, *matplotlib.rcParams["font.sans-serif"]],
                                  "font.family": "sans-serif", "axes.unicode_minus": False})


def report_charts(report, out_dir, *, window: int = ROLLING_WINDOW) -> ChartSet:
    """为 EvaluationReport 生成 PNG 图表到 ``out_dir``：wealth、drawdown，有基准且期数不少于 window 时 rolling，
    有滚动风格分析时 style。未生成的图表及原因记入 ``skipped``。"""
    require_matplotlib()
    lang, font = chart_language()
    out = Path(out_dir)
    paths: dict[str, Path] = {}
    skipped: dict[str, str] = {}
    data = report.data
    with _font_context(font):
        paths["wealth"] = _save(wealth_figure(data, lang), out / "wealth.png")
        paths["drawdown"] = _save(drawdown_figure(data, lang), out / "drawdown.png")
        if schema.BENCHMARK not in data.columns:
            skipped["rolling"] = "未提供基准"
        elif len(data) < window:
            skipped["rolling"] = f"样本 {len(data)} 期，少于滚动窗口 {window} 期"
        else:
            paths["rolling"] = _save(rolling_figure(data, report.periods_per_year, lang, window), out / "rolling.png")
        if report.style_rolling is not None:
            codes = _style_codes(report)
            paths["style"] = _save(
                style_figure(report.style_rolling.weights, report.style_rolling.window, lang, codes), out / "style.png"
            )
    return ChartSet(paths=paths, lang=lang, font=font, skipped=skipped)


def _style_codes(report) -> dict[str, str]:
    """风格列名 → 代码（英文标签用）：从口径“风格指数”（如“沪深300成长：H00918（全收益）”）解析。"""
    text = report.labels.get("style", "")
    out = {}
    for part in text.split("；"):
        name, sep, rest = part.partition("：")
        if sep:
            code = rest.split("（")[0].strip()
            out[name.strip()] = "cash" if "无风险收益" in rest else code
    return out


def comparison_charts(comparison, out_dir) -> ChartSet:
    """为 ComparisonReport 生成多基金财富指数图 compare.png（成功的基金，最多 8 只）。"""
    require_matplotlib()
    lang, font = chart_language()
    out = Path(out_dir)
    series = {}
    for code, rep in comparison.reports.items():
        name = code
        profile = comparison.profiles.get(code)
        if lang == "zh" and profile is not None and getattr(profile, "short_name", None):
            name = f"{code} {profile.short_name}"
        series[name] = rep.data[schema.PORTFOLIO]
    if not series:
        return ChartSet(paths={}, lang=lang, font=font, skipped={"compare": "没有评价成功的基金"})
    with _font_context(font):
        path = _save(compare_figure(series, lang), out / "compare.png")
    return ChartSet(paths={"compare": path}, lang=lang, font=font)


__all__ = [
    "CJK_FONTS", "CHART_TITLES", "ChartSet", "ROLLING_WINDOW", "chart_language", "comparison_charts",
    "compare_figure", "drawdown_figure", "find_cjk_font", "report_charts", "require_matplotlib", "rolling_active",
    "rolling_figure", "style_figure", "wealth_figure",
]
