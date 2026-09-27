"""一键评价报告：正文第九部分“从原始数据到绩效结果”与第十部分“形成可用于管理决策的结论”。

``evaluate`` 按正文六个评价维度组织结果：口径、收益表现、风险效率、下行与尾部、
Alpha 质量、持续监控，另附数据质量与未完成的检验。``conclusion`` 按第十部分把
“观察到的表现”“可以支持的解释”“需要进一步验证的判断”分三段写成文字。

口径沿用各模块：比率使用同频算术均值与样本标准差（n−1），乘以 √K 年化；
累计与年化收益为几何口径；M² 差额减去基准年化算术均值；分母为零时为 NaN。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fundeval import monitor, risk, tail
from fundeval import returns as ret
from fundeval.alpha.regression import RegressionResult, capm_regression
from fundeval.attribution.timing import TimingResult, henriksson_merton, treynor_mazuy
from fundeval.etl import clean, schema
from fundeval.etl.benchmark import PRICE_INDEX_CAVEAT, needs_price_caveat
from fundeval.etl.quality import QualityReport, data_quality_report

#: 正文第十部分：样本少于 36 个月（3 年）时只能写“观察到模型未解释的收益”
SHORT_SAMPLE_YEARS = 3
#: 大样本双侧 5% 的 t 值参考阈值
T_THRESHOLD = 1.96
#: 历史 VaR / ES 尾部观测少于该数时加注样本过小
MIN_TAIL_OBS = 5
#: 默认费用口径
DEFAULT_FEE_BASIS = "费用后净值"

SECTIONS = ("收益表现", "风险效率", "下行与尾部", "Alpha 质量", "持续监控")

_FREQ_NAMES = {252: "日度", 52: "周度", 12: "月度", 4: "季度", 1: "年度"}

# 单位：pct 百分比；pp 个百分点；ratio 比率；count 期数；date 日期；text 文字
PCT, PP, RATIO, COUNT, DATE, TEXT = "pct", "pp", "ratio", "count", "date", "text"


def format_value(value: Any, unit: str, digits: int | None = None) -> str:
    """按单位格式化数值：百分比带 %，百分点带“个百分点”，NaN 显示为“不适用”。"""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "不适用"
    if unit == PCT:
        return f"{value * 100:.{2 if digits is None else digits}f}%"
    if unit == PP:
        return f"{value * 100:.{2 if digits is None else digits}f} 个百分点"
    if unit == RATIO:
        return f"{value:.{2 if digits is None else digits}f}"
    if unit == COUNT:
        return f"{int(value)}"
    if unit == DATE:
        return value if isinstance(value, str) else f"{pd.Timestamp(value):%Y-%m-%d}"
    return str(value)


@dataclass(frozen=True)
class Metric:
    section: str
    key: str
    label: str
    value: Any
    unit: str
    note: str = ""


@dataclass
class EvaluationReport:
    """评价报告。

    - ``scope``：口径（有序字典，项目 → 内容）
    - ``metrics``：各维度指标表，索引为指标键，列为 section、label、value、unit、note
    - ``capm`` / ``timing``：CAPM 回归与 TM、HM 择时回归结果（样本不足时为 None）
    - ``monitoring``：持续监控结果（未给出 monitor_targets 时为 None）
    - ``quality``：数据质量报告
    - ``not_done``：未完成的检验（检验、状态、原因）
    - ``data``：对齐后的期间收益表
    - ``notes``：附注（如尾部样本过小、以基准代替市场）
    """

    scope: dict[str, str]
    metrics: pd.DataFrame
    capm: RegressionResult | None
    timing: dict[str, TimingResult | None]
    monitoring: dict[str, Any] | None
    quality: QualityReport
    not_done: pd.DataFrame
    data: pd.DataFrame
    periods_per_year: int
    labels: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    benchmark_return_type: str | None = None

    @property
    def price_index_caveat(self) -> bool:
        """基准为价格指数、含价格指数成分或收益类型未知，结论须写明价格指数的限定语。"""
        return needs_price_caveat(self.benchmark_return_type)

    @property
    def n(self) -> int:
        return len(self.data)

    @property
    def years(self) -> float:
        return self.n / self.periods_per_year

    @property
    def short_sample(self) -> bool:
        """样本少于 36 个月（或等价期数）。"""
        return self.years < SHORT_SAMPLE_YEARS - 1e-12

    @property
    def title(self) -> str:
        return self.labels.get("title") or f"{self.labels.get('portfolio', '组合')} 绩效评价报告"

    def metric(self, key: str) -> Any:
        """按键取指标数值，如 ``report.metric("sharpe")``。"""
        return self.metrics.loc[key, "value"]

    def section(self, name: str) -> pd.DataFrame:
        """某一维度的指标表（指标、数值、口径说明），数值已格式化。"""
        rows = self.metrics[self.metrics["section"] == name]
        return pd.DataFrame(
            {
                "指标": rows["label"],
                "数值": [format_value(v, u, 4 if u in (PCT, PP, RATIO) else None) for v, u in zip(rows["value"], rows["unit"])],
                "口径说明": rows["note"],
            }
        ).reset_index(drop=True)

    def regression_table(self) -> pd.DataFrame:
        """回归系数表：模型、系数、估计值、标准误、t、p、95% 置信区间、R²、n、标准误类型。"""
        rows = []
        models = [("CAPM", self.capm), ("Treynor–Mazuy", self.timing.get("TM")), ("Henriksson–Merton", self.timing.get("HM"))]
        for name, res in models:
            if res is None:
                continue
            se_type = _se_label(res)
            tbl = res.table()
            for coef, row in tbl.iterrows():
                rows.append(
                    {
                        "模型": name,
                        "系数": coef,
                        "估计值": row["coef"],
                        "标准误": row["se"],
                        "t": row["t"],
                        "p": row["p"],
                        "95% 下限": row["ci_lower"],
                        "95% 上限": row["ci_upper"],
                        "R²": res.rsquared,
                        "n": res.n,
                        "标准误类型": se_type,
                    }
                )
        return pd.DataFrame(rows)

    def conclusion(self) -> str:
        return conclusion(self)


def _se_label(res: RegressionResult) -> str:
    dist = "t 分布" if res.use_t else "正态近似"
    if res.cov_type == "HAC":
        return f"HAC（Newey–West，滞后 {res.hac_lags}，{dist}）"
    return f"OLS（{dist}）"


def _as_frame(returns, benchmark, risk_free, market) -> tuple[pd.Series, Any, Any, Any]:
    if isinstance(returns, pd.DataFrame):
        if schema.PORTFOLIO not in returns.columns:
            raise schema.SchemaError(f"收益表缺少列 {schema.PORTFOLIO!r}")
        if benchmark is None and schema.BENCHMARK in returns.columns:
            benchmark = returns[schema.BENCHMARK]
        if market is None and schema.MARKET in returns.columns:
            market = returns[schema.MARKET]
        if schema.RISK_FREE in returns.columns:
            if not (np.isscalar(risk_free) and float(risk_free) == 0.0):
                raise ValueError("收益表已含 risk_free 列，不能再传入 risk_free 参数")
            risk_free = returns[schema.RISK_FREE]
        returns = returns[schema.PORTFOLIO]
    if not isinstance(returns, pd.Series) or not isinstance(returns.index, pd.DatetimeIndex):
        raise schema.SchemaError("returns 须为以 DatetimeIndex 为索引的 Series 或标准收益表")
    return returns, benchmark, risk_free, market


def _drawdown_details(p: pd.Series) -> dict[str, Any]:
    """最大回撤的峰值、谷底、修复日期与期数。峰值可以是期初（财富指数初值 1）。"""
    w = ret.wealth_index(p)
    dd = risk.drawdown(p)
    mdd = float(-dd.min()) if len(dd) else float("nan")
    out = {"peak": None, "trough": None, "recovery": None, "drawdown_periods": float("nan"), "recovery_periods": float("nan")}
    if not np.isfinite(mdd) or mdd <= 0:
        return out
    trough_pos = int(np.argmin(dd.to_numpy()))
    before = w.iloc[: trough_pos + 1]
    peak_val = max(1.0, float(before.max()))
    if float(before.max()) >= 1.0:
        peak_pos = int(np.argmax(before.to_numpy()))
        out["peak"] = w.index[peak_pos]
        out["drawdown_periods"] = trough_pos - peak_pos
    else:
        out["peak"] = "期初"
        out["drawdown_periods"] = trough_pos + 1
    out["trough"] = w.index[trough_pos]
    after = w.iloc[trough_pos + 1 :]
    recovered = after[after >= peak_val - 1e-15]
    if len(recovered):
        rec_pos = w.index.get_loc(recovered.index[0])
        out["recovery"] = recovered.index[0]
        out["recovery_periods"] = rec_pos - trough_pos
    else:
        out["recovery"] = "未修复"
    return out


def _rf_description(risk_free, labels: Mapping[str, str]) -> str:
    if "risk_free" in labels:
        return labels["risk_free"]
    if np.isscalar(risk_free):
        return f"常数，每期 {float(risk_free):.4%}"
    return "逐期序列（调用方提供）"


_PART_NAMES = {schema.PORTFOLIO: "组合", schema.BENCHMARK: "基准", schema.MARKET: "市场"}


def _date_runs(index: pd.DatetimeIndex, dropped: pd.DatetimeIndex) -> list[str]:
    """把被丢弃的日期按其在原序列中的连续位置分段，写成“起 至 止（n 期）”。"""
    pos = np.sort(index.get_indexer(dropped))
    runs, start = [], 0
    for i in range(1, len(pos) + 1):
        if i == len(pos) or pos[i] != pos[i - 1] + 1:
            a, b = index[pos[start]], index[pos[i - 1]]
            n = i - start
            runs.append(f"{a:%Y-%m-%d}" if n == 1 else f"{a:%Y-%m-%d} 至 {b:%Y-%m-%d}（{n} 期）")
            start = i
    return runs


def _alignment(raw_parts: Mapping[str, pd.Series], common: pd.DatetimeIndex) -> tuple[str, list[str]]:
    """对齐口径（“组合 N 期、基准 M 期、共同 K 期”）与被丢弃期的附注。"""
    counts = [f"{_PART_NAMES.get(k, k)} {len(v)} 期" for k, v in raw_parts.items()]
    text = "、".join(counts) + f"、共同 {len(common)} 期"
    notes = []
    for key, series in raw_parts.items():
        idx = pd.DatetimeIndex(series.index)
        dropped = idx.difference(common)
        if len(dropped):
            notes.append(
                f"对齐时丢弃{_PART_NAMES.get(key, key)}不在共同日期内的 {len(dropped)} 期：" + "；".join(_date_runs(idx, dropped))
            )
    return text, notes


def _fmt_date(value) -> str:
    return value if isinstance(value, str) else f"{pd.Timestamp(value):%Y-%m-%d}"


def evaluate(
    returns,
    benchmark=None,
    risk_free=0.0,
    periods_per_year: int = 12,
    *,
    market=None,
    mar=None,
    hac_lags: int | None = None,
    monitor_targets: Mapping[str, float] | None = None,
    labels: Mapping[str, str] | None = None,
    cross_check: pd.DataFrame | None = None,
    confidence: float = 0.95,
    use_t: bool | None = None,
    notes: list[str] | tuple[str, ...] | None = None,
) -> EvaluationReport:
    """生成评价报告（正文第九、十部分）。

    参数
    ----
    returns : 组合单期收益 Series，或含 portfolio（及 benchmark、risk_free、market）列的标准收益表
    benchmark : 基准单期收益；缺省时不计算相对指标
    risk_free : 每期无风险收益，常数或序列
    periods_per_year : K，月度 12
    market : 市场收益，用于 Treynor、CAPM 与择时回归；缺省时以基准代替并在口径中注明
    mar : Sortino 的最低可接受收益（同频）；缺省时取无风险收益并注明
    hac_lags : 给出时回归使用 Newey–West 标准误
    monitor_targets : {"target_active_return": 年化目标主动收益, "target_te": 年化目标 TE,
        "window": 窗口期数}，给出时计算风险倍数、z 值与分区（阈值为正文演示值）
    labels : {"title", "portfolio", "benchmark", "market", "risk_free", "fees", "benchmark_note",
        "benchmark_return_type", "benchmark_source"}，用于口径说明；fees 缺省为“费用后净值”；
        benchmark_return_type 为“全收益”“价格指数”“含价格指数成分”或“未知”
        （见 etl.benchmark.benchmark_return_type），后三者会在口径、附注与结论中写明
        “价格指数不含成分股分红，超额收益与 Alpha 会高估约为股息率的幅度”
    cross_check : etl.quality.nav_growth_check 的结果，列入数据质量报告
    confidence : VaR / ES 置信水平
    use_t : 传给 CAPM、TM、HM 回归；True 时 HAC 的 p 值与置信区间也用 t 分布（小样本建议）
    notes : 调用方附加的说明（如数据源自动切换、使用旧缓存），写入附注

    组合、基准、市场按日期取交集；交集内仍有缺失时报错，不填零。口径中写明
    “组合 N 期、基准 M 期、共同 K 期”，有期数被丢弃时在附注中列出其起止日期；
    数据质量报告按日期并集统计缺失。
    """
    labels = dict(labels or {})
    k = int(periods_per_year)
    p_raw, benchmark, risk_free, market = _as_frame(returns, benchmark, risk_free, market)

    raw_parts = {schema.PORTFOLIO: p_raw}
    if benchmark is not None:
        raw_parts[schema.BENCHMARK] = benchmark
    if market is not None:
        raw_parts[schema.MARKET] = market
    outer = clean.align(*raw_parts.values(), how="outer", names=list(raw_parts))
    quality = data_quality_report(outer, cross_check=cross_check)

    common = pd.DatetimeIndex(p_raw.index)
    for part in raw_parts.values():
        common = common.intersection(pd.DatetimeIndex(part.index))
    if len(common) == 0:
        counts = "、".join(f"{_PART_NAMES.get(k, k)} {len(v)} 期" for k, v in raw_parts.items())
        raise ValueError(f"组合与基准（市场）没有共同日期（{counts}），请检查日期标签是否一致（如月末交易日与日历月末）")
    alignment_text, alignment_notes = _alignment(raw_parts, common)
    df = clean.align_returns(p_raw, benchmark, risk_free if not np.isscalar(risk_free) else float(risk_free), market)
    p = df[schema.PORTFOLIO]
    rf = df[schema.RISK_FREE]
    b = df[schema.BENCHMARK] if benchmark is not None else None
    market_proxy = market is None and b is not None
    m = df[schema.MARKET] if market is not None else b
    extra_notes = list(notes or [])
    notes: list[str] = list(alignment_notes)
    metrics: list[Metric] = []

    def add(section, key, label, value, unit, note=""):
        metrics.append(Metric(section, key, label, value, unit, note))

    # ------------------------------ 收益表现 ------------------------------
    s = "收益表现"
    add(s, "cumulative_return", "累计收益（组合）", ret.cumulative_return(p), PCT, "几何连乘 ∏(1 + r) − 1")
    add(s, "annualized_return", "年化收益（组合）", ret.annualized_return(p, k), PCT, f"几何年化，T = n / K = {len(p)}/{k}")
    if b is not None:
        add(s, "benchmark_cumulative_return", "累计收益（基准）", ret.cumulative_return(b), PCT, "几何连乘")
        add(s, "benchmark_annualized_return", "年化收益（基准）", ret.annualized_return(b, k), PCT, "几何年化")
        add(s, "cumulative_difference", "累计收益差额", ret.cumulative_difference(p, b), PP, "R_p − R_b")
        add(s, "relative_return", "几何相对收益", ret.relative_return(p, b), PCT, "(1 + R_p) / (1 + R_b) − 1")

    # ------------------------------ 风险效率 ------------------------------
    s = "风险效率"
    add(s, "volatility", "年化波动率", risk.volatility(p, k), PCT, "样本标准差 × √K")
    add(s, "max_drawdown", "最大回撤", risk.max_drawdown(p), PCT, f"按{_FREQ_NAMES.get(k, f'K={k} ')}数据计算，峰值含期初")
    dd = _drawdown_details(p)
    add(s, "max_drawdown_peak", "回撤峰值日期", dd["peak"], DATE)
    add(s, "max_drawdown_trough", "回撤谷底日期", dd["trough"], DATE)
    add(s, "max_drawdown_recovery", "修复日期", dd["recovery"], DATE, "财富指数回到前高的首期")
    add(s, "drawdown_periods", "回撤期数（峰值至谷底）", dd["drawdown_periods"], COUNT)
    add(s, "recovery_periods", "修复期数（谷底至修复）", dd["recovery_periods"], COUNT, "未修复时不适用")
    add(s, "sharpe", "Sharpe 比率", risk.sharpe_ratio(p, rf, k), RATIO, "年化：超额收益算术均值 / 样本标准差 × √K")
    if b is not None:
        add(s, "tracking_error", "跟踪误差 TE", risk.tracking_error(p, b, k), PCT, "年化：主动收益样本标准差 × √K")
        add(s, "information_ratio", "信息比率 IR", risk.information_ratio(p, b, k), RATIO, "年化：主动收益算术均值 / TE")
    if m is not None:
        note = "K × 超额收益均值 / β" + ("；以基准代替市场" if market_proxy else "")
        add(s, "treynor", "Treynor 比率", risk.treynor_ratio(p, m, rf, k), PCT, note)
    if b is not None:
        add(s, "m_squared", "M²", risk.m_squared(p, b, rf, k), PCT, "年化算术口径，按基准波动率调整")
        add(s, "m_squared_difference", "M² 差额", risk.m_squared_difference(p, b, rf, k), PP, "M² − 基准年化算术均值")

    # ------------------------------ 下行与尾部 ------------------------------
    s = "下行与尾部"
    mar_value = rf if mar is None else mar
    mar_note = "MAR = 无风险收益（未单独指定）" if mar is None else (
        f"MAR = 每期 {float(mar):.4%}" if np.isscalar(mar) else "MAR = 调用方给出的序列"
    )
    if mar is None:
        notes.append("Sortino 的 MAR 未单独指定，取无风险收益；正文要求 MAR 事前约定。")
    dd_per = tail.downside_deviation(p, mar_value)
    add(s, "downside_deviation", "下行偏差（年化）", dd_per * math.sqrt(k), PCT, f"每期下行偏差 × √K，分母为全样本 n；{mar_note}")
    add(s, "sortino", "Sortino 比率", tail.sortino_ratio(p, mar_value, k), RATIO, f"年化；{mar_note}")
    add(s, "calmar", "Calmar 比率", tail.calmar_ratio(p, k), RATIO, "几何年化收益 / 最大回撤")
    tail_obs = (1 - confidence) * len(p)
    tail_note = f"历史模拟，单期损失以正数报告；尾部观测 {tail_obs:.1f} 个"
    if tail_obs < MIN_TAIL_OBS:
        tail_note += f"（少于 {MIN_TAIL_OBS} 个，样本过小，仅供参考）"
        notes.append(f"历史 VaR / ES 的尾部观测只有 {tail_obs:.1f} 个，样本过小，结果仅供参考。")
    conf_label = f"{confidence:.0%}"
    add(s, "var", f"历史 VaR（{conf_label}）", tail.historical_var(p, confidence), PCT, tail_note)
    add(s, "es", f"历史 ES（{conf_label}）", tail.historical_es(p, confidence), PCT, tail_note)

    # ------------------------------ Alpha 质量 ------------------------------
    s = "Alpha 质量"
    capm = None
    timing: dict[str, TimingResult | None] = {"TM": None, "HM": None}
    regression_error = None
    if m is not None:
        try:
            capm = capm_regression(p, m, rf, hac_lags=hac_lags, use_t=use_t)
            timing["TM"] = treynor_mazuy(p, m, rf, hac_lags=hac_lags, use_t=use_t)
            timing["HM"] = henriksson_merton(p, m, rf, hac_lags=hac_lags, use_t=use_t)
        except ValueError as exc:
            regression_error = str(exc)
    if capm is not None:
        se = _se_label(capm)
        proxy = "；以基准代替市场" if market_proxy else ""
        add(s, "alpha", "CAPM Alpha（每期）", capm.alpha, PCT, f"截距，{_FREQ_NAMES.get(k, '')}{proxy}")
        add(s, "alpha_annualized", "CAPM Alpha（算术年化）", capm.annualized_alpha(k), PCT, "α × K，算术口径")
        add(s, "alpha_t", "Alpha t 值", capm.alpha_t, RATIO, se)
        add(s, "alpha_p", "Alpha p 值", capm.alpha_p, RATIO, "双侧")
        add(s, "alpha_ci_lower", "Alpha 95% 置信下限（每期）", float(capm.conf_int.loc["alpha", "lower"]), PCT)
        add(s, "alpha_ci_upper", "Alpha 95% 置信上限（每期）", float(capm.conf_int.loc["alpha", "upper"]), PCT)
        add(s, "beta", "CAPM Beta", float(capm.betas["market"]), RATIO)
        add(s, "r_squared", "R²", capm.rsquared, RATIO)
        add(s, "regression_n", "回归样本期数 n", capm.n, COUNT)
        add(s, "hac", "是否 HAC", "是" if capm.cov_type == "HAC" else "否", TEXT, se)
    for key, res in timing.items():
        if res is None:
            continue
        name = "TM" if key == "TM" else "HM"
        add(s, f"{key.lower()}_gamma", f"{name} 择时 γ", res.gamma, RATIO, "TM：x² 系数" if key == "TM" else "HM：max(x, 0) 系数")
        add(s, f"{key.lower()}_gamma_t", f"{name} γ t 值", res.gamma_t, RATIO, _se_label(res))

    # ------------------------------ 持续监控 ------------------------------
    monitoring = None
    if monitor_targets is not None:
        if b is None:
            raise ValueError("持续监控需要基准收益")
        try:
            mu = float(monitor_targets["target_active_return"])
            te_target = float(monitor_targets["target_te"])
            window = int(monitor_targets["window"])
        except KeyError as exc:
            raise ValueError("monitor_targets 须包含 target_active_return、target_te 与 window") from exc
        a = p - b
        if len(a) >= window:
            te_now = float(monitor.realized_tracking_error(a, window, k).iloc[-1])
            active_sum = float(a.iloc[-window:].sum())
        else:
            te_now = active_sum = float("nan")
            notes.append(f"样本 {len(a)} 期短于监控窗口 {window} 期，风险倍数与 z 值不适用。")
        rm = monitor.risk_multiple(te_now, te_target)
        z = monitor.z_score(active_sum, window, mu, te_target, k)
        status = monitor.classify(rm, z)
        monitoring = {"window": window, "target_active_return": mu, "target_te": te_target, "realized_te": te_now,
                      "risk_multiple": rm, "z": z, "status": status}
        s = "持续监控"
        add(s, "monitor_window", "监控窗口", window, COUNT, "期")
        add(s, "realized_te", "实现 TE（最近窗口）", te_now, PCT, "年化，样本标准差 × √K")
        add(s, "risk_multiple", "风险倍数", rm, RATIO, f"实现 TE / 目标 TE {te_target:.2%}")
        add(s, "z_value", "z 值", z, RATIO, f"(Σa − μ·h/K) / (TE·√(h/K))，目标主动收益 {mu:.2%}")
        add(s, "monitor_status", "分区", {"green": "Green", "yellow": "Yellow", "red": "Red"}[status], TEXT,
            "阈值为正文演示值，须按策略校准")

    # ------------------------------ 口径 ------------------------------
    freq_name = _FREQ_NAMES.get(k, f"每年 {k} 期")
    scope = {
        "样本起": f"{df.index[0]:%Y-%m-%d}" if len(df) else "—",
        "样本止": f"{df.index[-1]:%Y-%m-%d}" if len(df) else "—",
        "期数": f"{len(df)}",
        "样本对齐": alignment_text,
        "频率 K": f"{freq_name}，K = {k}",
        "组合": labels.get("portfolio", "组合"),
        "基准": labels.get("benchmark", "基准" if b is not None else "未提供"),
    }
    benchmark_type = labels.get("benchmark_return_type") if b is not None else None
    if b is not None:
        type_text = benchmark_type or "未说明"
        if needs_price_caveat(benchmark_type):
            type_text += f"（{PRICE_INDEX_CAVEAT}）"
            notes.append(f"基准收益类型为{benchmark_type}：{PRICE_INDEX_CAVEAT}；宜改用全收益指数复核。")
        scope["基准收益类型"] = type_text
    if b is not None and "benchmark_source" in labels:
        scope["基准数据源"] = labels["benchmark_source"]
    if "benchmark_note" in labels:
        scope["基准说明"] = labels["benchmark_note"]
    scope["市场代理"] = labels.get("market", "市场") if market is not None else ("以基准代替" if b is not None else "未提供")
    scope["无风险收益"] = _rf_description(risk_free, labels)
    scope["费用口径"] = labels.get("fees", DEFAULT_FEE_BASIS)
    scope["比率口径"] = "同频算术均值与样本标准差（n−1），乘以 √K 年化；累计与年化收益为几何口径"
    if market_proxy:
        notes.append("未提供市场收益，Treynor、CAPM 与择时回归以基准代替市场。")

    # ------------------------------ 未完成的检验 ------------------------------
    not_done = [
        ("风格分析（第五部分第 1 节）", "未做", "fundeval 尚未实现风格分析模块，且需风格指数收益"),
        ("多因子分解（第五部分第 3 节）", "未做", "尚未实现，需因子收益数据（French 等数据源待加入）"),
        ("Brinson 归因（第五部分第 2 节）", "未做", "需要持仓与行业权重；持仓齐备后可用 fundeval.attribution.brinson"),
        ("样本外检验（第四部分第 3 节）", "未做", "报告只做全样本估计；样本外须事前规定切分点（alpha.rolling.split_in_out_of_sample）"),
        ("扣费后净 Alpha 重估（第六部分）", "未做", f"当前为{scope['费用口径']}；毛费用拆分与交易成本需费率数据，costs 模块待实现"),
        ("交易容量（第六部分）", "未做", "需要持仓、成交量与冲击成本数据"),
        ("资金加权收益 MWR（第二部分）", "未做", "需要申购赎回现金流"),
    ]
    if b is None:
        not_done.append(("相对基准指标（TE、IR、M²）", "未做", "未提供基准收益"))
    if m is None:
        not_done.append(("CAPM 回归与择时检验", "未做", "未提供基准或市场收益"))
    elif regression_error is not None:
        not_done.append(("CAPM 回归与择时检验", "未做", f"样本不足：{regression_error}"))
    if monitor_targets is None:
        not_done.append(("持续监控分区（第七部分）", "未做", "未给出目标主动收益、目标 TE 与窗口"))

    data = df.copy()
    if b is not None:
        data["active"] = p - b
    data["wealth"] = ret.wealth_index(p)

    metrics_df = pd.DataFrame([m_.__dict__ for m_ in metrics]).set_index("key")
    return EvaluationReport(
        scope=scope,
        metrics=metrics_df,
        capm=capm,
        timing=timing,
        monitoring=monitoring,
        quality=quality,
        not_done=pd.DataFrame(not_done, columns=["检验", "状态", "原因"]),
        data=data,
        periods_per_year=k,
        labels=labels,
        notes=notes + extra_notes,
        benchmark_return_type=benchmark_type,
    )


# ---------------------------------------------------------------------------
# 第十部分：结论
# ---------------------------------------------------------------------------


def _pct(x, digits: int = 2) -> str:
    return format_value(x, PCT, digits)


_THRESHOLD_NOTE = "|t| ≥ 1.96 只是参考，实际阈值须结合自由度与多重比较调整"


def _significance(t: float, kind: str = "alpha") -> str:
    """按 t 值的大小与方向给出措辞。``kind`` 为 "alpha"（截距）或 "gamma"（择时 γ）。

    - 显著为正：在大样本 5% 双侧口径下提供初步统计支持
    - 显著为负：Alpha 提示在所用模型下持续落后；γ 表现为负向凸性，不支持择时能力
    - 不显著：未显著偏离零
    """
    if not np.isfinite(t):
        return "无法判断显著性"
    if abs(t) < T_THRESHOLD:
        return "未显著偏离零（|t| < 1.96）"
    if t > 0:
        return f"在大样本 5% 双侧口径下提供初步统计支持（{_THRESHOLD_NOTE}）"
    if kind == "gamma":
        return (
            "显著为负，表现为负向凸性（上涨时市场敞口相对较低或下跌时相对较高），不支持择时能力"
            f"（{_THRESHOLD_NOTE}）"
        )
    return f"显著为负，提示在所用模型下持续落后（{_THRESHOLD_NOTE}）"


def _m2_gap(diff: float) -> str:
    if not np.isfinite(diff):
        return "M² 差额不适用"
    word = "高" if diff >= 0 else "低"
    return f"较基准年化算术均值{word} {format_value(abs(diff), PP)}"


def _price_caveat_sentence(r: EvaluationReport) -> str:
    return (
        f"基准收益类型为“{r.benchmark_return_type}”：{PRICE_INDEX_CAVEAT}，"
        "上述超额收益与 Alpha 不能直接解读为超额能力，宜改用全收益指数复核。"
    )


def conclusion(report: EvaluationReport) -> str:
    """按正文第十部分写出三段结论：观察到的表现、可以支持的解释、需要进一步验证的判断。

    措辞规则：样本少于 36 个月（或等价期数）时不作“管理能力”判断，只写“观察到模型
    未解释的收益”；|t| ≥ 1.96 且为正时写“在大样本 5% 双侧口径下提供初步统计支持”并注明
    阈值须结合自由度与多重比较调整，显著为负时按系数写明含义（见 _significance），否则写
    “未显著偏离零”；基准含价格指数或类型未知时写明价格指数的限定语；所有数值带单位与口径。
    """
    r = report
    has = r.metrics.index.__contains__
    k = r.periods_per_year
    freq = _FREQ_NAMES.get(k, f"K = {k}")

    # 一、观察到的表现
    obs = [
        f"样本为 {r.scope['样本起']} 至 {r.scope['样本止']} 的 {r.n} 期{freq}数据（约 {r.years:.1f} 年，{r.scope['费用口径']}）。",
        f"组合累计收益 {_pct(r.metric('cumulative_return'))}（几何），年化收益 {_pct(r.metric('annualized_return'))}（几何年化）",
    ]
    if has("benchmark_cumulative_return"):
        obs[-1] += (
            f"；基准累计收益 {_pct(r.metric('benchmark_cumulative_return'))}，累计收益差额 "
            f"{format_value(r.metric('cumulative_difference'), PP)}，几何相对收益 {_pct(r.metric('relative_return'))}。"
        )
    else:
        obs[-1] += "。"
    risk_text = (
        f"年化波动率 {_pct(r.metric('volatility'))}，最大回撤 {_pct(r.metric('max_drawdown'))}"
    )
    if r.metric("max_drawdown_trough") is not None:
        rec = r.metric("max_drawdown_recovery")
        rec_text = "尚未修复" if rec == "未修复" else f"于 {_fmt_date(rec)} 修复"
        risk_text += f"（{_fmt_date(r.metric('max_drawdown_peak'))} 至 {_fmt_date(r.metric('max_drawdown_trough'))}，{rec_text}）"
    risk_text += f"；Sharpe 比率 {format_value(r.metric('sharpe'), RATIO)}（年化，算术均值口径）"
    if has("information_ratio"):
        risk_text += (
            f"，信息比率 {format_value(r.metric('information_ratio'), RATIO)}，跟踪误差 {_pct(r.metric('tracking_error'))}（年化），"
            f"M² {_pct(r.metric('m_squared'))}（年化算术），{_m2_gap(r.metric('m_squared_difference'))}"
        )
    risk_text += "。"
    obs.append(risk_text)

    # 二、可以支持的解释
    sup: list[str] = []
    if r.capm is not None:
        alpha, t = r.capm.alpha, r.capm.alpha_t
        alpha_text = (
            f"CAPM 回归的 Alpha 为每期 {_pct(alpha, 3)}（算术年化 {_pct(r.metric('alpha_annualized'))}），"
            f"t = {t:.2f}，p = {r.capm.alpha_p:.3f}，n = {r.capm.n}，{_se_label(r.capm)}，{_significance(t)}。"
        )
        sup.append(alpha_text)
        if r.short_sample:
            if alpha > 0:
                sup.append(
                    f"样本较短（{r.n} 期，不足 {SHORT_SAMPLE_YEARS * 12} 个月），只能说观察到模型未解释的收益，"
                    "尚不足以据此评价经理的管理能力。"
                )
            else:
                sup.append(f"样本较短（{r.n} 期，不足 {SHORT_SAMPLE_YEARS * 12} 个月），且未观察到正的模型未解释收益。")
        elif alpha > 0 and abs(t) >= T_THRESHOLD:
            sup.append("在所用模型下存在正截距的统计证据，但尚不足以单独证明管理能力，需结合风格、因子与样本外检验。")
        elif alpha > 0:
            sup.append("观察到模型未解释的收益，但统计上未显著偏离零。")
        else:
            sup.append("未观察到正的模型未解释收益。")
        if r.price_index_caveat:
            sup.append(_price_caveat_sentence(r))
        timing_text = [
            f"{name} 择时 γ = {res.gamma:.3f}（t = {res.gamma_t:.2f}，{_se_label(res)}），{_significance(res.gamma_t, 'gamma')}"
            for key, name in (("TM", "Treynor–Mazuy"), ("HM", "Henriksson–Merton"))
            if (res := r.timing.get(key)) is not None
        ]
        if timing_text:
            reminder = ""
            if any(res.gamma_t > -T_THRESHOLD for res in r.timing.values() if res is not None):
                reminder = "γ 显著为正也可能来自期权类或动态风险控制等非线性策略，不能单凭 γ 认定择时能力。"
            sup.append("；".join(timing_text) + "。" + reminder)
    else:
        sup.append("未做 CAPM 回归与择时检验（缺少基准或样本不足），无法区分收益来源。")
        if r.price_index_caveat:
            sup.append(_price_caveat_sentence(r))
        if r.short_sample:
            sup.append(f"样本较短（{r.n} 期，不足 {SHORT_SAMPLE_YEARS * 12} 个月），上述表现不足以支持能力判断。")
    if has("sortino"):
        sup.append(
            f"下行风险方面，Sortino 比率 {format_value(r.metric('sortino'), RATIO)}（年化），"
            f"Calmar 比率 {format_value(r.metric('calmar'), RATIO)}（几何年化收益 / 最大回撤）。"
        )

    # 三、需要进一步验证的判断
    ver: list[str] = []
    todo = [row["检验"] for _, row in r.not_done.iterrows()]
    if todo:
        ver.append("以下检验未做：" + "；".join(todo) + "。收益是否来自风格暴露、因子溢价或费用前后差异，需补做后再下结论。")
    if has("var"):
        ver.append(
            f"历史 VaR {_pct(r.metric('var'))}、ES {_pct(r.metric('es'))}（{r.metrics.loc['var', 'label'].split('（')[-1].rstrip('）')}，单期损失）"
            + ("，尾部观测过少，仅供参考。" if any("尾部观测" in n_ for n_ in r.notes) else "。")
        )
    if r.monitoring is not None:
        status = r.metrics.loc["monitor_status", "value"]
        ver.append(
            f"持续监控：风险倍数 {format_value(r.monitoring['risk_multiple'], RATIO)}，z 值 {format_value(r.monitoring['z'], RATIO)}，"
            f"当前分区 {status}（阈值为演示值，须按策略校准）。"
        )
    issues = r.quality.issues()
    if issues:
        ver.append(f"数据质量有 {len(issues)} 项需复核（见数据质量报告），结论须在复核后确认。")
    else:
        ver.append("数据质量检查未发现缺失、异常收益或疑似停牌。")
    if r.short_sample:
        ver.append("样本较短，应在更长样本与样本外区间重复检验，结果才可能稳定。")

    return "\n\n".join(
        [
            "观察到的表现：" + "".join(obs),
            "可以支持的解释：" + "".join(sup),
            "需要进一步验证的判断：" + "".join(ver),
        ]
    )
