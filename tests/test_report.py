"""第九、十部分：evaluate 汇总指标、三段结论、Markdown 与 Excel 导出。"""

import numpy as np
import pandas as pd
import pytest

from fundeval import risk, tail
from fundeval.alpha import capm_regression
from fundeval.report import conclusion, evaluate, to_excel, to_markdown

K = 12


def pct(x):
    return pytest.approx(x / 100, abs=0.5e-6)


@pytest.fixture
def report(worked_example):
    return evaluate(
        worked_example,
        periods_per_year=K,
        monitor_targets={"target_active_return": 0.03, "target_te": 0.02, "window": 6},
        labels={"portfolio": "演示组合", "benchmark": "演示基准"},
    )


def test_evaluate_matches_worked_example_benchmarks(report):
    # 与 tests/test_worked_example.py 的基准值一致
    assert report.metric("cumulative_return") == pct(10.2058)
    assert report.metric("benchmark_cumulative_return") == pct(6.7390)
    assert report.metric("cumulative_difference") == pct(3.4667)
    assert report.metric("relative_return") == pct(3.2479)
    assert report.metric("volatility") == pct(7.2864)
    assert report.metric("tracking_error") == pct(1.4780)
    assert report.metric("information_ratio") == pytest.approx(2.2327, abs=0.5e-4)
    assert report.metric("sharpe") == pytest.approx(1.1254, abs=0.5e-4)
    assert report.metric("max_drawdown") == pytest.approx(0.0300, abs=0.5e-4)
    assert report.metric("m_squared") == pct(8.4668)
    assert report.metric("m_squared_difference") == pct(1.7668)
    # 第八部分：MAR 取无风险收益 0.15%/月
    assert report.metric("sortino") == pytest.approx(2.0317, abs=0.5e-4)
    assert report.metric("calmar") == pytest.approx(3.4019, abs=0.5e-4)


def test_drawdown_dates_and_recovery(report, worked_example):
    idx = worked_example.index
    assert report.metric("max_drawdown_peak") == idx[6]
    assert report.metric("max_drawdown_trough") == idx[7]
    assert report.metric("max_drawdown_recovery") == idx[8]
    assert report.metric("recovery_periods") == 1


def test_scope_and_proxy_notes(report):
    assert report.scope["样本起"] == "2025-01-31" and report.scope["期数"] == "12"
    assert report.scope["频率 K"] == "月度，K = 12"
    assert report.scope["费用口径"] == "费用后净值"
    assert report.scope["市场代理"] == "以基准代替"
    assert "以基准代替市场" in report.metrics.loc["treynor", "note"]
    assert report.short_sample


def test_tail_small_sample_note(report):
    assert "样本过小" in report.metrics.loc["var", "note"]
    assert any("尾部观测" in n for n in report.notes)


def test_alpha_section_matches_regression(report, worked_example):
    df = worked_example
    res = capm_regression(df.portfolio, df.benchmark, df.risk_free)
    assert report.metric("alpha") == pytest.approx(res.alpha)
    assert report.metric("alpha_t") == pytest.approx(res.alpha_t)
    assert report.metric("regression_n") == 12
    assert report.metric("hac") == "否"
    assert {"tm_gamma", "tm_gamma_t", "hm_gamma", "hm_gamma_t"} <= set(report.metrics.index)
    tbl = report.regression_table()
    assert set(tbl["模型"]) == {"CAPM", "Treynor–Mazuy", "Henriksson–Merton"}


def test_hac_is_reported(worked_example):
    rep = evaluate(worked_example, hac_lags=2)
    assert rep.metric("hac") == "是" and "HAC" in rep.metrics.loc["alpha_t", "note"]


def test_monitoring_section(report, worked_example):
    a = worked_example.portfolio - worked_example.benchmark
    te = a.iloc[-6:].std(ddof=1) * np.sqrt(K)
    assert report.monitoring["realized_te"] == pytest.approx(te)
    assert report.monitoring["risk_multiple"] == pytest.approx(te / 0.02)
    z = (a.iloc[-6:].sum() - 0.03 * 6 / K) / (0.02 * np.sqrt(6 / K))
    assert report.monitoring["z"] == pytest.approx(z)
    assert report.monitoring["status"] == "green"


def test_not_done_lists_required_items(report):
    items = " ".join(report.not_done["检验"])
    for word in ("风格分析", "多因子分解", "Brinson", "样本外检验", "扣费后净 Alpha", "交易容量"):
        assert word in items
    assert (report.not_done["状态"] == "未做").all()
    assert report.not_done["原因"].str.len().gt(0).all()


def test_conclusion_wording_for_short_sample(report):
    text = conclusion(report)
    paragraphs = text.split("\n\n")
    assert [p.split("：")[0] for p in paragraphs] == ["观察到的表现", "可以支持的解释", "需要进一步验证的判断"]
    assert "样本较短" in text
    assert "具备管理能力" not in text
    assert "观察到模型未解释的收益" in text
    # CAPM t = 3.20 ≥ 1.96
    assert "在大样本 5% 双侧口径下提供初步统计支持" in text and "自由度" in text and "多重比较" in text
    # 择时 γ 不显著
    assert "未显著偏离零" in text
    # 数值带单位与口径
    assert "10.21%（几何）" in text and "3.47 个百分点" in text and "年化" in text
    assert report.conclusion() == text


def test_conclusion_insignificant_alpha():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2015-01-31", periods=60, freq="ME")
    b = pd.Series(rng.normal(0.005, 0.04, 60), index=idx)
    p = b + rng.normal(0, 0.01, 60)
    rep = evaluate(p, b, 0.001)
    text = conclusion(rep)
    assert abs(rep.metric("alpha_t")) < 1.96
    assert "未显著偏离零" in text.split("\n\n")[1].split("。")[0]
    assert not rep.short_sample and "样本较短" not in text
    assert "具备管理能力" not in text


def test_evaluate_without_benchmark_marks_not_done():
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    p = pd.Series(np.linspace(-0.02, 0.03, 24), index=idx)
    rep = evaluate(p, risk_free=0.001)
    assert "information_ratio" not in rep.metrics.index and rep.capm is None
    assert "相对基准指标" in " ".join(rep.not_done["检验"])
    assert "样本较短" in conclusion(rep)


def test_evaluate_refuses_missing_values_and_reports_date_mismatch(worked_example):
    p = worked_example.portfolio.copy()
    p.iloc[3] = np.nan
    with pytest.raises(ValueError, match="缺失"):
        evaluate(p, worked_example.benchmark, worked_example.risk_free)
    b = worked_example.benchmark.drop(worked_example.index[2])
    rep = evaluate(worked_example.portfolio, b, 0.0015)
    assert rep.n == 11 and rep.quality.missing.loc["benchmark", "missing"] == 1


def test_evaluate_accepts_series_inputs_and_market(worked_example):
    df = worked_example
    rep = evaluate(df.portfolio, df.benchmark, df.risk_free, market=df.benchmark * 1.1, mar=0.0)
    assert rep.scope["市场代理"] == "市场"
    assert rep.metric("sortino") == pytest.approx(tail.sortino_ratio(df.portfolio, 0.0, K))
    assert rep.metric("treynor") == pytest.approx(risk.treynor_ratio(df.portfolio, df.benchmark * 1.1, df.risk_free, K))
    with pytest.raises(ValueError, match="risk_free"):
        evaluate(df, risk_free=0.001)


# ------------------------------ 导出 ------------------------------


def test_to_markdown(report, tmp_path):
    path = tmp_path / "report.md"
    text = to_markdown(report, path)
    assert path.read_text(encoding="utf-8") == text
    assert text.startswith("# 演示组合 绩效评价报告")
    for heading in ("## 口径", "## 收益表现", "## 风险效率", "## 下行与尾部", "## Alpha 质量", "## 持续监控",
                    "## 回归系数", "## 数据质量", "## 结论", "## 未完成的检验"):
        assert heading in text
    assert "| 累计收益（组合） | 10.2058% |" in text
    assert "| 信息比率 IR | 2.2327 |" in text
    assert "| M² | 8.4668% |" in text
    assert "具备管理能力" not in text


def test_to_excel_round_trip(report, tmp_path):
    pytest.importorskip("openpyxl")
    path = to_excel(report, tmp_path / "report.xlsx")
    sheets = pd.read_excel(path, sheet_name=None)
    assert {"口径", "指标", "回归", "期间收益", "数据质量"} <= set(sheets)
    scope = sheets["口径"].set_index("项目")["内容"]
    assert scope["样本起"] == "2025-01-31" and str(scope["期数"]) == "12"
    metrics = sheets["指标"].set_index("键")
    assert float(metrics.loc["cumulative_return", "数值"]) == pytest.approx(0.102058, abs=0.5e-6)
    assert float(metrics.loc["information_ratio", "数值"]) == pytest.approx(2.2327, abs=0.5e-4)
    assert metrics.loc["m_squared", "显示值"] == "8.4668%"
    reg = sheets["回归"]
    capm_alpha = reg[(reg["模型"] == "CAPM") & (reg["系数"] == "alpha")].iloc[0]
    assert capm_alpha["估计值"] == pytest.approx(report.capm.alpha)
    periods = sheets["期间收益"].set_index("日期")
    assert len(periods) == 12
    np.testing.assert_allclose(periods["portfolio"].to_numpy(), report.data["portfolio"].to_numpy())
    assert periods["wealth"].iloc[-1] == pytest.approx(1.102058, abs=0.5e-6)
    quality = sheets["数据质量"]
    assert "期数" in set(quality["项目"])


# ------------------------------ 对齐口径 ------------------------------


def test_scope_reports_alignment_counts_and_dropped_periods(worked_example):
    df = worked_example
    b = df.benchmark.drop(df.index[[2, 3, 9]])
    rep = evaluate(df.portfolio, b, 0.0015)
    assert rep.scope["样本对齐"] == "组合 12 期、基准 9 期、共同 9 期"
    dropped = [n for n in rep.notes if n.startswith("对齐时丢弃")]
    assert dropped == ["对齐时丢弃组合不在共同日期内的 3 期：2025-03-31 至 2025-04-30（2 期）；2025-10-31"]
    assert evaluate(df).scope["样本对齐"] == "组合 12 期、基准 12 期、共同 12 期"
    assert not any(n.startswith("对齐时丢弃") for n in evaluate(df).notes)


def test_evaluate_without_common_dates_raises(worked_example):
    df = worked_example
    shifted = df.benchmark.copy()
    shifted.index = shifted.index - pd.Timedelta(days=3)  # 月末交易日与日历月末对不上
    with pytest.raises(ValueError, match="没有共同日期"):
        evaluate(df.portfolio, shifted, 0.0015)


# ------------------------------ 显著性措辞区分方向 ------------------------------


def test_significance_wording_by_sign():
    from fundeval.report.summary import _significance

    assert _significance(2.5).startswith("在大样本 5% 双侧口径下提供初步统计支持")
    assert _significance(-2.5).startswith("显著为负，提示在所用模型下持续落后")
    assert _significance(2.5, "gamma").startswith("在大样本 5% 双侧口径下提供初步统计支持")
    assert _significance(-2.5, "gamma").startswith("显著为负，表现为负向凸性（上涨时市场敞口相对较低或下跌时相对较高），不支持择时能力")
    for kind in ("alpha", "gamma"):
        assert _significance(1.2, kind) == _significance(-1.2, kind) == "未显著偏离零（|t| < 1.96）"


def _convex_sample(gamma, alpha, n=60, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-01-31", periods=n, freq="ME")
    x = pd.Series(rng.normal(0.005, 0.045, n), index=idx)
    y = alpha + 0.9 * x + gamma * x**2 + rng.normal(0, 0.002, n)
    return y, x


def test_conclusion_negative_alpha_and_negative_convexity():
    p, b = _convex_sample(gamma=-8.0, alpha=-0.004)
    rep = evaluate(p, b)
    assert rep.metric("alpha_t") <= -1.96 and rep.timing["TM"].gamma_t <= -1.96 and rep.timing["HM"].gamma_t <= -1.96
    sup = conclusion(rep).split("\n\n")[1]
    assert "显著为负，提示在所用模型下持续落后" in sup
    assert sup.count("显著为负，表现为负向凸性") == 2 and "不支持择时能力" in sup
    assert "提供初步统计支持" not in sup and "γ 显著为正也可能来自" not in sup


def test_conclusion_positive_alpha_and_positive_convexity():
    p, b = _convex_sample(gamma=8.0, alpha=0.004)
    rep = evaluate(p, b)
    assert rep.metric("alpha_t") >= 1.96 and rep.timing["TM"].gamma_t >= 1.96
    sup = conclusion(rep).split("\n\n")[1]
    assert "提供初步统计支持" in sup and "γ 显著为正也可能来自期权类或动态风险控制等非线性策略" in sup
    assert "显著为负" not in sup


# ------------------------------ use_t 传到报告层 ------------------------------


def test_use_t_is_passed_to_regressions(worked_example):
    normal = evaluate(worked_example, hac_lags=2)
    t_dist = evaluate(worked_example, hac_lags=2, use_t=True)
    assert not normal.capm.use_t and t_dist.capm.use_t
    assert all(res.use_t for res in t_dist.timing.values())
    assert set(normal.regression_table()["标准误类型"]) == {"HAC（Newey–West，滞后 2，正态近似）"}
    assert set(t_dist.regression_table()["标准误类型"]) == {"HAC（Newey–West，滞后 2，t 分布）"}
    assert t_dist.metric("alpha_p") > normal.metric("alpha_p")  # t 分布尾部更厚，p 值更大
    assert evaluate(worked_example).capm.use_t  # 普通 OLS 默认 t 分布
