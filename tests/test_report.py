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
