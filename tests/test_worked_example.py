"""基准测试：正文第九部分“从原始数据到绩效结果”的汇总指标。

正文结果按四位小数显示，这里按显示精度的一半作为容差。
"""

import pytest

from fundeval import returns, risk

K = 12


def pct(x):
    """正文百分比显示值转小数，容差为最后一位的一半。"""
    return pytest.approx(x / 100, abs=0.5e-6)


def test_portfolio_cumulative_return(worked_example):
    assert returns.cumulative_return(worked_example.portfolio) == pct(10.2058)


def test_benchmark_cumulative_return(worked_example):
    assert returns.cumulative_return(worked_example.benchmark) == pct(6.7390)


def test_cumulative_difference(worked_example):
    d = returns.cumulative_difference(worked_example.portfolio, worked_example.benchmark)
    assert d == pct(3.4667)


def test_geometric_relative_return(worked_example):
    r = returns.relative_return(worked_example.portfolio, worked_example.benchmark)
    assert r == pct(3.2479)


def test_annualized_equals_cumulative_for_twelve_months(worked_example):
    # 12 个完整月使累计收益与该区间的年化几何收益相同
    p = worked_example.portfolio
    assert returns.annualized_return(p, K) == pytest.approx(returns.cumulative_return(p), abs=1e-12)


def test_volatility(worked_example):
    assert risk.volatility(worked_example.portfolio, K) == pct(7.2864)


def test_tracking_error(worked_example):
    te = risk.tracking_error(worked_example.portfolio, worked_example.benchmark, K)
    assert te == pct(1.4780)


def test_sharpe(worked_example):
    s = risk.sharpe_ratio(worked_example.portfolio, worked_example.risk_free, K)
    assert s == pytest.approx(1.1254, abs=0.5e-4)


def test_information_ratio(worked_example):
    ir = risk.information_ratio(worked_example.portfolio, worked_example.benchmark, K)
    assert ir == pytest.approx(2.2327, abs=0.5e-4)


def test_max_drawdown(worked_example):
    assert risk.max_drawdown(worked_example.portfolio) == pytest.approx(0.0300, abs=0.5e-4)


def test_max_drawdown_occurs_in_month_8_and_recovers_in_month_9(worked_example):
    dd = risk.drawdown(worked_example.portfolio)
    assert dd.idxmin() == worked_example.index[7]
    assert dd.iloc[8] == 0


def test_m_squared(worked_example):
    m2 = risk.m_squared(worked_example.portfolio, worked_example.benchmark, worked_example.risk_free, K)
    assert m2 == pct(8.4668)


def test_m_squared_difference(worked_example):
    d = risk.m_squared_difference(
        worked_example.portfolio, worked_example.benchmark, worked_example.risk_free, K
    )
    assert d == pct(1.7668)


def test_benchmark_arithmetic_annual_mean(worked_example):
    # M² 须与同口径基准均值 6.70% 比较
    assert worked_example.benchmark.mean() * K == pytest.approx(0.0670, abs=1e-12)


def test_wealth_index_column(worked_example):
    # 正文表中“组合财富指数”列，六位小数
    expected = [1.020000, 1.009800, 1.040094, 1.019292, 1.034582, 1.039754,
                1.065748, 1.033776, 1.075127, 1.085878, 1.080449, 1.102058]
    w = returns.wealth_index(worked_example.portfolio)
    assert w.tolist() == pytest.approx(expected, abs=0.5e-6)


def test_active_return_column(worked_example):
    expected = [0.50, 0.20, 0.50, -0.50, 0.50, 0.20, 0.50, -0.50, 1.00, 0.20, 0.20, 0.50]
    a = returns.active_returns(worked_example.portfolio, worked_example.benchmark)
    assert (a * 100).tolist() == pytest.approx(expected, abs=1e-9)


def test_summary_matches_individual_metrics(worked_example):
    s = risk.summary(worked_example.portfolio, worked_example.benchmark, worked_example.risk_free, K)
    assert s["cumulative_return"] == pct(10.2058)
    assert s["information_ratio"] == pytest.approx(2.2327, abs=0.5e-4)
    assert s["sharpe"] == pytest.approx(1.1254, abs=0.5e-4)
    assert s["max_drawdown"] == pytest.approx(0.03, abs=0.5e-4)
    assert s["m_squared"] == pct(8.4668)
