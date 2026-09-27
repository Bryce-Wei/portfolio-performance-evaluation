"""第三部分：口径与边界情况。"""

import math

import numpy as np
import pandas as pd
import pytest

from fundeval import risk


def test_drawdown_peak_includes_initial_value():
    # 首期即亏损：峰值应为初始值 1，而不是第一期财富
    assert risk.max_drawdown([-0.05, 0.01]) == pytest.approx(0.05)


def test_no_drawdown_when_always_rising():
    assert risk.max_drawdown([0.01, 0.02]) == 0


def test_sharpe_zero_std_is_not_applicable():
    assert math.isnan(risk.sharpe_ratio([0.01, 0.01, 0.01], 0.0015))


def test_information_ratio_zero_te_is_not_applicable():
    p = [0.02, 0.03, 0.01]
    b = [0.01, 0.02, 0.00]
    assert math.isnan(risk.information_ratio(p, b))


def test_sharpe_constant_rf_equals_volatility_denominator():
    # 无风险收益固定时，Sharpe 分母等于组合收益波动率
    r = pd.Series([0.02, -0.01, 0.03, 0.00])
    s = risk.sharpe_ratio(r, 0.001, 12)
    expected = (r.mean() - 0.001) * 12 / risk.volatility(r, 12)
    assert s == pytest.approx(expected)


def test_beta_treynor_and_jensen_on_exact_linear_relation():
    rng = np.random.default_rng(0)
    m = pd.Series(rng.normal(0.01, 0.04, 60))
    rf = 0.002
    p = rf + 0.001 + 1.2 * (m - rf)
    assert risk.beta(p, m, rf) == pytest.approx(1.2)
    assert risk.jensen_alpha(p, m, rf) == pytest.approx(0.001)
    assert risk.treynor_ratio(p, m, rf, 12) == pytest.approx(12 * (p - rf).mean() / 1.2)


def test_risk_free_series_must_cover_all_periods():
    idx = pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31"])
    p = pd.Series([0.01, -0.02, 0.03], index=idx)
    rf = pd.Series([0.001, 0.001], index=idx[:2])
    with pytest.raises(ValueError):
        risk.sharpe_ratio(p, rf)
