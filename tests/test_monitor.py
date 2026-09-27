"""第七部分：持续风险监控与预警。"""

import math

import numpy as np
import pandas as pd
import pytest

from fundeval import monitor

K = 252


def test_demo_budget_risk_and_z():
    # 正文演示：目标年化主动收益 3%、目标年化 TE 4%，K = 252、h = 21
    h = 21
    assert monitor.expected_active_return(0.03, h, K) == pytest.approx(0.0025, abs=1e-12)
    assert monitor.target_risk(0.04, h, K) == pytest.approx(0.01155, abs=0.5e-5)
    # 月内算术主动收益合计 -2.00%：z = (-2.00% - 0.25%) / 1.155% ≈ -1.95
    z = monitor.z_score(-0.02, h, 0.03, 0.04, K)
    assert z == pytest.approx(-1.95, abs=0.005)
    assert monitor.classify(1.0, z) == "green"  # |z| < 2，风险倍数正常时仍为 Green


def test_z_score_zero_target_is_not_applicable():
    assert math.isnan(monitor.z_score(-0.02, 21, 0.03, 0.0, K))


def test_realized_te_matches_sample_std_annualized():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(0, 0.0025, 120), index=pd.bdate_range("2025-01-01", periods=120))
    te = monitor.realized_tracking_error(a, 20, K)
    assert te.iloc[:19].isna().all()
    assert te.iloc[-1] == pytest.approx(a.iloc[-20:].std(ddof=1) * math.sqrt(K))
    both = monitor.realized_tracking_error(a, (20, 60), K)
    assert list(both.columns) == ["te_20", "te_60"]
    assert both["te_60"].iloc[:59].isna().all()
    assert both["te_60"].iloc[-1] == pytest.approx(a.iloc[-60:].std(ddof=1) * math.sqrt(K))
    assert both["te_20"].equals(te.rename("te_20"))


def test_realized_te_rejects_bad_window():
    with pytest.raises(ValueError):
        monitor.realized_tracking_error([0.01, 0.02, 0.03], 1)


def test_risk_multiple():
    assert monitor.risk_multiple(0.05, 0.04) == pytest.approx(1.25)
    assert math.isnan(monitor.risk_multiple(0.05, 0.0))
    s = pd.Series([0.04, 0.06, np.nan])
    assert monitor.risk_multiple(s, 0.04).tolist()[:2] == pytest.approx([1.0, 1.5])
    assert monitor.risk_multiple(s, 0.0).isna().all()


@pytest.mark.parametrize(
    "rm, z, expected",
    [
        (1.0, 0.0, "green"),
        (0.8, 1.99, "green"),  # 端点含在 Green 区间内
        (1.2, -1.99, "green"),
        (0.7, 0.0, "yellow"),  # 风险预算未使用
        (1.3, 0.0, "yellow"),
        (1.5, 0.0, "yellow"),  # 恰为 1.5 未超过 Red 阈值
        (1.0, 2.0, "yellow"),
        (1.0, -2.5, "yellow"),
        (1.51, 0.0, "red"),
        (1.0, 3.0, "red"),
        (1.0, -3.0, "red"),  # 双侧检查
        (0.5, 3.5, "red"),  # 两项不同时取较高等级
        (float("nan"), 0.0, "yellow"),  # 指标不可用时不能判为 Green
        (float("nan"), 3.2, "red"),
    ],
)
def test_classify_thresholds(rm, z, expected):
    assert monitor.classify(rm, z) == expected


def test_classify_custom_thresholds():
    assert monitor.classify(1.3, 0.0, green_band=(0.7, 1.4)) == "green"
    assert monitor.classify(1.3, 0.0, red_multiple=1.25) == "red"
    assert monitor.classify(1.0, 2.5, z_red=2.5) == "red"


def test_classify_series():
    idx = pd.date_range("2025-01-31", periods=3, freq="ME")
    rm = pd.Series([1.0, 1.6, 0.9], index=idx)
    z = pd.Series([0.5, 0.0, -2.2], index=idx)
    assert monitor.classify(rm, z).tolist() == ["green", "red", "yellow"]


def test_consecutive_red():
    statuses = ["red", "red", "yellow", "red", "red", "red", "red", "green"]
    flags = monitor.consecutive_red(statuses, 3)
    assert flags.tolist() == [False, False, False, False, False, True, True, False]


def test_worst_status():
    assert monitor.worst_status(["green", "red", "yellow"]) == "red"
