"""第五部分第 4 节：Campisi 固定收益归因。"""

import pandas as pd
import pytest

from fundeval.attribution.campisi import CAVEAT, campisi, campisi_active

BP = 1e-4


def approx(x):
    return pytest.approx(x, abs=1e-12)


# 正文演示：基金收益 1.20%，基准 0.80%，主动收益 0.40 个百分点，
# 拆为收入 0.05、利率 0.20、利差 0.05、剩余 0.10（单位均为百分点）。
FUND = dict(income=0.0060, duration=5.0, delta_yield=-0.0010, spread_duration=3.0, delta_spread=-0.0005, total_return=0.0120)
BENCH = dict(income=0.0055, duration=3.0, delta_yield=-0.0010, spread_duration=1.0, delta_spread=-0.0010, total_return=0.0080)


def test_worked_example_active_split():
    act = campisi_active(FUND, BENCH)
    assert act.fund.total_return == approx(0.0120) and act.benchmark.total_return == approx(0.0080)
    assert act.active_return == approx(0.0040)
    assert act.active["收入"] == approx(0.0005)
    assert act.active["利率"] == approx(0.0020)
    assert act.active["利差"] == approx(0.0005)
    assert act.active["剩余"] == approx(0.0010)
    assert act.active.sum() == approx(act.active_return)
    tbl = act.table().set_index("项目")
    assert list(tbl.columns) == ["基金", "基准", "主动"]
    assert tbl.loc["总收益", "主动"] == approx(0.0040) and tbl.loc["利率", "基金"] == approx(0.0050)


def test_single_portfolio_components():
    r = campisi(**FUND)
    assert r.income == approx(0.0060)
    assert r.rate == approx(0.0050)  # −5 × (−10 bp)
    assert r.spread == approx(0.0015)  # −3 × (−5 bp)
    assert r.specific == approx(0.0120 - 0.0060 - 0.0050 - 0.0015)
    assert r.explained + r.specific == approx(r.total_return)
    assert list(r.table()["项目"]) == ["收入", "利率", "利差", "剩余", "总收益"]


def test_convexity_term():
    r = campisi(0.004, 6.0, 0.0050, convexity=50.0)
    assert r.rate == approx(-6 * 0.005 + 0.5 * 50 * 0.005**2)
    assert r.specific is None and r.total_return is None
    assert pd.isna(r.components["剩余"])


def test_key_rate_duration_version():
    krd = pd.Series({"2Y": 0.8, "5Y": 1.5, "10Y": 2.2})
    dy = pd.Series({"10Y": 20 * BP, "2Y": -10 * BP, "5Y": 5 * BP})  # 陡峭化，顺序不同也按期限对齐
    r = campisi(0.003, krd, dy, total_return=0.001)
    expected = -(0.8 * -10 * BP + 1.5 * 5 * BP + 2.2 * 20 * BP)
    assert r.rate == approx(expected)
    assert list(r.rate_by_tenor.index) == ["2Y", "5Y", "10Y"]
    assert r.rate_by_tenor["2Y"] == approx(0.8 * 10 * BP)
    conv = pd.Series({"2Y": 5.0, "5Y": 20.0, "10Y": 60.0})
    rc = campisi(0.003, krd, dy, convexity=conv)
    assert rc.rate == approx(expected + 0.5 * (5 * (10 * BP) ** 2 + 20 * (5 * BP) ** 2 + 60 * (20 * BP) ** 2))
    # 平行移动时 KRD 之和等于久期
    par = campisi(0.003, krd, pd.Series(10 * BP, index=krd.index))
    assert par.rate == approx(campisi(0.003, krd.sum(), 10 * BP).rate)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(income=0.005, duration=5, delta_yield=10), "疑似以百分数或基点输入"),  # 10 bp 误写成 10
        (dict(income=0.005, duration=5, delta_yield=0.25), "疑似以百分数或基点输入"),  # 0.25% 误写成 0.25
        (dict(income=0.005, duration=5, delta_yield=0.001, delta_spread=5), "delta_spread"),
        (dict(income=0.6, duration=5, delta_yield=0.001), "income"),  # 0.60% 误写成 0.6
        (dict(income=0.006, duration=5, delta_yield=0.001, total_return=1.2), "total_return"),
        (dict(income=float("nan"), duration=5, delta_yield=0.001), "缺失"),
        (dict(income=0.005, duration=None, delta_yield=0.001), "须为数值"),
    ],
)
def test_unit_and_missing_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        campisi(**kwargs)


def test_key_rate_input_errors():
    krd = pd.Series({"2Y": 0.8, "5Y": 1.5})
    with pytest.raises(ValueError, match="关键期限不一致"):
        campisi(0.003, krd, pd.Series({"2Y": 0.001, "10Y": 0.001}))
    with pytest.raises(ValueError, match="同为按期限索引"):
        campisi(0.003, krd, 0.001)
    with pytest.raises(ValueError, match="有缺失的期限"):
        campisi(0.003, krd, pd.Series({"2Y": 0.001, "5Y": None}))
    with pytest.raises(ValueError, match=r"delta_yield\[5Y\]"):
        campisi(0.003, krd, pd.Series({"2Y": 0.001, "5Y": 15.0}))
    with pytest.raises(ValueError, match="凸性须按期限给出"):
        campisi(0.003, krd, pd.Series({"2Y": 0.001, "5Y": 0.001}), convexity=30.0)


def test_active_requires_totals_and_same_method():
    no_total = {k: v for k, v in FUND.items() if k != "total_return"}
    with pytest.raises(ValueError, match="无法与主动收益对账"):
        campisi_active(no_total, BENCH)
    krd = campisi(0.005, pd.Series({"5Y": 4.0}), pd.Series({"5Y": 0.001}), total_return=0.001)
    with pytest.raises(ValueError, match="同一方法"):
        campisi_active(krd, BENCH)
    with pytest.raises(TypeError):
        campisi_active([1, 2], BENCH)


def test_caveat_in_docstring():
    for text in ("平行移动近似不适合所有债券", "关键期限久期", "有效久期", "不能全部视为选券能力"):
        assert text in campisi.__doc__ and text in CAVEAT
