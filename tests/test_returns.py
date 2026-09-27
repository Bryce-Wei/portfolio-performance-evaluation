"""第二部分：TWR、年化、MWR/XIRR 的正文演示与边界情况。"""

import pandas as pd
import pytest

from fundeval import returns
from fundeval.etl import returns as etl_returns


def test_twr_demo_from_part_two():
    # 期初 100，第一年末 110；追加 50 后第二年末 144。子期收益 10% 与 -10%
    assert returns.twr([0.10, -0.10]) == pytest.approx(-0.01, abs=1e-12)
    assert returns.annualized_return([0.10, -0.10], periods_per_year=1) == pytest.approx(-0.0050, abs=0.5e-4)


def test_twr_from_account_values_strips_cashflow():
    dates = pd.to_datetime(["2021-01-01", "2022-01-01", "2023-01-01"])
    value = pd.Series([100.0, 160.0, 144.0], index=dates)  # 第一年末 110 + 追加 50
    flows = pd.Series([50.0], index=dates[1:2])
    r = etl_returns.value_to_returns(value, flows)
    assert r.tolist() == pytest.approx([0.10, -0.10])
    assert returns.twr(r) == pytest.approx(-0.01)


def test_mwr_demo_from_part_two():
    # -100 - 50/(1+i) + 144/(1+i)^2 = 0，得到约 -2.42%
    assert returns.irr([-100, -50, 144]) == pytest.approx(-0.0242, abs=0.5e-4)


def test_xirr_with_whole_year_intervals_matches_irr():
    flows = pd.Series(
        [-100.0, -50.0, 144.0], index=pd.to_datetime(["2021-01-01", "2022-01-01", "2023-01-01"])
    )
    assert returns.xirr(flows) == pytest.approx(returns.irr([-100, -50, 144]), abs=1e-10)
    assert returns.mwr(flows) == pytest.approx(-0.0242, abs=0.5e-4)


def test_xirr_accepts_pairs_and_merges_same_day():
    pairs = [("2021-01-01", -60.0), ("2021-01-01", -40.0), ("2022-01-01", 110.0)]
    assert returns.xirr(pairs) == pytest.approx(0.10, abs=1e-10)


def test_xirr_requires_both_signs():
    with pytest.raises(ValueError):
        returns.xirr(pd.Series([100.0, 50.0], index=pd.to_datetime(["2021-01-01", "2022-01-01"])))


def test_annualize_return_with_fractional_years():
    assert returns.annualize_return(0.21, 2) == pytest.approx(0.10)
    with pytest.raises(ValueError):
        returns.annualize_return(0.1, 0)


def test_misaligned_series_are_rejected():
    a = pd.Series([0.01, 0.02], index=pd.to_datetime(["2025-01-31", "2025-02-28"]))
    b = pd.Series([0.01, 0.02], index=pd.to_datetime(["2025-01-31", "2025-03-31"]))
    with pytest.raises(ValueError):
        returns.active_returns(a, b)
