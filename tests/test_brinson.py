"""第五部分第 2 节：Brinson 单期归因（正文演示）与多期 Cariño 链接。"""

import warnings

import numpy as np
import pandas as pd
import pytest

from fundeval.attribution import brinson_multi_period, brinson_single

SECTORS = ["甲", "乙", "丙"]
WP = pd.Series([0.50, 0.30, 0.20], index=SECTORS)
WB = pd.Series([0.40, 0.40, 0.20], index=SECTORS)
RP = pd.Series([0.038, 0.010, -0.010], index=SECTORS)
RB = pd.Series([0.030, 0.010, -0.005], index=SECTORS)
PP = 1e-2  # 百分点


def test_bhb_article_example():
    res = brinson_single(WP, WB, RP, RB)
    assert res.method == "BHB"
    assert res.portfolio_return == pytest.approx(0.02, abs=1e-15)
    assert res.benchmark_return == pytest.approx(0.015, abs=1e-15)
    assert res.active_return == pytest.approx(0.005, abs=1e-15)
    expected = pd.DataFrame(
        {
            "allocation": [0.30, -0.10, 0.00],
            "selection": [0.32, 0.00, -0.10],
            "interaction": [0.08, 0.00, 0.00],
            "total": [0.70, -0.10, -0.10],
        },
        index=SECTORS,
    )
    np.testing.assert_allclose(res.effects.to_numpy() / PP, expected.to_numpy(), atol=1e-12)
    np.testing.assert_allclose(res.totals.to_numpy() / PP, [0.20, 0.22, 0.08, 0.50], atol=1e-12)
    table = res.table()
    assert list(table.index) == SECTORS + ["total"]
    assert res.reconciled and abs(res.residual) < 1e-15


def test_bf_article_example():
    res = brinson_single(WP, WB, RP, RB, method="BF")
    np.testing.assert_allclose(res.effects["allocation"].to_numpy() / PP, [0.15, 0.05, 0.00], atol=1e-12)
    # 完整权重下总配置与 BHB 一致，选择与交互不变
    assert res.totals["allocation"] / PP == pytest.approx(0.20, abs=1e-12)
    bhb = brinson_single(WP, WB, RP, RB)
    np.testing.assert_allclose(res.effects[["selection", "interaction"]], bhb.effects[["selection", "interaction"]])
    assert res.reconciled


def test_inputs_align_by_sector_name():
    res = brinson_single(WP, WB.iloc[::-1], RP.iloc[[1, 2, 0]], RB)
    assert res.totals["total"] == pytest.approx(0.005)
    assert res.effects.loc["甲", "selection"] == pytest.approx(0.0032)
    arr = brinson_single(WP.to_numpy(), WB.to_numpy(), RP.to_numpy(), RB.to_numpy())
    np.testing.assert_allclose(arr.effects.to_numpy(), res.effects.to_numpy())


def test_weights_must_sum_to_one_including_cash():
    with pytest.raises(ValueError, match="现金"):
        brinson_single([0.5, 0.3, 0.15], WB.to_numpy(), RP.to_numpy(), RB.to_numpy())
    # 容差可调
    res = brinson_single([0.5, 0.3, 0.199], WB.to_numpy(), RP.to_numpy(), RB.to_numpy(), tol=1e-2)
    assert res.reconciled  # BHB 按加权收益对账，恒成立
    with pytest.raises(ValueError):
        brinson_single(WP, WB, RP, RB, method="xyz")
    with pytest.raises(ValueError):
        brinson_single(WP, WB, RP.rename({"丙": "丁"}), RB)


def test_bf_residual_is_reported_when_weights_within_tolerance():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = brinson_single([0.5, 0.3, 0.199], WB.to_numpy(), RP.to_numpy(), RB.to_numpy(), "BF", tol=1e-2)
    assert not res.reconciled
    assert res.residual == pytest.approx(-res.benchmark_return * (-0.001))
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def multi_period_data(n=6, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2025-01", periods=n, freq="M")
    cols = ["股票", "债券", "现金"]

    def weights():
        w = rng.dirichlet([5, 3, 1], n)
        return pd.DataFrame(w, index=idx, columns=cols)

    rp = pd.DataFrame(rng.normal(0.01, 0.04, (n, 3)), index=idx, columns=cols)
    rb = pd.DataFrame(rng.normal(0.008, 0.035, (n, 3)), index=idx, columns=cols)
    return weights(), weights(), rp, rb


@pytest.mark.parametrize("method", ["BHB", "BF"])
def test_multi_period_carino_reconciles_cumulative_difference(method):
    wp, wb, rp, rb = multi_period_data()
    res = brinson_multi_period(wp, wb, rp, rb, method=method)
    r_p = (wp * rp).sum(axis=1)
    r_b = (wb * rb).sum(axis=1)
    cum_diff = (np.prod(1 + r_p) - 1) - (np.prod(1 + r_b) - 1)
    assert res.cumulative_active == pytest.approx(cum_diff, abs=1e-15)
    assert res.periods["total"].sum() == pytest.approx(cum_diff, abs=1e-14)
    assert res.totals["total"] == pytest.approx(cum_diff, abs=1e-14)
    assert res.by_sector["total"].sum() == pytest.approx(cum_diff, abs=1e-14)
    assert res.totals[["allocation", "selection", "interaction"]].sum() == pytest.approx(cum_diff, abs=1e-14)
    assert res.reconciled
    # 各期未调整贡献的算术和不等于累计差额，不能直接相加
    assert abs(res.unlinked["total"].sum() - cum_diff) > 1e-5
    assert res.periods.index.names == ["period", "sector"]
    assert len(res.periods) == 6 * 3


def test_multi_period_equal_returns_use_analytic_limit():
    wp, wb, rp, rb = multi_period_data(n=3)
    # 第 2 期组合与基准完全相同：R_p,t = R_b,t，系数取 1 / (1 + R)
    wp.iloc[1], rp.iloc[1] = wb.iloc[1], rb.iloc[1]
    res = brinson_multi_period(wp, wb, rp, rb)
    r_t = res.benchmark_returns.iloc[1]
    assert res.coefficients.iloc[1] == pytest.approx(1 / (1 + r_t))
    assert np.isfinite(res.coefficients).all() and res.reconciled
    # 整体组合与基准相同：累计系数同样取极限，全部贡献为零
    same = brinson_multi_period(wb, wb, rb, rb)
    assert same.k == pytest.approx(1 / (1 + same.cumulative_benchmark))
    assert same.totals["total"] == pytest.approx(0.0, abs=1e-15) and same.reconciled


def test_multi_period_single_period_equals_single():
    wp, wb, rp, rb = multi_period_data(n=1)
    multi = brinson_multi_period(wp, wb, rp, rb)
    single = brinson_single(wp.iloc[0], wb.iloc[0], rp.iloc[0], rb.iloc[0])
    np.testing.assert_allclose(multi.by_sector.to_numpy(), single.effects.to_numpy(), atol=1e-15)


def test_multi_period_checks_each_period_weights():
    wp, wb, rp, rb = multi_period_data(n=3)
    wp.iloc[2, 0] += 0.05
    with pytest.raises(ValueError, match="现金"):
        brinson_multi_period(wp, wb, rp, rb)
    with pytest.raises(ValueError):
        brinson_multi_period(wp.iloc[:2], wb, rp, rb)
