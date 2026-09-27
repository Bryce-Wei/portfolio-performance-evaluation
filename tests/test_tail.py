"""第八部分：下行风险与尾部风险。"""

import math

import numpy as np
import pandas as pd
import pytest

from fundeval import returns, risk, tail

K = 12


def teaching_losses():
    """正文独立教学例：100 个等权损失情景，第 95 个为 2.0%，最差 5 个为 2.2%～6.0%。"""
    body = np.linspace(-0.03, 0.019, 94)  # 前 94 个损失均小于 2.0%，具体取值不影响结果
    worst = [0.020, 0.022, 0.025, 0.030, 0.040, 0.060]
    losses = np.concatenate([body, worst])
    rng = np.random.default_rng(0)
    return rng.permutation(losses)  # 打乱顺序，验证函数自行排序


def test_teaching_example_var_and_es():
    r = -teaching_losses()  # 收益 = -损失
    var = tail.historical_var(r, 0.95)
    es = tail.historical_es(r, 0.95)
    assert var == pytest.approx(0.020, abs=1e-12)
    assert es == pytest.approx(0.0354, abs=1e-12)
    # 本金 1,000 万元对应 20 万元与 35.4 万元
    assert var * 10_000_000 == pytest.approx(200_000, abs=1e-4)
    assert es * 10_000_000 == pytest.approx(354_000, abs=1e-4)


def test_es_fractional_tail_weights_last_observation():
    # n = 10、c = 0.75：尾部比例 2.5，最差两个全额计入，第三个计权重 0.5，分母为 2.5
    losses = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.20])
    es = tail.historical_es(-losses, 0.75)
    assert es == pytest.approx((0.20 + 0.10 + 0.5 * 0.08) / 2.5)
    # 最近秩法：⌈0.75 × 10⌉ = 第 8 个
    assert tail.historical_var(-losses, 0.75) == pytest.approx(0.08)


def test_es_with_tail_smaller_than_one_observation_is_worst_loss():
    losses = np.linspace(0.0, 0.05, 50)
    assert tail.historical_es(-losses, 0.99) == pytest.approx(0.05)


def test_es_not_below_var():
    r = -teaching_losses()
    for c in (0.90, 0.95, 0.975, 0.99):
        assert tail.historical_es(r, c) >= tail.historical_var(r, c)


def test_invalid_confidence():
    with pytest.raises(ValueError):
        tail.historical_var([0.01, -0.02], 1.0)


def test_downside_deviation_uses_full_sample_denominator():
    r = [0.02, -0.01, 0.03, -0.03]
    # 仅 -1% 与 -3% 低于 MAR = 0，分母仍为 4
    assert tail.downside_deviation(r, 0.0) == pytest.approx(math.sqrt((0.01**2 + 0.03**2) / 4))


def test_downside_deviation_accepts_mar_series():
    idx = pd.date_range("2025-01-31", periods=3, freq="ME")
    r = pd.Series([0.01, 0.00, 0.02], index=idx)
    mar = pd.Series([0.02, 0.00, 0.01], index=idx)
    assert tail.downside_deviation(r, mar) == pytest.approx(math.sqrt(0.01**2 / 3))


def test_sortino_without_downside_is_not_applicable():
    assert math.isnan(tail.sortino_ratio([0.01, 0.02, 0.03], 0.0))


def test_calmar_without_drawdown_is_not_applicable():
    assert math.isnan(tail.calmar_ratio([0.01, 0.02, 0.03]))


def test_sortino_worked_example(worked_example):
    # 第八部分未为演示数据指定 MAR，这里取演示数据的月度无风险收益 0.15% 作为 MAR，K = 12。
    # 1) r - MAR 之和 = 10.00% - 12 × 0.15% = 8.20%，均值 = 8.20% / 12 = 0.683333%
    # 2) 低于 MAR 的偏差：2 月 -1.15%、4 月 -2.15%、8 月 -3.15%、11 月 -0.65%，其余计 0
    #    平方和 = 1.3225 + 4.6225 + 9.9225 + 0.4225 = 16.29（%²）= 0.001629
    # 3) DD = sqrt(0.001629 / 12) = 1.165118%
    # 4) Sortino = 0.683333% / 1.165118% × √12 = 0.586492 × 3.464102 = 2.0317
    p, rf = worked_example.portfolio, worked_example.risk_free
    assert tail.downside_deviation(p, rf) == pytest.approx(0.01165118, abs=0.5e-8)
    assert tail.sortino_ratio(p, rf, K) == pytest.approx(2.0317, abs=0.5e-4)


def test_calmar_worked_example(worked_example):
    # 12 个完整月，年化几何收益 = 累计收益 10.2058%；
    # 最大回撤：7 月末财富 1.065748 为峰值，8 月 -3.00% 后回撤恰为 3.00%
    # Calmar = 10.2058% / 3.00% = 3.4019
    p = worked_example.portfolio
    assert returns.annualized_return(p, K) == pytest.approx(0.102058, abs=0.5e-6)
    assert risk.max_drawdown(p) == pytest.approx(0.03, abs=1e-12)
    assert tail.calmar_ratio(p, K) == pytest.approx(3.4019, abs=0.5e-4)


def test_downside_and_sortino_drop_nan_like_var_es():
    idx = pd.date_range("2025-01-31", periods=5, freq="ME")
    r = pd.Series([0.02, np.nan, -0.01, 0.03, -0.03], index=idx)
    clean = r.dropna()
    # 分母 n 为去除 NaN 后的 4 期，与 VaR、ES 的缺失值处理一致
    assert tail.downside_deviation(r, 0.0) == pytest.approx(math.sqrt((0.01**2 + 0.03**2) / 4))
    assert tail.downside_deviation(r, 0.0) == pytest.approx(tail.downside_deviation(clean, 0.0))
    assert tail.sortino_ratio(r, 0.0, K) == pytest.approx(tail.sortino_ratio(clean, 0.0, K))
    assert not math.isnan(tail.sortino_ratio(r, 0.0, K))
    assert tail.historical_var(r, 0.75) == pytest.approx(tail.historical_var(clean, 0.75))


def test_mar_series_aligns_to_index_after_dropping_nan():
    idx = pd.date_range("2025-01-31", periods=4, freq="ME")
    r = pd.Series([0.01, np.nan, 0.00, 0.02], index=idx)
    # MAR 序列缺少收益为 NaN 的那一期，也不影响计算；顺序打乱后按索引对齐
    mar = pd.Series([0.01, 0.02, 0.02], index=idx[[3, 0, 2]])
    expected = math.sqrt((0.01**2 + 0.02**2) / 3)  # 1 月 -1%、3 月 -2%、4 月 +1%
    assert tail.downside_deviation(r, mar) == pytest.approx(expected)
    mean = (-0.01 - 0.02 + 0.01) / 3
    assert tail.sortino_ratio(r, mar, K) == pytest.approx(mean / expected * math.sqrt(K))


def test_mar_array_pairs_by_position_before_dropping_nan():
    r = [0.01, np.nan, 0.00, 0.02]
    mar = [0.02, 0.50, 0.02, 0.01]  # 第 2 期随收益一起去除
    assert tail.downside_deviation(r, mar) == pytest.approx(math.sqrt((0.01**2 + 0.02**2) / 3))


def test_calmar_returns_nan_on_missing_instead_of_skipping(worked_example):
    # Calmar 不去除缺失值：跳过缺失月份相当于把该期当作零收益，年数 T 也会按 11 期算错
    p = worked_example.portfolio.copy()
    p.iloc[4] = np.nan
    assert math.isnan(tail.calmar_ratio(p, K))
    skipped = tail.calmar_ratio(p.dropna(), K)
    assert np.isfinite(skipped) and skipped != pytest.approx(3.4019, abs=0.5e-4)
    # 其他尾部函数按模块约定先去除 NaN
    assert tail.historical_var(p, 0.95) == tail.historical_var(p.dropna(), 0.95)
    assert tail.sortino_ratio(p, 0.0015, K) == pytest.approx(tail.sortino_ratio(p.dropna(), 0.0015, K))
