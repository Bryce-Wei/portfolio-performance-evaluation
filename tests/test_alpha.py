"""第四部分：Alpha 回归、滚动稳定性、样本内外切分与主动管理基本定律。"""

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from fundeval import risk
from fundeval.alpha import (
    capm_regression,
    expected_ir,
    factor_regression,
    rolling_information_ratio,
    rolling_regression,
    split_in_out_of_sample,
)


def synthetic(n=600, alpha=0.002, betas=(1.1, 0.4, -0.3), noise=0.004, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    f = pd.DataFrame(rng.normal(0.005, 0.04, (n, len(betas))), index=idx, columns=["MKT", "SMB", "HML"][: len(betas)])
    rf = pd.Series(0.001, index=idx)
    r = rf + alpha + f.to_numpy() @ np.array(betas) + rng.normal(0, noise, n)
    return pd.Series(r, index=idx, name="fund"), f, rf


def test_capm_matches_jensen_alpha_and_beta(worked_example):
    # OLS 恒等式：截距 = mean(e_p) - β mean(e_m)，斜率 = cov / var，与 risk 模块完全一致
    p, b, rf = worked_example.portfolio, worked_example.benchmark, worked_example.risk_free
    res = capm_regression(p, b, rf)
    assert res.alpha == pytest.approx(risk.jensen_alpha(p, b, rf), abs=1e-14)
    assert res.betas["market"] == pytest.approx(risk.beta(p, b, rf), abs=1e-12)
    assert res.n == 12
    assert res.cov_type == "nonrobust" and res.hac_lags is None
    assert res.annualized_alpha(12) == pytest.approx(res.alpha * 12)
    lo, hi = res.conf_int.loc["alpha"]
    assert lo < res.alpha < hi
    assert res.alpha_t == pytest.approx(res.alpha / res.alpha_se)


def test_factor_regression_recovers_known_coefficients():
    r, f, rf = synthetic()
    res = factor_regression(r, f, rf)
    assert res.alpha == pytest.approx(0.002, abs=3 * res.alpha_se)
    for name, true in zip(["MKT", "SMB", "HML"], [1.1, 0.4, -0.3]):
        assert res.betas[name] == pytest.approx(true, abs=0.02)
        lo, hi = res.conf_int.loc[name]
        assert lo < true < hi
    assert res.rsquared > 0.95
    assert list(res.table().columns) == ["coef", "se", "t", "p", "ci_lower", "ci_upper"]
    assert res.alpha_p < 0.01


def test_hac_matches_statsmodels_directly():
    r, f, rf = synthetic(n=240, seed=3)
    res = factor_regression(r, f, rf, hac_lags=4)
    direct = sm.OLS(r - rf, sm.add_constant(f)).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    assert res.cov_type == "HAC" and res.hac_lags == 4
    np.testing.assert_allclose(res.params.to_numpy(), direct.params.to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(res.bse.to_numpy(), direct.bse.to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(res.tvalues.to_numpy(), direct.tvalues.to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(res.conf_int.to_numpy(), direct.conf_int().to_numpy(), rtol=1e-12)
    ols = factor_regression(r, f, rf)
    assert not np.allclose(ols.bse.to_numpy(), res.bse.to_numpy())


def test_regression_input_checks():
    r, f, rf = synthetic(n=30)
    with pytest.raises(ValueError):
        factor_regression(r, f.iloc[:-1], rf)
    with pytest.raises(ValueError):
        factor_regression(r, f.rename(columns={"MKT": "alpha"}))
    with pytest.raises(ValueError):
        factor_regression(r, f, rf, hac_lags=-1)
    # 缺失期整行剔除，不补零
    r2 = r.copy()
    r2.iloc[3] = np.nan
    assert factor_regression(r2, f, rf).n == 29


def test_rolling_regression_last_window_matches_full_fit():
    r, f, rf = synthetic(n=80, seed=11)
    roll = rolling_regression(r, f, 36, rf)
    assert list(roll.columns) == ["alpha", "MKT", "SMB", "HML", "alpha_t", "r_squared"]
    assert roll.iloc[:35].isna().all().all()
    last = factor_regression(r.iloc[-36:], f.iloc[-36:], rf.iloc[-36:])
    assert roll["alpha"].iloc[-1] == pytest.approx(last.alpha)
    assert roll["HML"].iloc[-1] == pytest.approx(last.betas["HML"])
    assert roll["alpha_t"].iloc[-1] == pytest.approx(last.alpha_t)
    assert roll["r_squared"].iloc[-1] == pytest.approx(last.rsquared)


def test_rolling_information_ratio(worked_example):
    p, b = worked_example.portfolio, worked_example.benchmark
    ir = rolling_information_ratio(p, b, 12, 12)
    assert ir.iloc[:11].isna().all()
    assert ir.iloc[-1] == pytest.approx(risk.information_ratio(p, b, 12))
    assert ir.iloc[-1] == pytest.approx(2.2327, abs=0.5e-4)
    # 主动收益恒定时 TE 为零，返回 NaN
    flat = rolling_information_ratio([0.02, 0.03, 0.01], [0.01, 0.02, 0.00], 2, 12)
    assert flat.isna().all()
    with pytest.raises(TypeError):
        rolling_information_ratio(p, b, 12)  # K 必须显式给出


def test_split_by_date_and_ratio(worked_example):
    ins, outs = split_in_out_of_sample(worked_example, "2025-08-31")
    assert len(ins) == 8 and len(outs) == 4
    assert ins.index.max() < outs.index.min()
    ins, outs = split_in_out_of_sample(worked_example.portfolio, 0.75)
    assert len(ins) == 9 and len(outs) == 3
    s = pd.Series(np.arange(100.0))
    assert len(split_in_out_of_sample(s, 0.29)[0]) == 29  # 浮点误差不应少取一行
    assert len(split_in_out_of_sample(s, 0.295)[0]) == 29  # ⌊29.5⌋


def test_split_refuses_shuffled_or_empty(worked_example):
    shuffled = worked_example.sample(frac=1.0, random_state=0)
    with pytest.raises(ValueError):
        split_in_out_of_sample(shuffled, 0.5)
    with pytest.raises(ValueError):
        split_in_out_of_sample(worked_example, "2026-12-31")
    with pytest.raises(ValueError):
        split_in_out_of_sample(worked_example, 1.0)


def test_fundamental_law_article_values():
    assert expected_ir(0.05, 100) == pytest.approx(0.50)
    assert expected_ir(0.05, 100, 0.70) == pytest.approx(0.35)
    np.testing.assert_allclose(expected_ir(0.05, np.array([25, 100])), [0.25, 0.50])
    with pytest.raises(ValueError):
        expected_ir(0.05, -1)
    assert math.isclose(expected_ir(0.0, 100), 0.0)
