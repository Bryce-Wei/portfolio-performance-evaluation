"""Sharpe 收益型风格分析（正文第五部分第 1 节）：合成数据还原权重、约束、共线性、收敛与滚动估计。"""

import warnings

import numpy as np
import pandas as pd
import pytest

from fundeval.attribution import style as st
from fundeval.attribution.style import StyleNotConverged, rolling_style, style_analysis

N = 120
IDX = pd.date_range("2015-01-31", periods=N, freq="ME")


def styles(seed=0, k=3, n=N):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0.008, 0.05, (n, k)), index=IDX[:n], columns=list("ABCDE")[:k])


def noise(seed=1, scale=0.002, n=N):
    return pd.Series(np.random.default_rng(seed).normal(0, scale, n), index=IDX[:n])


@pytest.mark.parametrize("objective", ["variance", "sse"])
def test_recovers_weights(objective):
    x = styles()
    r = 0.6 * x["A"] + 0.4 * x["B"] + noise()
    res = style_analysis(r, x, objective=objective)
    assert res.weights["A"] == pytest.approx(0.6, abs=0.03)
    assert res.weights["B"] == pytest.approx(0.4, abs=0.03)
    assert res.weights["C"] == pytest.approx(0.0, abs=0.03)
    assert res.weights.sum() == pytest.approx(1.0, abs=1e-10) and (res.weights >= 0).all()
    assert res.r_squared > 0.95 and res.n == N and res.objective == objective
    assert res.diagnostics["converged"] and res.collinear_pairs == []


def test_negative_true_weight_is_estimated_as_zero():
    x = styles()
    r = 0.8 * x["A"] + 0.5 * x["B"] - 0.3 * x["C"] + noise()
    res = style_analysis(r, x)
    assert res.weights["C"] == pytest.approx(0.0, abs=1e-8)
    assert (res.weights >= 0).all() and res.weights.sum() == pytest.approx(1.0, abs=1e-10)
    assert res.weights["A"] > res.weights["B"] > 0


def test_r_squared_and_residual_statistics():
    x = styles()
    r = 0.5 * x["A"] + 0.5 * x["B"] + 0.001 + noise()
    res = style_analysis(r, x)
    resid = r - x @ res.weights
    assert res.r_squared == pytest.approx(1 - resid.var() / r.var())
    assert res.residual_mean == pytest.approx(resid.mean())
    assert res.annualized_residual_mean(12) == pytest.approx(resid.mean() * 12)
    assert res.annualized_residual_volatility(12) == pytest.approx(resid.std(ddof=1) * np.sqrt(12))
    # 目标为残差方差时，常数超额收益进入残差均值（相当于正文公式中的 α），不扭曲权重
    assert res.residual_mean == pytest.approx(0.001, abs=0.0005)
    assert res.weights["A"] == pytest.approx(0.5, abs=0.03)
    assert list(res.top(2).index) == sorted(["A", "B"], key=lambda c: -res.weights[c])


def test_collinear_styles_warn_and_are_recorded():
    x = styles()
    x["A2"] = x["A"] + np.random.default_rng(5).normal(0, 0.002, N)
    r = 0.6 * x["A"] + 0.4 * x["B"] + noise()
    with pytest.warns(RuntimeWarning, match="高度相关.*A 与 A2"):
        res = style_analysis(r, x)
    assert [(a, b) for a, b, _ in res.collinear_pairs] == [("A", "A2")]
    assert res.collinear_pairs[0][2] > st.COLLINEARITY_THRESHOLD
    assert "collinearity_warning" in res.diagnostics
    assert res.weights["A"] + res.weights["A2"] == pytest.approx(0.6, abs=0.03)


def test_risk_free_adds_cash_asset():
    x = styles()
    r = 0.5 * x["A"] + 0.5 * 0.002 + noise(scale=0.001)
    res = style_analysis(r, x, risk_free=0.002)
    assert list(res.weights.index) == ["A", "B", "C", "cash"]
    assert res.weights["cash"] == pytest.approx(0.5, abs=0.05)
    with pytest.raises(ValueError, match="已含 'cash'"):
        style_analysis(r, x.assign(cash=0.002), risk_free=0.002)


def test_not_converged_raises():
    x = styles()
    r = 0.6 * x["A"] + 0.4 * x["B"] + noise()
    with pytest.raises(StyleNotConverged, match="未收敛"):
        style_analysis(r, x, maxiter=1)


def test_failed_solver_result_is_not_returned(monkeypatch):
    x = styles()
    r = 0.6 * x["A"] + 0.4 * x["B"] + noise()

    def fake_minimize(*args, **kwargs):
        from scipy.optimize import OptimizeResult

        return OptimizeResult(x=np.array([0.5, 0.5, 0.0]), success=False, message="Positive directional derivative", nit=7)

    monkeypatch.setattr(st.optimize, "minimize", fake_minimize)
    with pytest.raises(StyleNotConverged, match="Positive directional derivative"):
        style_analysis(r, x)


def test_input_validation_and_missing_rows():
    x = styles()
    r = 0.6 * x["A"] + 0.4 * x["B"] + noise()
    with pytest.raises(ValueError, match="objective"):
        style_analysis(r, x, objective="ols")
    with pytest.raises(ValueError, match="少于风格资产数"):
        style_analysis(r.iloc[:4], x.iloc[:4])
    r2 = r.copy()
    r2.iloc[3] = np.nan
    assert style_analysis(r2, x).n == N - 1  # 缺失的期整行剔除，不补零
    with pytest.raises(ValueError, match="观测期不一致"):
        style_analysis(r.iloc[1:], x)


def test_rolling_style_shape_and_window_validation():
    x = styles()
    r = 0.6 * x["A"] + 0.4 * x["B"] + noise()
    roll = rolling_style(r, x, 36)
    assert roll.weights.shape == (N - 36 + 1, 3) and list(roll.weights.columns) == ["A", "B", "C"]
    assert roll.r_squared.shape == (N - 36 + 1,)
    assert roll.weights.index[0] == IDX[35] and roll.weights.index[-1] == IDX[-1]
    np.testing.assert_allclose(roll.weights.sum(axis=1), 1.0, atol=1e-10)
    assert (roll.weights.to_numpy() >= 0).all()
    last = style_analysis(r.iloc[-36:], x.iloc[-36:])
    pd.testing.assert_series_equal(roll.weights.iloc[-1], last.weights, check_names=False)
    assert roll.table().shape == (N - 35, 4)

    with pytest.raises(ValueError, match="小于风格资产数 3 \\+ 2"):
        rolling_style(r, x, 4)
    with pytest.raises(ValueError, match="小于风格资产数 4 \\+ 2"):
        rolling_style(r, x, 5, risk_free=0.001)  # cash 计入资产数
    with pytest.raises(ValueError, match="超过有效样本"):
        rolling_style(r, x, N + 1)
    with pytest.raises(ValueError, match="整数"):
        rolling_style(r, x, 36.0)
    assert rolling_style(r, x, 5).weights.shape[0] == N - 4


def test_rolling_style_warns_once_for_collinearity():
    x = styles()
    x["A2"] = x["A"] + np.random.default_rng(5).normal(0, 0.002, N)
    r = 0.6 * x["A"] + 0.4 * x["B"] + noise()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        roll = rolling_style(r, x, 60)
    assert sum("高度相关" in str(w.message) for w in caught) == 1
    assert roll.collinear_pairs and roll.collinear_pairs[0][:2] == ("A", "A2")
