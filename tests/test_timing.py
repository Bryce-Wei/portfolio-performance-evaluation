"""第五部分第 5 节：Treynor–Mazuy 与 Henriksson–Merton 择时回归。"""

import numpy as np
import pandas as pd
import pytest

from fundeval.alpha import RegressionResult
from fundeval.attribution import henriksson_merton, treynor_mazuy


def market(n=720, seed=21):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1966-01-31", periods=n, freq="ME")
    rf = pd.Series(0.002, index=idx)
    m = rf + pd.Series(rng.normal(0.006, 0.045, n), index=idx)
    return m, rf, rng


def test_treynor_mazuy_recovers_gamma():
    m, rf, rng = market()
    x = m - rf
    y = 0.001 + 0.9 * x + 3.0 * x**2 + rng.normal(0, 0.005, len(x))
    res = treynor_mazuy(rf + y, m, rf)
    assert isinstance(res, RegressionResult) and res.model == "TM"
    assert list(res.params.index) == ["alpha", "beta", "gamma"]
    assert res.gamma == pytest.approx(3.0, abs=3 * res.bse["gamma"])
    assert res.beta == pytest.approx(0.9, abs=0.02)
    assert res.gamma_p < 0.01
    with pytest.raises(AttributeError):
        res.upside_beta


def test_henriksson_merton_recovers_up_and_down_beta():
    m, rf, rng = market()
    x = m - rf
    y = 0.0005 + 0.7 * x + 0.5 * np.maximum(x, 0) + rng.normal(0, 0.005, len(x))
    res = henriksson_merton(rf + y, m, rf, hac_lags=3)
    assert res.model == "HM" and res.hac_lags == 3
    assert res.downside_beta == pytest.approx(0.7, abs=0.03)
    assert res.upside_beta == pytest.approx(1.2, abs=0.03)
    assert res.gamma == pytest.approx(0.5, abs=3 * res.bse["gamma"])
    assert res.gamma_p < 0.01


@pytest.mark.parametrize("model", [treynor_mazuy, henriksson_merton])
def test_constant_beta_gamma_not_significant(model):
    m, rf, rng = market(seed=4)
    x = m - rf
    y = 0.001 + 1.05 * x + rng.normal(0, 0.01, len(x))
    res = model(rf + y, m, rf)
    assert res.gamma_p > 0.05
    assert res.beta == pytest.approx(1.05, abs=0.05 if res.model == "HM" else 0.02)
