"""稳健性检验的“敏感”判定规则（alpha.robustness._changed）与择时提醒的触发条件。

规则：(a) |t| 跨过 1.96；或 (b) 变号且剔除前或剔除后至少一边显著。两边都不显著时变号不算敏感。
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fundeval.alpha.robustness import ExclusionSensitivity, _changed, exclusion_sensitivity
from fundeval.report import conclusion, evaluate

IDX = pd.date_range("2021-01-31", periods=60, freq="ME")
EXCLUDED = pd.DatetimeIndex([pd.Timestamp("2024-09-30")])


def _fake(coef: str, value: float, t: float, n: int):
    """只含 key_table 用到的字段的回归结果替身（params、tvalues、n）。"""
    params = pd.Series({"alpha": 0.0, coef: value}) if coef != "alpha" else pd.Series({"alpha": value})
    tvalues = pd.Series({"alpha": 0.0, coef: t}) if coef != "alpha" else pd.Series({"alpha": t})
    return SimpleNamespace(params=params, tvalues=tvalues, n=n)


def _sensitivity(full: dict, trimmed: dict) -> ExclusionSensitivity:
    """{模型: (系数值, t)} → ExclusionSensitivity；CAPM 取 alpha，TM、HM 取 gamma。"""
    coef = {"CAPM": "alpha", "TM": "gamma", "HM": "gamma"}
    return ExclusionSensitivity(
        excluded=EXCLUDED,
        full={m: _fake(coef[m], *v, 60) for m, v in full.items()},
        trimmed={m: _fake(coef[m], *v, 59) for m, v in trimmed.items()},
    )


@pytest.mark.parametrize(
    "a, ta, b, tb, expected",
    [
        (-0.02, -0.79, 0.01, 0.51, False),  # 110011 TM γ：两边都不显著，变号不算
        (-0.001, -0.03, 0.03, 0.79, False),  # 110011 HM γ
        (-0.141, -6.22, -0.042, -2.02, False),  # 110020 TM γ：仍显著、同号，只是幅度缩小
        (0.004, 2.60, 0.001, 0.90, True),  # 显著 → 不显著
        (0.001, 0.90, 0.004, 2.60, True),  # 不显著 → 显著
        (0.03, 2.50, -0.001, -0.20, True),  # 变号且剔除前显著（同时跨过 1.96）
        (0.03, 2.50, -0.03, -2.40, True),  # 变号且两边都显著，|t| 未跨过 1.96 也算
        (0.03, 2.50, 0.02, 2.10, False),  # 两边都显著、同号
        (float("nan"), 1.0, 0.1, 2.5, False),  # 无法估计时不判定
    ],
)
def test_changed_rule(a, ta, b, tb, expected):
    assert _changed(a, ta, b, tb) is expected


def test_real_counterexample_110011_is_not_sensitive():
    """110011 对 H00300，2021-01 至 2025-12 月度，HAC 滞后 3、t 分布，剔除 2024-09（本地实测）：
    TM γ t −0.79 → 0.51，HM γ t −0.03 → 0.79，两边都不显著，结论不变。"""
    rob = _sensitivity(
        full={"CAPM": (0.0005, 0.40), "TM": (-0.02, -0.79), "HM": (-0.001, -0.03)},
        trimmed={"CAPM": (0.0004, 0.35), "TM": (0.01, 0.51), "HM": (0.03, 0.79)},
    )
    assert not rob.sensitive and rob.changed == []
    tbl = rob.key_table().set_index("模型")
    assert list(tbl["符号改变"]) == ["否", "是", "是"] and set(tbl["结论改变"]) == {"否"}
    text = rob.summary()
    assert "结论对" not in text and "敏感" not in text
    assert "关键系数的符号与显著性结论不变" in text
    assert "其中 TM γ、HM γ 的符号有变化，但剔除前后均不显著" in text
    assert "TM γ t 值 -0.79 → 0.51" in text and "HM γ t 值 -0.03 → 0.79" in text


def test_real_110020_shrinkage_is_not_sensitive():
    rob = _sensitivity(
        full={"CAPM": (-0.0002, -0.5), "TM": (-0.141, -6.22), "HM": (-0.3, -3.0)},
        trimmed={"CAPM": (-0.0001, -0.3), "TM": (-0.042, -2.02), "HM": (-0.2, -2.5)},
    )
    assert not rob.sensitive
    assert "符号有变化" not in rob.summary()


def test_significant_to_insignificant_is_sensitive():
    rob = _sensitivity(
        full={"CAPM": (0.004, 2.60), "TM": (0.5, 0.3), "HM": (0.1, 0.2)},
        trimmed={"CAPM": (0.001, 0.90), "TM": (0.4, 0.2), "HM": (0.1, 0.3)},
    )
    assert rob.sensitive and rob.changed == ["CAPM"]
    text = rob.summary()
    assert "结论对 1 个异常期敏感（剔除 2024-09-30）" in text and "CAPM Alpha t 值 2.60 → 0.90" in text
    assert "TM γ" not in text


def test_synthetic_significant_to_insignificant_on_real_regressions():
    """合成数据：Alpha 主要来自 4 个异常期（各 +1.2%），全样本显著、剔除后不显著。"""
    rng = np.random.default_rng(0)
    m = pd.Series(rng.normal(0.005, 0.04, len(IDX)), IDX)
    p = 0.9 * m + 0.0002 + rng.normal(0, 0.002, len(IDX))
    excluded = IDX[[10, 25, 40, 55]]
    p[excluded] += 0.012
    rob = exclusion_sensitivity(p, m, 0.0, excluded)
    tbl = rob.key_table().set_index("模型")
    assert tbl.loc["CAPM", "全样本 t"] >= 1.96 and abs(tbl.loc["CAPM", "剔除后 t"]) < 1.96
    assert rob.sensitive and "CAPM" in rob.changed


# ------------------------------ 择时 γ 的非线性策略提醒 ------------------------------

REMINDER = "γ 显著为正也可能来自期权类或动态风险控制等非线性策略"


def _sample(gamma, seed=7, noise=0.002):
    rng = np.random.default_rng(seed)
    x = pd.Series(rng.normal(0.005, 0.045, len(IDX)), IDX)
    y = 0.9 * x + gamma * x**2 + rng.normal(0, noise, len(IDX))
    return y, x


def test_reminder_absent_when_no_gamma_is_significantly_positive():
    p, b = _sample(gamma=0.0, seed=3, noise=0.01)
    rep = evaluate(p, b, robustness=False)
    assert all(abs(res.gamma_t) < 1.96 for res in rep.timing.values())
    assert REMINDER not in conclusion(rep)


def test_reminder_present_when_a_gamma_is_significantly_positive():
    p, b = _sample(gamma=8.0)
    rep = evaluate(p, b, robustness=False)
    assert any(res.gamma_t >= 1.96 for res in rep.timing.values())
    assert REMINDER in conclusion(rep)
