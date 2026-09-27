"""第一部分补充：频率转换、复合基准、合同基准解析与数据质量报告。"""

import numpy as np
import pandas as pd
import pytest

from fundeval.etl import benchmark, quality, schema
from fundeval.etl.returns import to_frequency


def daily(values, start="2025-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


# ------------------------------ to_frequency ------------------------------


def test_to_frequency_monthly_compounds_within_period():
    idx = pd.bdate_range("2025-01-01", "2025-02-28")
    r = pd.Series(np.linspace(-0.01, 0.012, len(idx)), index=idx, name="portfolio")
    out = to_frequency(r, "M")
    assert list(out.index) == [pd.Timestamp("2025-01-31"), pd.Timestamp("2025-02-28")]
    jan = r[r.index.month == 1]
    assert out.iloc[0] == pytest.approx(np.prod(1 + jan) - 1, abs=1e-15)
    assert out.name == "portfolio"


def test_to_frequency_weekly_and_quarterly_labels():
    r = daily([0.01] * 10, start="2025-03-24")  # 周一开始，两周
    weekly = to_frequency(r, "W")
    assert list(weekly.index) == [pd.Timestamp("2025-03-28"), pd.Timestamp("2025-04-04")]
    assert weekly.iloc[0] == pytest.approx(1.01**5 - 1)
    quarterly = to_frequency(r, "Q")
    assert list(quarterly.index) == [pd.Timestamp("2025-03-31"), pd.Timestamp("2025-06-30")]


def test_to_frequency_missing_day_makes_period_nan():
    idx = pd.bdate_range("2025-01-01", "2025-03-31")
    r = pd.Series(0.001, index=idx)
    r.loc["2025-02-10"] = np.nan
    out = to_frequency(r, "M")
    assert np.isnan(out.loc["2025-02-28"])
    assert out.loc["2025-01-31"] == pytest.approx(1.001 ** len(r["2025-01"]) - 1)


def test_to_frequency_min_obs_and_dataframe():
    idx = pd.bdate_range("2025-01-29", "2025-02-28")  # 1 月只有 3 天
    df = pd.DataFrame({"portfolio": 0.001, "benchmark": 0.002}, index=idx)
    out = to_frequency(df, "M", min_obs=10)
    assert out.loc["2025-01-31"].isna().all()
    assert out.loc["2025-02-28", "benchmark"] == pytest.approx(1.002**20 - 1)
    with pytest.raises(ValueError):
        to_frequency(df, "M", min_obs=0)
    with pytest.raises(schema.SchemaError):
        to_frequency(df, "D")


# ------------------------------ composite_benchmark ------------------------------


def components():
    idx = pd.date_range("2025-01-31", periods=4, freq="ME")
    return pd.DataFrame({"000300": [0.05, -0.03, 0.02, 0.01], "H11001": [0.004, 0.003, -0.001, 0.002]}, index=idx)


def test_composite_period_rebalance_is_weighted_sum():
    c = components()
    out = benchmark.composite_benchmark(c, {"000300": 0.8, "H11001": 0.2})
    np.testing.assert_allclose(out.to_numpy(), (0.8 * c["000300"] + 0.2 * c["H11001"]).to_numpy())
    assert out.name == "benchmark"
    same = benchmark.composite_benchmark({"000300": c["000300"], "H11001": c["H11001"]}, [0.8, 0.2])
    pd.testing.assert_series_equal(out, same)


def test_composite_buy_and_hold_tracks_drifting_weights():
    c = components()
    out = benchmark.composite_benchmark(c, {"000300": 0.8, "H11001": 0.2}, rebalance="none")
    wealth = 0.8 * (1 + c["000300"]).cumprod() + 0.2 * (1 + c["H11001"]).cumprod()
    assert np.prod(1 + out) == pytest.approx(wealth.iloc[-1])
    assert out.iloc[0] == pytest.approx(0.8 * 0.05 + 0.2 * 0.004)
    assert out.iloc[1] != pytest.approx(0.8 * -0.03 + 0.2 * 0.003)  # 第二期权重已漂移


def test_composite_missing_and_weight_checks():
    c = components()
    c.iloc[1, 1] = np.nan
    out = benchmark.composite_benchmark(c, {"000300": 0.8, "H11001": 0.2})
    assert np.isnan(out.iloc[1]) and np.isfinite(out.iloc[2])
    held = benchmark.composite_benchmark(c, {"000300": 0.8, "H11001": 0.2}, rebalance="none")
    assert held.iloc[1:].isna().all() and np.isfinite(held.iloc[0])
    with pytest.raises(ValueError, match="加总为 1"):
        benchmark.composite_benchmark(components(), {"000300": 0.8, "H11001": 0.3})
    with pytest.raises(ValueError, match="不一致"):
        benchmark.composite_benchmark(components(), {"000300": 1.0})
    with pytest.raises(ValueError):
        benchmark.composite_benchmark(components(), [0.5, 0.5], rebalance="monthly")


# ------------------------------ parse_benchmark ------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("沪深300指数收益率*80%+中债综合指数收益率*20%", [("沪深300指数", 0.8), ("中债综合指数", 0.2)]),
        ("沪深300指数收益率×60% + 中证全债指数收益率×40%", [("沪深300指数", 0.6), ("中证全债指数", 0.4)]),
        ("80%×中证500指数收益率＋20%×中债综合指数收益率", [("中证500指数", 0.8), ("中债综合指数", 0.2)]),
        ("沪深300指数收益率×95%+银行活期存款利率(税后)×5%", [("沪深300指数", 0.95), ("银行活期存款利率(税后)", 0.05)]),
        ("中证800指数收益率*100%", [("中证800指数", 1.0)]),
    ],
)
def test_parse_benchmark(text, expected):
    got = benchmark.parse_benchmark(text)
    assert [n for n, _ in got] == [n for n, _ in expected]
    assert [w for _, w in got] == pytest.approx([w for _, w in expected])


@pytest.mark.parametrize(
    "text",
    [
        "沪深300指数收益率*80%+中债综合指数收益率",  # 缺少权重
        "沪深300指数收益率*60%+中债综合指数收益率*30%",  # 加总不为 100%
        "沪深300指数收益率*80%+中债综合指数收益率*20%-1%",  # 含减号
        "max(沪深300指数收益率, 0)",
        "",
    ],
)
def test_parse_benchmark_refuses_to_guess(text):
    with pytest.raises(ValueError, match="手动指定|为空"):
        benchmark.parse_benchmark(text)


def test_lookup_index_code():
    assert benchmark.lookup_index_code("沪深300指数") == "000300"
    assert benchmark.lookup_index_code("中证全债指数收益率") == "H11001"
    assert benchmark.lookup_index_code("中债综合指数") == "cbond:composite"
    with pytest.raises(KeyError, match="手动指定"):
        benchmark.lookup_index_code("银行活期存款利率(税后)")


# ------------------------------ data_quality_report ------------------------------


def test_quality_report_flags_missing_outliers_stale_and_cross_check():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2025-01-01", periods=60)
    p = pd.Series(rng.normal(0.0005, 0.01, 60), index=idx)
    p.iloc[10:14] = 0.0  # 连续 4 期零收益：疑似停牌
    p.iloc[30] = 0.25  # 异常收益
    b = pd.Series(rng.normal(0.0004, 0.009, 60), index=idx)
    b.iloc[5] = np.nan
    df = pd.DataFrame({"portfolio": p, "benchmark": b, "risk_free": 0.0001})
    growth = p.copy()
    growth.iloc[40] += 0.003
    check = quality.nav_growth_check(p, growth, tolerance=0.0005)
    rep = quality.data_quality_report(df, cross_check=check)
    assert rep.periods == 60 and rep.start == idx[0] and rep.end == idx[-1]
    assert rep.missing.loc["benchmark", "missing"] == 1
    assert (rep.outliers["date"] == idx[30]).any()
    assert "risk_free" not in set(rep.outliers["column"])
    stale = rep.stale[rep.stale["column"] == "portfolio"].iloc[0]
    assert stale["start"] == idx[10] and stale["end"] == idx[13] and stale["length"] == 4
    assert list(rep.cross_check.index) == [idx[40]]
    assert rep.cross_check["difference"].iloc[0] == pytest.approx(-0.003)
    summary = rep.summary()
    assert "净值与日增长率差异超过 5 个基点的日期" in set(summary["项目"])
    issues = rep.issues()
    assert any("疑似停牌" in i for i in issues) and any("基点" in i for i in issues)
    assert rep.issue_count == len(issues)


def test_quality_report_with_nav_uses_flag_stale_on_nav():
    idx = pd.bdate_range("2025-01-01", periods=8)
    nav = pd.Series([1.0, 1.01, 1.01, 1.01, 1.02, 1.03, 1.02, 1.04], index=idx)
    r = (nav / nav.shift(1) - 1).dropna()
    rep = quality.data_quality_report(r, nav=nav)
    assert len(rep.stale) == 1 and rep.stale.iloc[0]["length"] == 3
    assert rep.cross_check is None
    assert "未提供核对数据" in set(rep.summary()["结果"])


def test_nav_growth_check_missing_is_not_flagged():
    idx = pd.bdate_range("2025-01-01", periods=3)
    check = quality.nav_growth_check(pd.Series([0.01, 0.02, 0.0], index=idx), pd.Series([0.01, np.nan, 0.0], index=idx))
    assert not check["flagged"].any() and np.isnan(check["difference"].iloc[1])
    with pytest.raises(ValueError):
        quality.nav_growth_check(check["total_return"], check["total_return"], tolerance=-1)
