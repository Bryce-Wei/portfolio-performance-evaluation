"""第一部分第 2 节：持仓表与成交表（schema）、现金流识别（clean）与样本长度检查（quality）。"""

import numpy as np
import pandas as pd
import pytest

from fundeval import costs
from fundeval.attribution.brinson import brinson_multi_period
from fundeval.etl import clean, quality, schema


def holdings_frame():
    return pd.DataFrame(
        {
            "date": ["2025-01-31"] * 4 + ["2025-02-28"] * 3,
            "asset": ["600519", "000858", "601318", "现金", "600519", "601318", "现金"],
            "sector": ["食品饮料", "食品饮料", "金融", "现金", "食品饮料", "金融", "现金"],
            "weight": [0.30, 0.20, 0.45, 0.05, 0.50, 0.40, 0.10],
            "market_value": [300.0, 200.0, 450.0, 50.0, 520.0, np.nan, 104.0],
        }
    )


# ------------------------------ 持仓表 ------------------------------


def test_validate_holdings_ok_and_sorted():
    df = schema.validate_holdings(holdings_frame().iloc[::-1])
    assert list(df.columns[:4]) == ["date", "asset", "sector", "weight"]
    assert df["date"].is_monotonic_increasing and pd.api.types.is_datetime64_any_dtype(df["date"])
    assert np.isnan(df.loc[(df["asset"] == "601318") & (df["date"] == "2025-02-28"), "market_value"]).all()  # 不填零


def test_validate_holdings_weights_must_sum_to_one():
    bad = holdings_frame()
    bad.loc[3, "weight"] = 0.0  # 漏记现金
    with pytest.raises(schema.SchemaError, match="2025-01-31 的权重合计为 0.950000.*现金须单列"):
        schema.validate_holdings(bad)
    # 容差可调：0.9995 在 1e-3 容差内通过
    near = holdings_frame()
    near.loc[3, "weight"] = 0.0495
    with pytest.raises(schema.SchemaError):
        schema.validate_holdings(near)
    assert len(schema.validate_holdings(near, tol=1e-3)) == 7


def test_validate_holdings_errors():
    df = holdings_frame()
    with pytest.raises(schema.SchemaError, match="缺少列"):
        schema.validate_holdings(df.drop(columns="sector"))
    missing = df.copy()
    missing.loc[0, "weight"] = np.nan
    with pytest.raises(schema.SchemaError, match="缺失值"):
        schema.validate_holdings(missing)
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(schema.SchemaError, match="重复"):
        schema.validate_holdings(dup)
    cash = df.copy()
    cash.loc[3, "sector"] = "货币"
    with pytest.raises(schema.SchemaError, match="现金须单列为行业“现金”"):
        schema.validate_holdings(cash)
    neg = df.copy()
    neg.loc[0, "market_value"] = -1
    with pytest.raises(schema.SchemaError, match="market_value"):
        schema.validate_holdings(neg)


def test_sector_weights_feed_brinson_multi_period():
    wp = schema.sector_weights(holdings_frame())
    assert list(wp.columns) == ["现金", "金融", "食品饮料"]
    assert wp.loc["2025-01-31", "食品饮料"] == pytest.approx(0.50)
    assert wp.sum(axis=1).to_numpy() == pytest.approx([1.0, 1.0])
    wb = pd.DataFrame({"现金": [0.0, 0.0], "金融": [0.6, 0.6], "食品饮料": [0.4, 0.4]}, index=wp.index)
    rp = pd.DataFrame({"现金": [0.001, 0.001], "金融": [0.02, -0.01], "食品饮料": [0.03, 0.01]}, index=wp.index)
    rb = pd.DataFrame({"现金": [0.001, 0.001], "金融": [0.015, -0.012], "食品饮料": [0.025, 0.012]}, index=wp.index)
    res = brinson_multi_period(wp, wb, rp, rb)
    cum_p = np.prod(1 + (wp * rp).sum(axis=1)) - 1
    cum_b = np.prod(1 + (wb * rb).sum(axis=1)) - 1
    assert res.cumulative_active == pytest.approx(cum_p - cum_b)
    assert float(res.totals["total"]) == pytest.approx(cum_p - cum_b)


# ------------------------------ 成交表 ------------------------------


def trades_frame():
    return pd.DataFrame(
        {
            "date": ["2025-01-10", "2025-01-03", "2025-02-12", "2025-03-05"],
            "asset": ["600519", "601318", "000858", "600519"],
            "side": ["买入", "buy", "SELL", "s"],
            "amount": [120.0, 80.0, 60.0, 40.0],
            "price": [1500.0, 45.0, 130.0, 1520.0],
            "fee": [0.1, 0.05, 0.06, 0.04],
        }
    )


def test_validate_trades_normalizes_side_and_sorts():
    df = schema.validate_trades(trades_frame())
    assert list(df["side"]) == ["buy", "buy", "sell", "sell"]
    assert list(df["date"]) == list(pd.to_datetime(["2025-01-03", "2025-01-10", "2025-02-12", "2025-03-05"]))


def test_validate_trades_errors():
    for col, value, pattern in (
        ("side", "hold", "只能为 buy 或 sell"),
        ("amount", -5.0, "amount"),
        ("price", 0.0, "price"),
        ("fee", -0.1, "fee"),
        ("amount", np.nan, "缺失值"),
    ):
        df = trades_frame()
        df.loc[0, col] = value
        with pytest.raises(schema.SchemaError, match=pattern):
            schema.validate_trades(df)
    with pytest.raises(schema.SchemaError, match="缺少列"):
        schema.validate_trades(trades_frame().drop(columns="fee"))


def test_turnover_from_trades_matches_costs_turnover():
    to = schema.turnover_from_trades(trades_frame(), 1000.0)
    assert to == pytest.approx(costs.turnover([120.0, 80.0], [60.0, 40.0], 1000.0))
    assert to == pytest.approx((200 + 100) / 2000)
    only_buys = trades_frame().iloc[:2]
    assert schema.turnover_from_trades(only_buys, 1000.0) == pytest.approx(0.1)
    assert np.isnan(schema.turnover_from_trades(trades_frame(), 0.0))


# ------------------------------ 现金流识别 ------------------------------


def test_infer_cashflows_units_times_nav():
    idx = pd.date_range("2025-01-31", periods=4, freq="ME")
    units = pd.Series([100.0, 120.0, 110.0, np.nan], index=idx)
    nav = pd.Series([1.00, 1.05, 1.10, 1.08], index=idx)
    flows = clean.infer_cashflows(units, nav)
    assert list(flows.index) == list(idx[1:]) and flows.name == "cashflow"
    assert flows.iloc[0] == pytest.approx(20 * 1.05)  # 净申购为正
    assert flows.iloc[1] == pytest.approx(-10 * 1.10)  # 净赎回为负
    assert np.isnan(flows.iloc[2])  # 缺失不填零
    with pytest.raises(schema.SchemaError, match="日期须一致"):
        clean.infer_cashflows(units, nav.iloc[1:])


def test_flag_large_cashflows():
    idx = pd.date_range("2025-01-31", periods=4, freq="ME")
    flows = pd.Series([3.0, -8.0, np.nan, 1.0], index=idx)
    assets = pd.Series([100.0, 100.0, 100.0, 0.0], index=idx)
    out = clean.flag_large_cashflows(flows, assets)
    assert list(out.columns) == ["cashflow", "assets", "ratio", "flagged"]
    assert list(out["flagged"]) == [False, True, False, False]
    assert out["ratio"].iloc[1] == pytest.approx(0.08)
    assert np.isnan(out["ratio"].iloc[2]) and np.isnan(out["ratio"].iloc[3])  # 缺失、资产为零 → NaN
    assert out.attrs["threshold"] == 0.05
    assert clean.flag_large_cashflows(flows, 100.0, threshold=0.02)["flagged"].sum() == 2
    assert "拆分子期" in clean.flag_large_cashflows.__doc__
    with pytest.raises(ValueError):
        clean.flag_large_cashflows(flows, assets, threshold=0)


# ------------------------------ 样本长度检查 ------------------------------


def test_quality_short_sample_monthly():
    idx = pd.date_range("2021-01-31", periods=24, freq="ME")
    rep = quality.data_quality_report(pd.Series(np.linspace(-0.02, 0.03, 24), index=idx))
    assert rep.periods_per_year == 12 and rep.min_periods == 36 and rep.short_sample
    issue = rep.issues()[0]
    assert "样本较短" in issue and "月度少于 36 期" in issue and "统计推断与能力判断受限" in issue
    assert rep.issue_count == len(rep.issues())
    assert "样本较短，统计推断与能力判断受限" in set(rep.summary()["结果"])

    long_idx = pd.date_range("2021-01-31", periods=36, freq="ME")
    ok = quality.data_quality_report(pd.Series(np.linspace(-0.02, 0.03, 36), index=long_idx))
    assert not ok.short_sample and not any("样本较短" in i for i in ok.issues())
    assert "满足" in set(ok.summary()["结果"])


def test_quality_short_sample_other_frequencies():
    q = pd.date_range("2022-03-31", periods=10, freq="QE")
    rep = quality.data_quality_report(pd.Series(np.linspace(0, 0.01, 10), index=q))
    assert rep.periods_per_year == 4 and rep.min_periods == 12 and "季度少于 12 期" in rep.issues()[0]
    d = pd.bdate_range("2024-01-01", periods=300)
    rep = quality.data_quality_report(pd.Series(np.linspace(0, 0.01, 300), index=d))
    assert rep.periods_per_year == 252 and rep.min_periods == 756 and rep.short_sample
    w = pd.date_range("2023-01-06", periods=200, freq="W-FRI")
    assert quality.data_quality_report(pd.Series(0.001 * np.arange(200), index=w)).min_periods == 156
    # 显式给出 K 与最短年数
    rep = quality.data_quality_report(pd.Series(np.linspace(0, 0.01, 10), index=q), periods_per_year=4, min_years=2)
    assert rep.min_periods == 8 and not rep.short_sample
    assert quality.infer_periods_per_year(pd.DatetimeIndex(["2024-01-31"])) is None


def test_evaluate_quality_flags_short_worked_example(worked_example):
    from fundeval.report import evaluate

    rep = evaluate(worked_example, periods_per_year=12)
    assert rep.quality.short_sample
    assert any("样本较短（12 期，月度少于 36 期" in i for i in rep.quality.issues())
    # 结论中样本长度由“样本较短”一句说明，不计入“数据质量有 N 项需复核”
    assert rep.quality.data_issues() == [] and "数据质量检查未发现" in rep.conclusion()
