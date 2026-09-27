"""第六部分：交易成本与策略容量，以及报告的“成本与容量”一节。"""

import numpy as np
import pandas as pd
import pytest

from fundeval import costs
from fundeval.report import evaluate, to_excel, to_markdown
from fundeval.report.summary import NAV_FEE_NOTE

BP = 1e-4


# ------------------------------ 正文演示 ------------------------------


def test_worked_example_linear_cost():
    # TO = 100%、单位成交额成本 20 bp → 成本约 40 bp
    assert costs.linear_cost_rate(1.0, 20 * BP) == pytest.approx(40 * BP)


def test_worked_example_net_alpha():
    # 费用前 Alpha 4.00%、交易成本 1.20%、其他费用 0.60% → 近似净 Alpha 2.20%
    assert costs.net_alpha(0.04, 0.012, 0.006) == pytest.approx(0.022)


# ------------------------------ 换手率 ------------------------------


def test_turnover():
    buys = pd.Series([30.0, 20.0, 50.0])
    sells = pd.Series([-40.0, 20.0, 40.0])  # 卖出金额以负数记录也取绝对值
    assert costs.turnover(buys, sells, 100.0) == pytest.approx(1.0)
    assert costs.turnover(buys, sells, pd.Series([90.0, 110.0])) == pytest.approx(1.0)
    assert costs.turnover(50, 50, 100) == pytest.approx(0.5)
    assert np.isnan(costs.turnover(buys, sells, 0.0))


def test_turnover_errors():
    with pytest.raises(ValueError, match="缺失值"):
        costs.turnover(pd.Series([1.0, np.nan]), pd.Series([1.0, 1.0]), 100)
    with pytest.raises(ValueError, match="不能为负"):
        costs.turnover(1, 1, -5)


def test_linear_cost_unit_errors():
    with pytest.raises(ValueError, match="疑似以百分数或基点输入"):
        costs.linear_cost_rate(1.0, 20)  # 20 bp 误写成 20
    with pytest.raises(ValueError, match="疑似以百分数或基点输入"):
        costs.linear_cost_rate(1.0, 0.2)  # 0.2% 误写成 0.2
    with pytest.raises(ValueError, match="不能为负"):
        costs.linear_cost_rate(-1.0, 0.002)
    with pytest.raises(ValueError, match="有限数值"):
        costs.net_alpha(float("nan"), 0.01, 0.0)


# ------------------------------ 容量 ------------------------------


def test_capacity_check():
    trade = pd.Series({"A": 5e7, "B": 2e7, "C": 1e7})
    adv = pd.Series({"A": 1e8, "B": 5e8, "C": 0.0})
    tbl = costs.capacity_check(trade, adv, max_participation=0.1, days=3)
    assert tbl.loc["A", "所需参与率"] == pytest.approx(5e7 / 3e8)
    assert tbl.loc["A", "是否超限"] == "是" and tbl.loc["B", "是否超限"] == "否"
    assert np.isnan(tbl.loc["C", "所需参与率"]) and tbl.loc["C", "是否超限"] == "无法判断"
    same = costs.capacity_check(trade.iloc[:2], 1e8, 0.2)
    assert same.loc["B", "所需参与率"] == pytest.approx(0.2) and same.loc["B", "是否超限"] == "否"


def test_capacity_errors():
    trade = pd.Series({"A": 1e7, "B": 1e7})
    with pytest.raises(ValueError, match="缺少资产"):
        costs.capacity_check(trade, pd.Series({"A": 1e8}), 0.1)
    with pytest.raises(ValueError, match="缺失值"):
        costs.capacity_check(trade, pd.Series({"A": 1e8, "B": np.nan}), 0.1)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        costs.capacity_check(trade, 1e8, 10)
    with pytest.raises(ValueError, match="days"):
        costs.capacity_check(trade, 1e8, 0.1, days=0)


# ------------------------------ 规模敏感性 ------------------------------


def test_square_root_impact():
    model = costs.square_root_impact(0.8, 0.02)
    assert model(np.array([0.04]))[0] == pytest.approx(0.8 * 0.02 * 0.2)


def test_cost_sensitivity_grows_nonlinearly():
    grid = [1e8, 1e9, 1e10]
    tbl = costs.cost_sensitivity(
        grid, turnover=2.0, adv=pd.Series({"A": 5e8, "B": 2e9}), weights=pd.Series({"A": 0.4, "B": 0.6}),
        unit_cost=15 * BP, impact_params={"coefficient": 0.7, "daily_volatility": 0.02}, max_participation=0.1,
    )
    assert list(tbl["规模"]) == grid
    assert np.allclose(tbl["线性成本率"], 2 * 2.0 * 15 * BP)
    # 平方根模型：规模扩大 10 倍，冲击成本率扩大 √10 倍
    assert tbl["冲击成本率"].iloc[1] / tbl["冲击成本率"].iloc[0] == pytest.approx(np.sqrt(10))
    assert (tbl["总成本率"] == tbl["线性成本率"] + tbl["冲击成本率"]).all()
    a = 2 * 2.0 * 1e8 * 0.4 / 250 / 5e8
    b = 2 * 2.0 * 1e8 * 0.6 / 250 / 2e9
    assert tbl["最大参与率"].iloc[0] == pytest.approx(max(a, b))
    expected = 2 * 2.0 * (0.4 * 0.7 * 0.02 * np.sqrt(a) + 0.6 * 0.7 * 0.02 * np.sqrt(b))
    assert tbl["冲击成本率"].iloc[0] == pytest.approx(expected)
    assert list(tbl["超过参与率上限"]) == ["否", "否", "是"]


def test_cost_sensitivity_custom_model_and_errors():
    linear_impact = lambda q: 0.5 * q  # noqa: E731
    tbl = costs.cost_sensitivity([1e9], 1.0, 1e9, impact_model=linear_impact)
    q = 2 * 1.0 * 1e9 / 250 / 1e9
    assert tbl["冲击成本率"].iloc[0] == pytest.approx(2 * 1.0 * 0.5 * q)
    with pytest.raises(ValueError, match="没有通用默认参数"):
        costs.cost_sensitivity([1e9], 1.0, 1e9)
    with pytest.raises(ValueError, match="须同时给出 weights"):
        costs.cost_sensitivity([1e9], 1.0, pd.Series({"A": 1e9}), impact_model=linear_impact)
    with pytest.raises(ValueError, match="和为 1"):
        costs.cost_sensitivity([1e9], 1.0, pd.Series({"A": 1e9, "B": 1e9}), weights=pd.Series({"A": 0.5, "B": 0.4}),
                               impact_model=linear_impact)
    with pytest.raises(ValueError, match="资产不一致"):
        costs.cost_sensitivity([1e9], 1.0, pd.Series({"A": 1e9}), weights=pd.Series({"B": 1.0}), impact_model=linear_impact)


# ------------------------------ 报告接入 ------------------------------

COSTS = {"turnover": 1.0, "unit_cost": 20 * BP, "other_fees": 0.006}


def test_report_fee_after_nav_does_not_double_deduct(worked_example):
    rep = evaluate(worked_example, periods_per_year=12, costs=COSTS)
    alpha = rep.capm.annualized_alpha(12)
    assert rep.metric("trade_cost") == pytest.approx(40 * BP)
    assert rep.metric("net_alpha") == pytest.approx(alpha)  # 费用后净值：不重复扣减
    assert rep.metric("gross_alpha") == pytest.approx(alpha + 40 * BP + 0.006)
    text = rep.conclusion()
    assert NAV_FEE_NOTE in text and "近似净 Alpha 即CAPM Alpha" in text
    assert any(NAV_FEE_NOTE in n for n in rep.notes)
    assert not rep.not_done["检验"].str.contains("净 Alpha").any()
    md = to_markdown(rep)
    assert "## 成本与容量" in md and "近似净 Alpha（算术年化）" in md


def test_report_gross_basis_deducts_costs(worked_example):
    rep = evaluate(worked_example, periods_per_year=12, costs=COSTS, labels={"fees": "费用前收益"})
    alpha = rep.capm.annualized_alpha(12)
    assert rep.metric("net_alpha") == pytest.approx(alpha - 40 * BP - 0.006)
    text = rep.conclusion()
    assert "α_net ≈ α_gross − c_trade − c_fee" in text and "严格口径须对扣费后净收益序列重新回归" in text
    assert rep.not_done["检验"].str.contains("扣费后净 Alpha 重估").any()


def test_report_explicit_gross_alpha_matches_worked_example(worked_example):
    spec = {"trade_cost": 0.012, "other_fees": 0.006, "gross_alpha": 0.04}
    rep = evaluate(worked_example, periods_per_year=12, costs=spec)
    assert rep.metric("net_alpha") == pytest.approx(0.022)
    assert "近似净 Alpha 约 2.20%" in rep.conclusion()


def test_report_capacity(worked_example, tmp_path):
    cap = {"trade_amount": pd.Series({"A": 5e7, "B": 1e7}), "adv": pd.Series({"A": 1e8, "B": 1e9}),
           "max_participation": 0.1, "days": 2}
    rep = evaluate(worked_example, periods_per_year=12, costs={**COSTS, "capacity": cap})
    assert rep.metric("capacity_over_limit") == 1
    assert rep.capacity_table().loc["A", "是否超限"] == "是"
    assert "所需参与率超过上限的资产：A" in rep.conclusion()
    assert not rep.not_done["检验"].str.contains("交易容量").any()
    assert "## 容量检查" in to_markdown(rep)
    pytest.importorskip("openpyxl")
    sheets = pd.read_excel(to_excel(rep, tmp_path / "r.xlsx"), sheet_name=None)
    assert "成本与容量" in sheets


def test_report_cost_input_errors(worked_example):
    with pytest.raises(ValueError, match="未知的键"):
        evaluate(worked_example, periods_per_year=12, costs={**COSTS, "tunrover": 1})
    with pytest.raises(ValueError, match="turnover 与 unit_cost"):
        evaluate(worked_example, periods_per_year=12, costs={"other_fees": 0.01})
    with pytest.raises(ValueError, match="疑似以百分数或基点输入"):
        evaluate(worked_example, periods_per_year=12, costs={"turnover": 1.0, "unit_cost": 20})
    with pytest.raises(ValueError, match="capacity"):
        evaluate(worked_example, periods_per_year=12, costs={**COSTS, "capacity": {"adv": 1e8}})


def test_report_without_costs_lists_not_done(worked_example):
    rep = evaluate(worked_example, periods_per_year=12)
    assert rep.costs is None and rep.section("成本与容量").empty
    assert rep.not_done["检验"].str.contains("扣费后净 Alpha").any()
    assert rep.not_done["检验"].str.contains("交易容量").any()
