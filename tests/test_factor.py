"""第五部分第 3 节：多因子分解、A 股指数代理因子预设与报告接入。

akshare 一律用 tests/fake_akshare.py 模拟，不访问网络。
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from fundeval import cli
from fundeval.alpha.regression import factor_regression
from fundeval.attribution.factor import (
    PROXY_CAVEAT,
    factor_decomposition,
    factor_preset,
    index_proxy_factors,
    preset_codes,
)
from fundeval.etl.sources import akshare as aks
from fundeval.report import evaluate, to_excel, to_markdown
from fake_akshare import FakeAkshare

N = 60
K = 12
IDX = pd.date_range("2021-01-31", periods=N, freq="ME")
RF = 0.0015


def index_frame(seed=5):
    """四个全收益指数的合成月度收益。"""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.006, 0.05, N)
    return pd.DataFrame(
        {
            "H00300": base,
            "H00852": base + rng.normal(0.002, 0.03, N),
            "H00919": base + rng.normal(0.001, 0.02, N),
            "H00918": base + rng.normal(-0.001, 0.02, N),
        },
        index=IDX,
    )


def fund_on(factors, alpha=0.002, betas=(0.95, 0.3, 0.6), noise=0.004, seed=6):
    rng = np.random.default_rng(seed)
    y = RF + alpha + factors.to_numpy() @ np.asarray(betas) + rng.normal(0, noise, N)
    return pd.Series(y, index=IDX, name="portfolio")


# ------------------------------ 预设与因子构造 ------------------------------


def test_preset_definition():
    spec = factor_preset("cn_index_proxy")
    assert spec == {"MKT": ("H00300", "rf"), "SMB": ("H00852", "H00300"), "HML": ("H00919", "H00918")}
    assert "UMD" not in spec  # 没有经过核实的动量指数，默认不构造
    assert preset_codes("cn_index_proxy") == ["H00300", "H00852", "H00919", "H00918"]
    spec["X"] = ("a", "b")
    assert "X" not in factor_preset("cn_index_proxy")  # 返回副本
    with pytest.raises(KeyError, match="未知的因子预设"):
        factor_preset("ff3")


def test_index_proxy_factors_construction():
    idx = index_frame()
    f = index_proxy_factors(idx, RF)
    assert list(f.columns) == ["MKT", "SMB", "HML"]
    assert np.allclose(f["MKT"], idx["H00300"] - RF)
    assert np.allclose(f["SMB"], idx["H00852"] - idx["H00300"])
    assert np.allclose(f["HML"], idx["H00919"] - idx["H00918"])
    assert f.attrs["factor_type"] == "index_proxy"
    rf = pd.Series(np.linspace(0.001, 0.002, N), IDX)
    assert np.allclose(index_proxy_factors(idx, rf)["MKT"], idx["H00300"] - rf)


def test_index_proxy_factors_custom_column_and_errors():
    idx = index_frame()
    umd = pd.Series(np.random.default_rng(1).normal(0, 0.02, N), IDX, name="UMD")
    f = index_proxy_factors(idx, RF, extra=umd.to_frame())
    assert list(f.columns) == ["MKT", "SMB", "HML", "UMD"]
    with pytest.raises(ValueError, match="缺少指数：H00918"):
        index_proxy_factors(idx.drop(columns="H00918"), RF)
    gap = idx.copy()
    gap.iloc[3, 1] = np.nan
    with pytest.raises(ValueError, match="1 期缺失"):
        index_proxy_factors(gap, RF)
    with pytest.raises(ValueError, match="重名"):
        index_proxy_factors(idx, RF, extra=pd.DataFrame({"SMB": 0.0}, index=IDX))
    with pytest.raises(ValueError, match="未覆盖全部日期"):
        index_proxy_factors(idx, RF, extra=umd.iloc[5:].to_frame())


# ------------------------------ 分解与对账 ------------------------------


def test_decomposition_reconciles_and_matches_regression():
    f = index_proxy_factors(index_frame(), RF)
    p = fund_on(f)
    d = factor_decomposition(p, f, RF, hac_lags=3, use_t=True)
    reg = factor_regression(p, f, RF, hac_lags=3, use_t=True)
    assert d.alpha == pytest.approx(reg.alpha) and d.alpha_t == pytest.approx(reg.alpha_t)
    assert d.regression.use_t and d.regression.hac_lags == 3
    assert d.betas["HML"] == pytest.approx(0.6, abs=0.05)
    means = f.mean()
    for name in f.columns:
        assert d.contributions[name] == pytest.approx(d.betas[name] * means[name])
    excess = float((p - RF).mean())
    assert d.excess_mean == pytest.approx(excess)
    assert d.alpha + d.contributions.sum() + d.residual_mean == pytest.approx(excess, abs=1e-12)
    rec = d.reconciliation(K)
    assert rec["合计"] == pytest.approx(rec["平均超额收益"]) and rec["Alpha"] == pytest.approx(d.alpha * K)
    assert d.annualized_contributions(K)["MKT"] == pytest.approx(d.contributions["MKT"] * K)
    assert d.annualized_alpha(K) == pytest.approx(d.alpha * K)
    tbl = d.table(K)
    assert list(tbl.columns) == ["因子", "系数", "t", "p", "因子均值（每期）", "贡献（每期）", "贡献（算术年化）"]
    assert d.index_proxy


def test_exposure_text_directions():
    f = index_proxy_factors(index_frame(), RF)
    d = factor_decomposition(fund_on(f, betas=(0.9, -0.5, 0.6)), f, RF)
    text = d.exposure_text()
    assert "HML 显著为正" in text and "提示价值暴露" in text
    assert "SMB 显著为负" in text and "提示大盘暴露" in text
    d2 = factor_decomposition(fund_on(f, betas=(0.9, 0.0, -0.6)), f, RF)
    assert "提示成长暴露" in d2.exposure_text()
    noise = pd.Series(np.random.default_rng(9).normal(0, 0.02, N), IDX)
    d3 = factor_decomposition(noise, f.iloc[:, 1:], 0.0)
    assert d3.exposure_text().startswith("各因子系数均未显著偏离零")


def test_missing_rows_are_dropped_not_zero_filled():
    f = index_proxy_factors(index_frame(), RF)
    p = fund_on(f)
    p.iloc[7] = np.nan
    d = factor_decomposition(p, f, RF)
    assert d.n == N - 1
    assert d.factor_means["MKT"] == pytest.approx(f["MKT"].drop(IDX[7]).mean())


def test_decomposition_errors():
    f = index_proxy_factors(index_frame(), RF)
    p = fund_on(f)
    with pytest.raises(ValueError, match="观测期不一致"):
        factor_decomposition(p.iloc[1:], f, RF)
    with pytest.raises(ValueError, match="不足以估计"):
        factor_decomposition(p.iloc[:4], f.iloc[:4], RF)


def test_decomposition_reconciliation_failure_raises(monkeypatch):
    import fundeval.attribution.factor as fm

    f = index_proxy_factors(index_frame(), RF)
    monkeypatch.setattr(fm, "RECONCILE_TOL", -1.0)  # 任何差额都视为无法对账
    with pytest.raises(ValueError, match="无法对账"):
        fm.factor_decomposition(fund_on(f), f, RF)


# ------------------------------ 报告接入 ------------------------------


def report_inputs():
    idx = index_frame()
    f = index_proxy_factors(idx, RF)
    p = fund_on(f, betas=(0.95, 0.0, 0.6))
    return p, idx["H00300"].rename("benchmark"), f


def test_evaluate_with_factor_returns():
    p, b, f = report_inputs()
    rep = evaluate(p, b, RF, K, factor_returns=f, hac_lags=3, use_t=True)
    assert rep.factor is not None and rep.factor.regression.use_t
    assert rep.metric("factor_alpha_annualized") == pytest.approx(rep.factor.alpha * K)
    assert rep.metric("factor_alpha_t") == pytest.approx(rep.factor.alpha_t)
    assert rep.metric("factor_beta_HML") == pytest.approx(rep.factor.betas["HML"])
    assert rep.metric("factor_contribution_HML") == pytest.approx(rep.factor.contributions["HML"] * K)
    section = rep.section("Alpha 质量")
    assert {"多因子 Alpha（算术年化）", "多因子 Alpha t 值", "多因子 Alpha p 值", "因子暴露：HML"} <= set(section["指标"])
    assert not rep.not_done["检验"].str.contains("多因子分解").any()
    assert rep.scope["因子类型"] == PROXY_CAVEAT
    text = rep.conclusion()
    assert "多因子回归（MKT、SMB、HML）的 Alpha 为每期" in text
    assert "HML 显著为正" in text and "提示价值暴露" in text
    assert "与 Fama–French 的分组构造不同，系数不能与学术因子结果直接比较" in text
    assert "多因子" in set(rep.regression_table()["模型"])
    md = to_markdown(rep)
    assert "## 多因子暴露与贡献" in md and "收益对账" in md and "| HML |" in md


def test_evaluate_without_factors_lists_not_done(worked_example):
    rep = evaluate(worked_example, periods_per_year=K)
    row = rep.not_done[rep.not_done["检验"].str.contains("多因子分解")]
    assert len(row) == 1 and "--factors cn_index_proxy" in row["原因"].iloc[0]
    assert rep.factor is None and rep.factor_table().empty and "## 多因子暴露与贡献" not in to_markdown(rep)


def test_custom_factors_have_no_proxy_caveat():
    p, b, f = report_inputs()
    custom = pd.DataFrame(f.to_numpy(), index=f.index, columns=["F1", "F2", "F3"])
    rep = evaluate(p, b, RF, K, factor_returns=custom)
    assert not rep.factor.index_proxy and "因子类型" not in rep.scope
    assert "不能与学术因子结果直接比较" not in rep.conclusion()
    rep2 = evaluate(p, b, RF, K, factor_returns=custom, labels={"factor_type": "index_proxy"})
    assert rep2.factor.index_proxy


def test_factor_returns_must_cover_all_periods():
    p, b, f = report_inputs()
    with pytest.raises(ValueError, match="因子收益未覆盖全部 60 个共同期"):
        evaluate(p, b, RF, K, factor_returns=f.iloc[3:])
    with pytest.raises(Exception, match="DatetimeIndex"):
        evaluate(p, b, RF, K, factor_returns=f.reset_index(drop=True))


def test_factor_excel_sheet(tmp_path):
    pytest.importorskip("openpyxl")
    p, b, f = report_inputs()
    rep = evaluate(p, b, RF, K, factor_returns=f)
    sheets = pd.read_excel(to_excel(rep, tmp_path / "r.xlsx"), sheet_name=None)
    assert "多因子" in sheets and list(sheets["多因子"].columns[:3]) == ["因子", "系数", "t"]
    assert PROXY_CAVEAT in sheets["多因子"].astype(str).to_numpy()


# ------------------------------ CLI --factors ------------------------------


def test_cli_factors_preset(monkeypatch, tmp_path):
    fake = FakeAkshare()
    monkeypatch.setattr(aks, "_ak", lambda: fake)
    monkeypatch.setattr(aks, "_sleep", lambda s: None)
    out = tmp_path / "r.md"
    args = ["report", "--fund", "110011", "--start", "2024-01-01", "--end", "2024-02-29", "--benchmark", "000300",
            "--freq", "D", "--rf", "0.018", "--factors", "cn_index_proxy", "--hac-lags", "3", "--use-t",
            "--cache-dir", str(tmp_path / "cache"), "--out", str(out)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert cli.main(args) == 0
    text = out.read_text(encoding="utf-8")
    for sym in ("H00300", "H00852", "H00918", "H00919"):
        assert ("stock_zh_index_hist_csindex", {"symbol": sym}) in fake.calls
    assert "| 因子 | cn_index_proxy：MKT = 沪深300全收益 H00300 − 无风险收益；" in text
    assert "HML = 沪深300价值全收益 H00919 − 沪深300成长全收益 H00918" in text
    assert "| 因子类型 | 因子为指数代理因子" in text
    assert "多因子 Alpha t 值" in text and "HAC（Newey–West，滞后 3，t 分布）" in text
    assert "## 多因子暴露与贡献" in text and "多因子分解（第五部分第 3 节）" not in text


def test_cli_factors_rejects_unknown_preset(capsys):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["report", "--input", "x.csv", "--factors", "ff3"])
