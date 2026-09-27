"""第六步：风格分析接入报告、剔除异常期的稳健性检验、指数表补充、数据源切换与 --rf 顺序列表。

akshare 一律用 tests/fake_akshare.py 模拟，不访问网络；真实案例放在 @pytest.mark.network 测试中。
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import requests

from fundeval import cli
from fundeval.alpha.robustness import exclusion_sensitivity
from fundeval.etl import benchmark as bm
from fundeval.etl.sources import akshare as aks
from fundeval.report import evaluate, to_excel, to_markdown
from fake_akshare import FakeAkshare

N = 60
IDX = pd.date_range("2021-01-31", periods=N, freq="ME")
RF = 0.0015


def one_extreme_period(seed=1):
    """单个极端期主导 TM γ：市场在第 45 期大涨 21%，组合当期明显跟不上（凹性），其余各期为线性关系。"""
    rng = np.random.default_rng(seed)
    m = pd.Series(rng.normal(0.005, 0.04, N), IDX)
    m.iloc[44] = 0.21
    p = RF + 0.9 * (m - RF) + rng.normal(0, 0.004, N)
    p.iloc[44] = RF + 0.9 * (0.21 - RF) - 3 * (0.21 - RF) ** 2
    return p.rename("portfolio"), m.rename("benchmark")


def style_frame(m, seed=2):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"成长": 1.1 * m + rng.normal(0, 0.01, N), "价值": 0.8 * m + rng.normal(0, 0.01, N), "现金": RF}, index=IDX
    )


# ------------------------------ 稳健性：剔除异常期 ------------------------------


def test_single_extreme_period_drives_gamma_and_is_reported():
    p, m = one_extreme_period()
    rep = evaluate(p, m, RF, 12, hac_lags=3, use_t=True)
    rob = rep.robustness
    assert list(rob.excluded) == [IDX[44]]  # 数据质量报告标出的异常期
    tbl = rep.robustness_table().set_index("模型")
    assert tbl.loc["TM", "全样本 t"] < -1.96 and abs(tbl.loc["TM", "剔除后 t"]) < 1.96
    assert tbl.loc["TM", "显著性或符号改变"] == "是" and tbl.loc["TM", "剔除后 n"] == N - 1
    assert tbl.loc["TM", "全样本估计"] == pytest.approx(rep.timing["TM"].gamma)
    assert rob.sensitive and "TM" in rob.changed
    text = rep.conclusion()
    assert "结论对 1 个异常期敏感（剔除 2024-09-30）" in text
    assert f"TM γ t 值 {tbl.loc['TM', '全样本 t']:.2f} → {tbl.loc['TM', '剔除后 t']:.2f}" in text
    assert "的结论对异常期敏感，见下文稳健性检验" in text
    md = to_markdown(rep)
    assert "## 稳健性：剔除异常期" in md and "| TM | TM γ |" in md


def test_exclusion_matches_manual_refit():
    p, m = one_extreme_period()
    rob = exclusion_sensitivity(p, m, RF, [IDX[44]], hac_lags=3, use_t=True)
    from fundeval.attribution.timing import treynor_mazuy

    keep = p.index != IDX[44]
    manual = treynor_mazuy(p[keep], m[keep], RF, hac_lags=3, use_t=True)
    assert rob.trimmed["TM"].gamma == pytest.approx(manual.gamma)
    assert rob.trimmed["TM"].gamma_t == pytest.approx(manual.gamma_t)
    full = rob.table()
    assert set(full["模型"]) == {"CAPM", "TM", "HM"} and {"全样本 t", "剔除后 t"} <= set(full.columns)
    with pytest.raises(ValueError, match="不在样本内"):
        exclusion_sensitivity(p, m, RF, ["2030-01-31"])


def test_sign_flip_counts_as_change_even_without_crossing_threshold():
    from fundeval.alpha.robustness import _changed

    assert _changed(0.1, 0.5, -0.1, -0.5)
    assert _changed(0.1, 2.5, 0.1, 1.5)
    assert not _changed(-0.141, -6.22, -0.042, -2.02)  # 真实案例：变小但仍显著、同号


def test_insensitive_exclusion_is_reported_as_unchanged():
    rng = np.random.default_rng(3)
    m = pd.Series(rng.normal(0.005, 0.03, N), IDX)
    m.iloc[10] = 0.25
    p = RF + 0.9 * (m - RF) + 0.004 + rng.normal(0, 0.002, N)
    rep = evaluate(p, m, RF, 12)
    assert len(rep.robustness.excluded) >= 1 and not rep.robustness.sensitive
    assert "关键系数的符号与显著性不变" in rep.conclusion()


def test_no_outliers_means_no_exclusion(worked_example):
    rep = evaluate(worked_example, periods_per_year=12)
    assert rep.quality.outliers.empty
    assert rep.robustness is not None and len(rep.robustness.excluded) == 0
    assert "稳健性（剔除异常期）：无异常期，未做剔除。" in rep.conclusion()
    assert rep.robustness_table().empty
    assert evaluate(worked_example, periods_per_year=12, robustness=False).robustness is None


# ------------------------------ 风格分析接入报告 ------------------------------


def test_style_section_conclusion_and_not_done():
    p, m = one_extreme_period()
    rep = evaluate(p, m, RF, 12, style_returns=style_frame(m), style_window=24)
    assert rep.style is not None and rep.style_rolling.weights.shape == (N - 23, 3)
    section = rep.section("收益来源")
    assert "风格权重：成长" in set(section["指标"]) and "残差均值（算术年化）" in set(section["指标"])
    assert rep.metric("style_residual_mean_annualized") == pytest.approx(rep.style.residual_mean * 12)
    assert rep.metric("style_r_squared") == pytest.approx(rep.style.r_squared)
    top = rep.style.top(2)
    text = rep.conclusion()
    assert f"收益变化最接近{top.index[0]}（权重" in text and f"与{top.index[1]}（权重" in text
    assert "风格权重是统计估计，不等于实际持仓" in text
    assert not rep.not_done["检验"].str.contains("风格分析").any()
    assert rep.scope["风格指数"] == "成长、价值、现金"
    md = to_markdown(rep)
    assert "## 收益来源" in md and "## 滚动风格权重（窗口 24 期）" in md


def test_style_not_done_without_style_returns(worked_example):
    rep = evaluate(worked_example, periods_per_year=12)
    row = rep.not_done[rep.not_done["检验"].str.contains("风格分析")]
    assert len(row) == 1 and "--style" in row["原因"].iloc[0]
    assert rep.section("收益来源").empty and "收益来源" not in to_markdown(rep)


def test_style_returns_must_cover_all_periods():
    p, m = one_extreme_period()
    x = style_frame(m).iloc[2:]
    with pytest.raises(ValueError, match="风格指数未覆盖全部 60 个共同期"):
        evaluate(p, m, RF, 12, style_returns=x)
    with pytest.raises(ValueError, match="style_window"):
        evaluate(p, m, RF, 12, style_window=24)


def test_style_excel_sheet(tmp_path):
    pytest.importorskip("openpyxl")
    p, m = one_extreme_period()
    rep = evaluate(p, m, RF, 12, style_returns=style_frame(m), style_window=24)
    path = to_excel(rep, tmp_path / "r.xlsx")
    sheets = pd.read_excel(path, sheet_name=None)
    assert {"风格分析", "稳健性"} <= set(sheets)
    style_sheet = sheets["风格分析"]
    assert list(style_sheet.columns[:2]) == ["风格", "权重"]
    assert "窗口末期（窗口 24 期）" in style_sheet.astype(str).to_numpy()
    assert "结论对 1 个异常期敏感" in str(sheets["稳健性"].iloc[0, 0])


# ------------------------------ 指数表与风格预设 ------------------------------


def test_new_index_records():
    assert bm.total_return_code("000918") == "H00918"
    assert bm.total_return_code("000919") == "H00919"
    assert bm.total_return_code("000922") == "H00922"
    assert bm.total_return_code("932000") is None
    assert bm.index_return_type("H00922") == "total" and bm.index_return_type("932000") == "price"
    assert bm.index_record("H20932") is None  # H20932 不是中证2000全收益
    assert bm.lookup_index_code("中证红利指数") == "000922"
    assert bm.index_return_type("H11001") == "unknown"
    assert aks.index_source("932000") == "csindex" and aks.index_source("000918") == "em"


def test_style_presets():
    eq = bm.style_preset("cn_equity")
    assert eq == {"沪深300成长": "H00918", "沪深300价值": "H00919", "中证500": "H00905", "中证1000": "H00852", "现金": "cash"}
    bal = bm.style_preset("cn_balanced")
    assert bal == {**eq, "中证全债": "H11001"}
    assert "H00922" not in eq.values()  # 中证红利与沪深300价值相关性高，不放进默认预设
    eq["x"] = "y"
    assert "x" not in bm.style_preset("cn_equity")  # 返回副本
    with pytest.raises(KeyError, match="未知的风格预设"):
        bm.style_preset("us_equity")


def test_parse_style_spec():
    assert cli.parse_style_spec("cn_equity") == bm.style_preset("cn_equity")
    assert cli.parse_style_spec("H00918, H00919,cash") == {"H00918": "H00918", "H00919": "H00919", "cash": "cash"}
    for bad in ("H00918,,cash", "H00918,H00918", "H00918"):
        with pytest.raises(ValueError):
            cli.parse_style_spec(bad)


# ------------------------------ CLI --style ------------------------------


START, END = "2024-01-01", "2024-02-29"


@pytest.fixture
def sleeps(monkeypatch):
    waited = []
    monkeypatch.setattr(aks, "_sleep", waited.append)
    return waited


def run_cli(monkeypatch, tmp_path, *extra, fake=None):
    fake = fake or FakeAkshare()
    monkeypatch.setattr(aks, "_ak", lambda: fake)
    out = tmp_path / "r.md"
    args = ["report", "--fund", "110011", "--start", START, "--end", END, "--benchmark", "000300",
            "--cache-dir", str(tmp_path / "cache"), "--out", str(out), *extra]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        code = cli.main(args)
    return code, (out.read_text(encoding="utf-8") if out.exists() else ""), fake


def test_cli_style_preset(monkeypatch, tmp_path, sleeps):
    code, text, fake = run_cli(
        monkeypatch, tmp_path, "--freq", "D", "--rf", "0.018", "--style", "cn_equity", "--style-window", "20"
    )
    assert code == 0
    for sym in ("H00918", "H00919", "H00905", "H00852"):
        assert ("stock_zh_index_hist_csindex", {"symbol": sym}) in fake.calls
    assert "| 风格指数 | 沪深300成长：H00918（全收益）；" in text and "现金：无风险收益" in text
    assert "| 风格指数数据源 | H00918：中证指数官网" in text
    assert "## 收益来源" in text and "风格权重：沪深300成长" in text and "## 滚动风格权重（窗口 20 期）" in text
    assert "收益变化最接近" in text and "风格分析（第五部分第 1 节）" not in text


def test_cli_style_code_list_uses_total_return_codes(monkeypatch, tmp_path, sleeps):
    code, text, fake = run_cli(monkeypatch, tmp_path, "--freq", "D", "--rf", "0.018", "--style", "000918,000919,cash")
    assert code == 0
    assert ("stock_zh_index_hist_csindex", {"symbol": "H00918"}) in fake.calls
    assert fake.count("index_zh_a_hist") == 0  # 基准与风格都换成全收益代码，走中证官网
    assert "风格权重：000918" in text and "000918：H00918（全收益）" in text


def test_cli_style_balanced_notes_unknown_bond_type(monkeypatch, tmp_path, sleeps):
    code, text, _ = run_cli(monkeypatch, tmp_path, "--freq", "D", "--rf", "0.018", "--style", "cn_balanced")
    assert code == 0
    assert "中证全债：H11001（未知）" in text
    assert "风格指数 中证全债（H11001）收益类型为未知" in text


def test_cli_style_window_requires_style(monkeypatch, tmp_path, capsys, sleeps):
    code, _, _ = run_cli(monkeypatch, tmp_path, "--rf", "0.018", "--style-window", "20")
    assert code == 2


# ------------------------------ 数据源：非最后候选只试 1 次；--rf 顺序列表 ------------------------------

SHIBOR_DOWN = requests.exceptions.ChunkedEncodingError("Connection broken: IncompleteRead")
CGB_DOWN = requests.exceptions.ReadTimeout("Read timed out")


def test_parse_rate_sources():
    assert aks.parse_rate_sources("auto") == ("shibor3m", "cgb2y")
    assert aks.parse_rate_sources("cgb2y, shibor3m") == ("cgb2y", "shibor3m")
    assert aks.parse_rate_sources(("cgb10y",)) == ("cgb10y",)
    for bad in ("cgb2y,,shibor3m", "cgb2y,cgb2y", "auto,cgb2y", "libor"):
        with pytest.raises(ValueError):
            aks.parse_rate_sources(bad)


def test_rf_list_tries_in_order_and_only_last_is_retried(monkeypatch, tmp_path, sleeps):
    fake = FakeAkshare(fail={"bond_zh_us_rate": CGB_DOWN})
    monkeypatch.setattr(aks, "_ak", lambda: fake)
    with pytest.warns(RuntimeWarning, match="已改用Shibor 3M"):
        rf = aks.risk_free_returns(START, END, "M", source="cgb2y,shibor3m", cache_dir=tmp_path)
    assert fake.count("bond_zh_us_rate") == 1 and fake.count("rate_interbank") == 1 and sleeps == []
    assert rf.attrs["source"] == "shibor3m" and rf.attrs["failed"] == ["cgb2y"]


def test_rf_list_last_candidate_uses_full_retries(monkeypatch, tmp_path, sleeps):
    fake = FakeAkshare(fail={"rate_interbank": SHIBOR_DOWN, "bond_zh_us_rate": CGB_DOWN})
    monkeypatch.setattr(aks, "_ak", lambda: fake)
    with pytest.raises(aks.DataSourceUnavailable, match="shibor3m → cgb2y"):
        aks.risk_free_returns(START, END, "M", source="auto", cache_dir=tmp_path)
    assert fake.count("rate_interbank") == 1
    assert fake.count("bond_zh_us_rate") == aks.MAX_ATTEMPTS and sleeps == [1.0, 2.0]


def test_cli_rf_accepts_ordered_list(monkeypatch, tmp_path, sleeps):
    fake = FakeAkshare(fail={"bond_zh_us_rate": CGB_DOWN})
    code, text, fake = run_cli(monkeypatch, tmp_path, "--freq", "W", "--rf", "cgb2y,shibor3m", fake=fake)
    assert code == 0 and fake.count("bond_zh_us_rate") == 1
    assert "Shibor 3M（中国国债 2 年期收益率 获取失败后改用）" in text


def test_cli_rf_auto_equals_shibor_then_cgb(monkeypatch, tmp_path, sleeps):
    fake = FakeAkshare(fail={"rate_interbank": SHIBOR_DOWN})
    code, text, fake = run_cli(monkeypatch, tmp_path, "--freq", "W", "--rf", "auto", fake=fake)
    assert code == 0 and fake.count("rate_interbank") == 1 and fake.count("bond_zh_us_rate") == 1
    assert "获取失败后改用" in text
