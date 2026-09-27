"""只输入基金代码即可评价：基金概况、指数目录、合同基准解析、--benchmark contract 与多基金对比。

样本按 akshare 1.18.97 的真实返回格式录制（见 tests/data/akshare/README.md）；测试不联网。
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import pytest

from fundeval import cli
from fundeval.etl import benchmark as bm
from fundeval.etl.sources import akshare as aks
from fundeval.report.compare import SORT_CAVEATS, _sort, compare
from fake_akshare import FakeAkshare

START, END = "2024-01-01", "2024-02-29"

TEXTS = {
    "110011": "沪深300指数收益率*50%+中证香港300指数收益率*30%+中债总指数收益率*20%",
    "110020": "沪深300指数收益率*95%+活期存款利率(税后)*5%",
    "000001": "中证800成长指数收益率*70%+中债-综合全价(总值)指数收益率*30%",
}
TYPES = {"110011": "QDII-混合偏股", "110020": "指数型-股票", "000001": "混合型-灵活"}
MAP_110011 = ["--benchmark-map", "中债总指数=cbond:composite"]


@pytest.fixture
def fake(monkeypatch):
    ak = FakeAkshare()
    monkeypatch.setattr(aks, "_ak", lambda: ak)
    return ak


@pytest.fixture
def cache(tmp_path):
    return {"cache_dir": tmp_path / "cache"}


@pytest.fixture
def catalog(fake, cache):
    return aks.index_catalog(**cache)


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(*args, **kwargs)


def _age(path, days):
    t = time.time() - days * 86400
    os.utime(path, (t, t))


# ------------------------------ 基金概况 ------------------------------


@pytest.mark.parametrize("code", list(TEXTS))
def test_fund_profile_type_and_benchmark_text(fake, cache, code):
    p = aks.fund_profile(code, **cache)
    assert p.code == code and p.fund_type == TYPES[code] and p.benchmark_text == TEXTS[code]
    assert p.full_name and p.short_name and p.manager == p.raw["基金管理人"]
    assert ("fund_overview_em", {"symbol": code}) in fake.calls


def test_fund_profile_parses_dates_fees_and_tracking(fake, cache):
    p = aks.fund_profile("110011", **cache)
    assert p.inception_date == pd.Timestamp("2008-06-19")  # “2008年06月19日 / 12.267亿份”
    assert p.management_fee == pytest.approx(0.015) and p.custodian_fee == pytest.approx(0.0025)
    assert p.sales_service_fee == "---"  # 无法解析时保留原文，不猜测
    assert p.tracking_target is None and p.is_qdii and not p.is_index_fund
    idx = aks.fund_profile("110020", **cache)
    assert idx.tracking_target == "沪深300指数" and idx.is_index_fund and not idx.is_qdii


@pytest.mark.parametrize(
    "text, expected",
    [("1.50%（每年）", 0.015), ("0.00%（每年）", 0.0), ("0.15%", 0.0015), ("---", "---"), ("", None), (None, None)],
)
def test_parse_fee_rate(text, expected):
    assert aks.parse_fee_rate(text) == expected


def test_parse_inception_date_keeps_none_when_unparseable():
    assert aks.parse_inception_date("2001年12月18日 / 32.368亿份") == pd.Timestamp("2001-12-18")
    assert aks.parse_inception_date("--- / ---") is None


def test_fund_profile_unknown_code_raises(fake, cache):
    with pytest.raises(ValueError, match="未取到基金 999999 的概况"):
        aks.fund_profile("999999", **cache)


def test_fund_profile_cache_lasts_one_day(fake, cache):
    aks.fund_profile("000001", **cache)
    again = aks.fund_profile("000001", **cache)
    assert fake.count("fund_overview_em") == 1 and again.raw["基金代码"].startswith("000001")  # 前导零保留
    path = aks.cache_path(cache["cache_dir"], "fund_overview_em", "000001", None, None)
    _age(path, 1.5)
    aks.fund_profile("000001", **cache)
    assert fake.count("fund_overview_em") == 2


def test_index_catalog_cache_lasts_seven_days_and_keeps_codes(fake, cache):
    cat = aks.index_catalog(**cache)
    assert {"000300", "H11164", "H30355"} <= set(cat["指数代码"])
    path = aks.cache_path(cache["cache_dir"], "index_csindex_all", "all", None, None)
    _age(path, 6)
    cached = aks.index_catalog(**cache)
    assert fake.count("index_csindex_all") == 1 and "000300" in set(cached["指数代码"])
    _age(path, 8)
    aks.index_catalog(**cache)
    assert fake.count("index_csindex_all") == 2


# ------------------------------ 合同基准解析 ------------------------------


def test_parse_real_contract_texts():
    assert bm.parse_benchmark(TEXTS["110011"]) == [("沪深300指数", 0.5), ("中证香港300指数", 0.3), ("中债总指数", 0.2)]
    assert bm.parse_benchmark(TEXTS["110020"]) == [("沪深300指数", 0.95), ("活期存款利率(税后)", 0.05)]
    assert bm.parse_benchmark(TEXTS["000001"]) == [("中证800成长指数", 0.7), ("中债-综合全价(总值)指数", 0.3)]
    # 减法仍然拒绝
    with pytest.raises(ValueError):
        bm.parse_benchmark("沪深300指数收益率*80%-中债综合指数收益率*20%")


def test_resolve_110020_total_return_plus_cash():
    comps = bm.resolve_benchmark(TEXTS["110020"])
    assert [(c.code, c.weight, c.return_type) for c in comps] == [("H00300", 0.95, "total"), ("cash:demand", 0.05, "cash")]
    cash = comps[1]
    assert cash.rate == pytest.approx(0.0035) and "2015-10-24" in cash.note and cash.source == "constant"
    assert bm.components_return_type(comps) == "全收益"
    # 价格口径时沿用价格指数
    assert bm.resolve_benchmark(TEXTS["110020"], return_type="price")[0].code == "000300"
    table = bm.resolution_table(comps)
    assert list(table.columns[:7]) == ["成分", "权重", "代码", "收益类型", "币种", "数据源", "解析依据"]


def test_resolve_000001_catalog_and_clean_price_cbond(catalog):
    comps = bm.resolve_benchmark(TEXTS["000001"], catalog)
    growth, bond = comps
    assert (growth.code, growth.return_type, growth.method, growth.currency) == ("H30355", "unknown", "catalog", "人民币")
    assert (bond.code, bond.return_type, bond.source) == ("cbond:composite_full", "price", "cbond")
    assert bm.components_return_type(comps) == "含价格指数成分"
    assert bm.needs_price_caveat(bm.components_return_type(comps))


def test_resolve_110011_ambiguous_bond_index_lists_component_and_hint(catalog):
    with pytest.raises(bm.BenchmarkResolutionError) as info:
        quiet(bm.resolve_benchmark, TEXTS["110011"], catalog)
    msg = str(info.value)
    assert "中债总指数" in msg and "歧义" in msg and "--benchmark-map" in msg and "中债总指数=" in msg
    assert [n for n, _ in info.value.unresolved] == ["中债总指数"]


def test_resolve_collects_all_unresolved_components(catalog):
    with pytest.raises(bm.BenchmarkResolutionError) as info:
        bm.resolve_benchmark("中证不存在指数收益率*60%+中债总指数收益率*40%", catalog)
    assert [n for n, _ in info.value.unresolved] == ["中证不存在指数", "中债总指数"]


def test_catalog_requires_exactly_one_match(catalog):
    dup = pd.concat([catalog, catalog[catalog["指数代码"] == "H30355"].assign(指数代码="H99999")], ignore_index=True)
    with pytest.raises(bm.BenchmarkResolutionError, match="匹配到多条.*H30355.*H99999"):
        bm.resolve_component("中证800成长指数", dup)
    with pytest.raises(bm.BenchmarkResolutionError, match="没有简称或全称为“800”.*候选.*H30355.*H30356"):
        bm.resolve_component("800指数", catalog)
    with pytest.raises(bm.BenchmarkResolutionError, match="未提供中证指数目录"):
        bm.resolve_component("中证800成长指数")


def test_catalog_is_loaded_only_when_needed():
    calls = []

    def loader():
        calls.append(1)
        raise AssertionError("不应调用")

    bm.resolve_benchmark(TEXTS["110020"], loader)
    assert not calls


def test_overrides_come_first_and_accept_cash(catalog):
    comps = quiet(bm.resolve_benchmark, TEXTS["110011"], catalog, {"中债总指数": "cbond:composite"})
    assert [c.code for c in comps] == ["H00300", "H11164", "cbond:composite"]
    assert comps[2].method == "override"
    comp = bm.resolve_component("沪深300指数收益率", overrides={"沪深300指数": "cash:0.02"})
    assert comp.is_cash and comp.rate == pytest.approx(0.02)


def test_hkd_component_warns_and_notes_fx(catalog):
    with pytest.warns(RuntimeWarning, match="港元.*未做汇率换算"):
        comp = bm.resolve_component("中证香港300指数", catalog)
    assert comp.currency == "港元" and bm.FX_CAVEAT in comp.note and comp.return_type == "unknown"


def test_deposit_rates_can_be_overridden():
    comp = bm.resolve_component("一年期定期存款利率(税后)")
    assert comp.code == "cash:time" and comp.rate == pytest.approx(0.015)
    comp = bm.resolve_component("银行活期存款利率", deposit_rate=0.005)
    assert comp.rate == pytest.approx(0.005) and "调用方指定" in comp.note


def test_cash_returns_use_compounding():
    idx = pd.DatetimeIndex(["2024-01-31", "2024-02-29", "2024-03-31"])
    r = bm.cash_returns(0.0035, idx, 12)
    assert r.iloc[0] == pytest.approx(1.0035 ** (1 / 12) - 1) and len(r) == 3


def test_clean_price_cbond_uses_quanjia_indicator(fake, cache):
    r = aks.index_returns("cbond:composite_full", START, END, freq="M", **cache)
    assert len(r) == 2
    assert ("bond_composite_index_cbond", {"indicator": "全价", "period": "总值"}) in fake.calls


# ------------------------------ CLI：--benchmark contract ------------------------------


def _report(tmp_path, *args):
    out = tmp_path / "r.md"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        code = cli.main([
            "report", *args, "--start", START, "--end", END, "--freq", "W", "--rf", "0.018",
            "--cache-dir", str(tmp_path / "cache"), "--out", str(out),
        ])
    return code, (out.read_text(encoding="utf-8") if out.exists() else "")


def test_fund_report_defaults_to_contract_benchmark(fake, tmp_path):
    code, text = _report(tmp_path, "--fund", "110020")
    assert code == 0
    assert "| 基金全称 | 易方达沪深300交易型开放式指数发起式证券投资基金联接基金 |" in text
    assert "| 基金类型 | 指数型-股票 |" in text and f"| 合同业绩比较基准 | {TEXTS['110020']} |" in text
    assert "## 基准解析" in text and "| 沪深300指数 | 95% | H00300 | 全收益 |" in text
    assert "cash:demand" in text and "常数年化 0.35%" in text
    assert "管理费 0.15%/年，托管费 0.05%/年" in text and "已含于费用后净值，不再扣减" in text
    assert "| 基准收益类型 | 全收益 |" in text
    # 指数型基金：结论先报告跟踪误差与跟踪偏离
    obs = text.split("观察到的表现：")[1]
    assert obs.index("跟踪误差") < obs.index("组合累计收益") and "年化跟踪偏离" in text
    assert fake.count("index_csindex_all") == 0  # 不需要目录时不联网取


def test_fund_report_resolution_failure_is_one_line_error(fake, tmp_path, capsys):
    code, _ = _report(tmp_path, "--fund", "110011")
    err = capsys.readouterr().err
    assert code == 2 and "Traceback" not in err
    assert "中债总指数" in err and '--benchmark-map "中债总指数=' in err


def test_fund_report_with_benchmark_map_notes_fx_and_qdii(fake, tmp_path):
    code, text = _report(tmp_path, "--fund", "110011", *MAP_110011)
    assert code == 0
    assert "| 基准币种 | 含非人民币成分：中证香港300指数（H11164，港元）；未做汇率换算，基准收益含汇率差异 |" in text
    notes = text.split("## 附注")[1].split("## 结论")[0]
    assert "未做汇率换算" in notes and "QDII 基金投资境外市场" in notes
    assert "汇率" in text.split("需要进一步验证的判断：")[1]
    assert "| 中债总指数 | 20% | cbond:composite | 全收益 | 人民币 |" in text and "调用方指定" in text


def test_fund_report_000001_uses_catalog_and_price_caveat(fake, tmp_path):
    code, text = _report(tmp_path, "--fund", "000001")
    assert code == 0 and fake.count("index_csindex_all") == 1
    assert "| 中证800成长指数 | 70% | H30355 | 未知 |" in text and "cbond:composite_full" in text
    assert "含价格指数成分（价格指数不含成分股分红" in text


def test_explicit_benchmark_still_works_and_contract_needs_fund(fake, tmp_path, capsys):
    code, text = _report(tmp_path, "--fund", "110011", "--benchmark", "000300")
    assert code == 0 and "H00300" in text and "| 基金类型 | QDII-混合偏股 |" in text and "## 基准解析" not in text
    code = cli.main(["report", "--input", "x.csv", "--benchmark", "contract"])
    assert code == 2 and "--fund" in capsys.readouterr().err


def test_benchmark_map_parsing():
    assert cli.parse_benchmark_map("中债总指数=cbond:composite，中证香港300指数=H11164") == {
        "中债总指数": "cbond:composite", "中证香港300指数": "H11164",
    }
    with pytest.raises(ValueError, match="名称=代码"):
        cli.parse_benchmark_map("中债总指数")


# ------------------------------ compare ------------------------------


def _compare(**kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return compare(["110020", "110011", "000001"], START, END, freq="W", **kwargs)


def test_compare_continues_after_single_failure(fake, cache):
    res = _compare(rf="0.018", **cache)
    t = res.table
    assert list(t["基金代码"]) == ["110020", "110011", "000001"]  # 默认按输入顺序
    assert set(res.reports) == {"110020", "000001"} and set(res.failures) == {"110011"}
    failed = t.set_index("基金代码").loc["110011"]
    assert "中债总指数" in failed["失败原因"] and np.isnan(failed["Sharpe"])
    assert failed["简称"] == "易方达优质精选混合(QDII)"  # 失败时仍列出概况
    ok = t.set_index("基金代码").loc["110020"]
    assert ok["基准收益类型"] == "全收益" and ok["期数"] == 7 and "样本不足 36 个月" in ok["标注"]
    assert t.set_index("基金代码").loc["000001", "基准收益类型"] == "含价格指数成分"
    text = res.to_markdown()
    assert "不做综合打分" in text and SORT_CAVEATS[0] not in text
    assert "不同类型的基金" in text  # 类型不同时提醒


def test_compare_sort_adds_caveats_and_puts_failures_last(fake, cache):
    res = _compare(rf="0.018", sort="sharpe", **cache)
    t = res.table
    assert t["基金代码"].iloc[-1] == "110011"
    sharpe = t["Sharpe"].iloc[:2].to_numpy()
    assert sharpe[0] >= sharpe[1]
    text = res.to_markdown()
    for caveat in SORT_CAVEATS:
        assert caveat in text


def test_compare_uses_one_risk_free_series(fake, cache):
    res = _compare(rf="shibor3m", benchmark_map={"中债总指数": "cbond:composite"}, **cache)
    assert not res.failures and fake.count("rate_interbank") == 1
    rfs = [rep.data["risk_free"] for rep in res.reports.values()]
    assert all(r.equals(rfs[0]) for r in rfs[1:]) and "Shibor 3M" in res.scope["无风险收益"]


def test_compare_factor_and_style_columns(fake, cache):
    res = _compare(rf="0.018", factors="cn_index_proxy", style="cn_equity", hac_lags=1, **cache)
    t = res.table.set_index("基金代码")
    assert {"多因子 Alpha（算术年化）", "多因子 Alpha t", "风格前两项"} <= set(t.columns)
    assert np.isfinite(t.loc["110020", "多因子 Alpha t"]) and "%" in t.loc["110020", "风格前两项"]


def test_treynor_sort_excludes_near_zero_beta():
    table = pd.DataFrame({
        "基金代码": ["a", "b", "c"], "Treynor": [0.5, 0.2, 0.1], "β": [0.05, 0.9, 1.1], "失败原因": ["", "", ""],
    })
    assert list(_sort(table, "treynor")["基金代码"]) == ["b", "c", "a"]


def test_compare_cli_excel(fake, tmp_path):
    pytest.importorskip("openpyxl")
    out = tmp_path / "compare.xlsx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        code = cli.main([
            "compare", "--funds", "110011,110020", "--start", START, "--end", END, "--freq", "W", "--rf", "0.018",
            "--sort", "sharpe", "--cache-dir", str(tmp_path / "cache"), "--out", str(out),
        ])
    assert code == 0
    sheets = pd.read_excel(out, sheet_name=None, dtype=str)
    assert {"对比", "口径", "110011", "110020", "失败原因"} <= set(sheets)
    assert "中债总指数" in sheets["110011"].to_string() and "中债总指数" in sheets["失败原因"].to_string()
    assert "Sharpe 比率" in sheets["110020"].to_string()
    assert any(SORT_CAVEATS[0] in str(v) for v in sheets["对比"].to_numpy().ravel())
