"""外币基准成分的汇率换算、报告图表、--style auto 与中债取数失败时的替代提示。

测试不联网：akshare 用 tests/fake_akshare.py 模拟（currency_boc_safe 样本见 tests/data/akshare/README.md）；
联网测试标 @pytest.mark.network，连接类异常时 skip。
"""

import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from fundeval import cli
from fundeval.etl import benchmark as bm
from fundeval.etl import fx
from fundeval.etl.sources import akshare as aks
from fundeval.report import evaluate, inputs, to_excel, to_markdown
from fundeval.report.compare import compare
from fake_akshare import FakeAkshare
from test_akshare import skip_if_unreachable

START, END = "2024-01-01", "2024-02-29"
MAP_110011 = ["--benchmark-map", "中债总指数=cbond:composite"]
HKD_CONVERTED = "港元指数按国家外汇管理局人民币汇率中间价换算为人民币收益"


@pytest.fixture
def fake(monkeypatch):
    ak = FakeAkshare()
    monkeypatch.setattr(aks, "_ak", lambda: ak)
    monkeypatch.setattr(aks, "_sleep", lambda s: None)
    return ak


@pytest.fixture
def cache(tmp_path):
    return {"cache_dir": tmp_path / "cache"}


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(*args, **kwargs)


def _cli(tmp_path, *args, out="r.md"):
    path = tmp_path / out
    code = quiet(cli.main, [
        "report", "--start", START, "--end", END, "--freq", "W", "--rf", "0.018",
        "--cache-dir", str(tmp_path / "cache"), "--out", str(path), *args,  # 后给的参数覆盖默认（如 --freq D）
    ])
    return code, (path.read_text(encoding="utf-8") if path.exists() and path.suffix == ".md" else "")


# ------------------------------ 一、汇率换算：公式与 asof ------------------------------


def test_convert_returns_formula():
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
    local = pd.Series([0.01, -0.02, 0.005], index=idx)
    rates = pd.Series([0.90, 0.91, 0.905, 0.92], index=pd.DatetimeIndex(["2024-01-01", *idx]))
    out = fx.convert_returns(local, rates, base_date="2024-01-01")
    expected = [1.01 * 0.91 / 0.90 - 1, 0.98 * 0.905 / 0.91 - 1, 1.005 * 0.92 / 0.905 - 1]
    assert out.to_numpy() == pytest.approx(expected)
    # 不给 base_date：第一个观测没有上一个交易日，为 NaN（不填补）
    assert np.isnan(fx.convert_returns(local, rates).iloc[0])


def test_convert_returns_telescopes_to_price_level_conversion():
    """日度换算后连乘，等于按人民币计的价格首尾之比：P_T S_T / (P_0 S_0)。"""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-01", periods=30)
    price = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 30)), index=idx)
    rates = pd.Series(0.9 + rng.normal(0, 0.002, 30), index=idx)
    local = price.pct_change().dropna()
    out = fx.convert_returns(local, rates, base_date=idx[0])
    assert float(np.prod(1 + out) - 1) == pytest.approx(price.iloc[-1] * rates.iloc[-1] / (price.iloc[0] * rates.iloc[0]) - 1)


def test_asof_uses_latest_rate_not_later_than_date():
    """指数交易日没有中间价（如内地休市、香港交易）时用此前最近一个中间价，绝不用之后的。"""
    rates = pd.Series([0.90, 0.95], index=pd.DatetimeIndex(["2024-01-02", "2024-01-05"]))
    got = fx.fx_asof(rates, pd.DatetimeIndex(["2024-01-01", "2024-01-03", "2024-01-04", "2024-01-05"]))
    assert np.isnan(got.iloc[0])  # 之前没有中间价：NaN，不用 01-02 的“未来”值
    assert got.iloc[1:].tolist() == [0.90, 0.90, 0.95]

    idx = pd.DatetimeIndex(["2024-01-03", "2024-01-04", "2024-01-05"])
    local = pd.Series([0.0, 0.0, 0.0], index=idx)
    out = fx.convert_returns(local, rates, base_date="2024-01-02")
    # 01-03、01-04 两天中间价都取 01-02 的 0.90，汇率贡献为 0；01-05 才反映 0.95 / 0.90
    assert out.to_numpy() == pytest.approx([0.0, 0.0, 0.95 / 0.90 - 1])


def test_future_rate_would_change_result_but_is_not_used():
    idx = pd.DatetimeIndex(["2024-01-03", "2024-01-04"])
    local = pd.Series([0.01, 0.01], index=idx)
    past = pd.Series([0.90], index=pd.DatetimeIndex(["2024-01-02"]))
    with_future = pd.concat([past, pd.Series([2.0], index=pd.DatetimeIndex(["2024-01-05"]))])
    a = fx.convert_returns(local, past, base_date="2024-01-02")
    b = fx.convert_returns(local, with_future, base_date="2024-01-02")
    assert a.equals(b)


def test_fx_mode_is_validated():
    with pytest.raises(ValueError, match="--fx 须为 convert 或 none"):
        fx.check_mode("auto")


# ------------------------------ 一、汇率换算：fx_rates ------------------------------


def test_fx_rates_per_unit_and_cache(fake, cache):
    s = aks.fx_rates("港元", "2021-01-01", "2025-12-31", **cache)
    assert s.loc["2021-01-04"] == pytest.approx(0.84363) and s.loc["2025-12-31"] == pytest.approx(0.90322)
    assert s.attrs["source_label"] == "国家外汇管理局人民币汇率中间价" and s.name == "港元"
    aks.fx_rates("美元", end="2025-12-31", **cache)  # 同一张全表，1 天内不重复请求
    assert fake.count("currency_boc_safe") == 1


def test_fx_rates_drops_unpublished_days(fake, cache):
    s = aks.fx_rates("新西兰元", "2023-10-01", "2023-10-31", **cache)
    assert not s.isna().any() and s.index[0] > pd.Timestamp("2023-10-11")


def test_fx_rates_missing_currency_column_raises(fake, cache):
    with pytest.raises(ValueError, match="没有“卢布”列.*可选币种：美元、欧元.*不做猜测"):
        aks.fx_rates("卢布", **cache)


def test_contract_conversion_matches_price_level_method(fake, cache):
    """H11164 的人民币周收益 = 按人民币计的收盘价（收盘价 × asof 中间价）的周度变化。"""
    got = inputs._fx_converted_returns("H11164", "港元", START, END, "W", cache)
    price = aks.index_prices("H11164", START, END, **cache)
    rates = aks.fx_rates("港元", **cache)
    cny = price * rates.asof(price.index).to_numpy()
    weekly = cny.resample("W-FRI").last()
    expected = (weekly / weekly.shift(1) - 1).loc[START:]
    # 首周：上期末取 START 之前最后一个收盘
    first_prev = cny.loc[: pd.Timestamp(START) - pd.Timedelta(days=1)].iloc[-1]
    expected.iloc[0] = weekly.iloc[weekly.index.get_loc(expected.index[0])] / first_prev - 1
    assert got.to_numpy() == pytest.approx(expected.reindex(got.index).to_numpy())
    local = aks.index_returns("H11164", START, END, freq="W", **cache)
    assert not np.allclose(got.to_numpy(), local.reindex(got.index).to_numpy())  # 确实换算了


def test_conversion_without_rate_before_start_raises(fake, cache):
    rates = pd.Series([0.9], index=pd.DatetimeIndex(["2024-02-01"]))
    fake_rates = lambda *a, **k: rates  # noqa: E731
    import fundeval.etl.sources.akshare as mod

    orig = mod.fx_rates
    mod.fx_rates = fake_rates
    try:
        with pytest.raises(ValueError, match="没有港元的国家外汇管理局人民币汇率中间价.*--fx none"):
            inputs._fx_converted_returns("H11164", "港元", START, END, "W", cache)
    finally:
        mod.fx_rates = orig


def test_unknown_currency_column_error_mentions_fx_none(fake, cache):
    with pytest.raises(ValueError, match="无法换算为人民币收益.*没有“卢布”列.*--fx none"):
        inputs._fx_converted_returns("H11164", "卢布", START, END, "W", cache)


# ------------------------------ 一、汇率换算：CLI 与 compare ------------------------------


def test_cli_default_converts_and_states_basis(fake, tmp_path):
    code, text = _cli(tmp_path, "--fund", "110011", *MAP_110011)
    assert code == 0
    assert f"| 基准币种 | 含非人民币成分：中证香港300指数（H11164，港元）；{HKD_CONVERTED} |" in text
    assert "未做汇率换算" not in text
    assert "按国家外汇管理局人民币汇率中间价（港元）换算为人民币" in text  # 基准数据源
    assert fake.count("currency_boc_safe") == 1


def test_cli_fx_none_keeps_caveat_and_differs(fake, tmp_path):
    code, text = _cli(tmp_path, "--fund", "110011", *MAP_110011, "--fx", "none")
    assert code == 0 and bm.FX_CAVEAT in text and HKD_CONVERTED not in text
    assert fake.count("currency_boc_safe") == 0


def test_benchmark_series_differs_between_modes(fake, cache):
    kw = dict(fund="110011", start=START, end=END, freq="W", rf="0.018", benchmark_map={"中债总指数": "cbond:composite"}, **cache)
    conv = quiet(inputs.build_report_inputs, inputs.ReportOptions(**kw))
    none = quiet(inputs.build_report_inputs, inputs.ReportOptions(**kw, fx="none"))
    hk_c = inputs._fx_converted_returns("H11164", "港元", START, END, "W", cache)
    hk_n = aks.index_returns("H11164", START, END, freq="W", **cache)
    idx = conv["benchmark"].index
    diff = (conv["benchmark"] - none["benchmark"]).to_numpy()
    assert diff == pytest.approx(0.3 * (hk_c.reindex(idx) - hk_n.reindex(idx)).to_numpy())


def test_fx_fetch_failure_is_an_error_with_fx_none_hint(monkeypatch, tmp_path, capsys):
    ak = FakeAkshare(fail={"currency_boc_safe": ConnectionError("Remote end closed connection")})
    monkeypatch.setattr(aks, "_ak", lambda: ak)
    monkeypatch.setattr(aks, "_sleep", lambda s: None)
    code, _ = _cli(tmp_path, "--fund", "110011", *MAP_110011)
    err = capsys.readouterr().err
    assert code == 2 and "Traceback" not in err
    assert "汇率中间价（港元）取数失败" in err and "--fx none" in err
    assert ak.count("currency_boc_safe") == aks.MAX_ATTEMPTS  # 按网络异常重试，不静默跳过


def test_explicit_benchmark_code_looks_up_currency(fake, tmp_path):
    code, text = _cli(tmp_path, "--fund", "110011", "--benchmark", "000300:0.7,H11164:0.3")
    assert code == 0 and fake.count("index_csindex_all") == 1
    assert f"含非人民币成分：H11164（H11164，港元）；{HKD_CONVERTED}" in text
    # 指数表内的代码不查目录
    fake.calls.clear()
    code, _ = _cli(tmp_path, "--fund", "110020", "--benchmark", "000300")
    assert code == 0 and fake.count("index_csindex_all") == 0


def test_resolve_component_convert_note_has_no_warning(fake, cache):
    catalog = aks.index_catalog(**cache)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        comp = bm.resolve_component("中证香港300指数", catalog, fx="convert")
    assert HKD_CONVERTED in comp.note and bm.FX_CAVEAT not in comp.note


def test_compare_scope_states_fx_mode(fake, cache):
    res = quiet(compare, ["110011"], START, END, freq="W", rf="0.018",
                benchmark_map={"中债总指数": "cbond:composite"}, **cache)
    assert res.scope["汇率换算"].startswith("convert") and not res.failures
    rep = res.reports["110011"]
    assert HKD_CONVERTED in rep.scope["基准币种"]
    res = quiet(compare, ["110011"], START, END, freq="W", rf="0.018", fx="none",
                benchmark_map={"中债总指数": "cbond:composite"}, **cache)
    assert "未做汇率换算" in res.scope["汇率换算"] and bm.FX_CAVEAT in res.reports["110011"].scope["基准币种"]


# ------------------------------ 三、--style auto ------------------------------


@pytest.mark.parametrize(
    "fund_type, preset, keyword",
    [
        ("QDII-混合偏股", "cn_equity", "偏股"),  # 同时含“混合”与“偏股”：先查限定词
        ("混合型-偏股", "cn_equity", "偏股"),
        ("混合型-偏债", "cn_balanced", "偏债"),
        ("混合型-灵活", "cn_balanced", "混合"),
        ("债券型-长债", "cn_balanced", "债券"),
        ("FOF-稳健型", "cn_balanced", "FOF"),
        ("fof-均衡型", "cn_balanced", "FOF"),
        ("指数型-股票", "cn_equity", "股票"),
        ("股票型", "cn_equity", "股票"),
    ],
)
def test_auto_style_preset_rules(fund_type, preset, keyword):
    got, basis = bm.auto_style_preset(fund_type)
    assert got == preset and f"含“{keyword}”" in basis and fund_type in basis


@pytest.mark.parametrize("fund_type", ["货币型", "Reits", "", None])
def test_auto_style_preset_fallback(fund_type):
    got, basis = bm.auto_style_preset(fund_type)
    assert got == "cn_balanced" and "无法判断" in basis and "H11001" in basis


def test_cli_style_auto_picks_balanced_for_000001(fake, tmp_path):
    code, text = _cli(tmp_path, "--fund", "000001", "--style", "auto", "--freq", "D")
    assert code == 0
    assert "| 风格预设 | cn_balanced（--style auto：基金类型“混合型-灵活”含“混合”，选用 cn_balanced（含中证全债 H11001）） |" in text
    assert "中证全债：H11001" in text and "风格预设按基金类型自动选择" in text


def test_cli_style_auto_picks_equity_for_qdii_equity(fake, tmp_path):
    code, text = _cli(tmp_path, "--fund", "110011", *MAP_110011, "--style", "auto", "--freq", "D")
    assert code == 0 and "基金类型“QDII-混合偏股”含“偏股”，选用 cn_equity" in text
    style_line = next(line for line in text.splitlines() if line.startswith("| 风格指数 |"))
    assert "H00918" in style_line and "H11001" not in style_line


def test_style_auto_requires_fund(capsys, tmp_path):
    code = cli.main(["report", "--input", str(tmp_path / "x.csv"), "--style", "auto"])
    assert code == 2 and "--style auto 须与 --fund 一起使用" in capsys.readouterr().err


def test_compare_style_auto_column(fake, cache):
    res = quiet(compare, ["110020", "000001"], START, END, freq="D", rf="0.018", style="auto", **cache)
    t = res.table.set_index("基金代码")
    assert t.loc["110020", "风格预设"].startswith("cn_equity") and t.loc["000001", "风格预设"].startswith("cn_balanced")
    assert res.scope["风格指数"].startswith("auto")


# ------------------------------ 四、中债取数失败时的替代提示 ------------------------------


def test_cbond_failure_suggests_benchmark_map_alternative(monkeypatch, tmp_path, capsys):
    ak = FakeAkshare(fail={"bond_composite_index_cbond": ConnectionError("Connection refused")})
    monkeypatch.setattr(aks, "_ak", lambda: ak)
    monkeypatch.setattr(aks, "_sleep", lambda s: None)
    code, _ = _cli(tmp_path, "--fund", "000001")
    err = capsys.readouterr().err
    assert code == 2 and "Traceback" not in err
    assert "Connection refused" in err  # 原有内容保留
    assert '--benchmark-map "中债-综合全价(总值)=H11001"' in err
    assert "改变基准口径" in err and "调用方指定" in err and "不会自动替换" in err
    assert ak.count("stock_zh_index_hist_csindex") >= 1 and "H11001" not in str(
        [c for c in ak.calls if c[0] == "stock_zh_index_hist_csindex"]
    )  # 没有自动改取 H11001


def test_cbond_alternative_via_benchmark_map_is_labeled(fake, tmp_path):
    code, text = _cli(tmp_path, "--fund", "000001", "--benchmark-map", "中债-综合全价(总值)=H11001")
    assert code == 0
    assert "| 中债-综合全价(总值)指数 | 30% | H11001 |" in text and "调用方指定" in text
    assert fake.count("bond_composite_index_cbond") == 0


def test_non_network_cbond_error_is_unchanged(monkeypatch, tmp_path, capsys):
    ak = FakeAkshare(fail={"bond_composite_index_cbond": KeyError("date")})
    monkeypatch.setattr(aks, "_ak", lambda: ak)
    code, _ = _cli(tmp_path, "--fund", "000001")
    err = capsys.readouterr().err
    assert code == 2 and "--benchmark-map" not in err


# ------------------------------ 二、报告图表 ------------------------------


def _monthly_report(n=36, style=True):
    rng = np.random.default_rng(7)
    idx = pd.date_range("2021-01-31", periods=n, freq="ME")
    b = pd.Series(rng.normal(0.006, 0.04, n), index=idx)
    p = pd.Series(0.9 * b + rng.normal(0.001, 0.01, n), index=idx)
    kw = {}
    if style:
        styles = pd.DataFrame({
            "沪深300成长": b + rng.normal(0, 0.01, n), "沪深300价值": b + rng.normal(0, 0.01, n), "现金": 0.001,
        }, index=idx)
        kw = dict(style_returns=styles, style_window=24,
                  labels={"style": "沪深300成长：H00918（全收益）；沪深300价值：H00919（全收益）；现金：无风险收益（与报告无风险收益同口径）"})
    return quiet(evaluate, p, b, 0.001, 12, **kw)


@pytest.fixture
def mpl():
    return pytest.importorskip("matplotlib")


def test_report_charts_files_exist(mpl, tmp_path):
    from fundeval.report.charts import report_charts

    res = quiet(report_charts, _monthly_report(), tmp_path / "c")
    assert list(res.paths) == ["wealth", "drawdown", "rolling", "style"]
    for path in res.paths.values():
        assert path.exists() and path.stat().st_size > 1000


def test_rolling_active_definition():
    rep = _monthly_report(style=False)
    from fundeval.report.charts import rolling_active

    roll = rolling_active(rep.data, 12)
    p, b = rep.data["portfolio"], rep.data["benchmark"]
    last = p.iloc[-12:], b.iloc[-12:]
    assert roll["excess"].iloc[-1] == pytest.approx(np.prod(1 + last[0]) - np.prod(1 + last[1]))
    assert roll["te"].iloc[-1] == pytest.approx((last[0] - last[1]).std(ddof=1) * np.sqrt(12))
    assert roll["excess"].iloc[:11].isna().all()


def test_short_sample_skips_rolling(mpl, tmp_path, worked_example):
    from fundeval.report.charts import report_charts

    rep = quiet(evaluate, worked_example.iloc[:10], periods_per_year=12)
    res = quiet(report_charts, rep, tmp_path)
    assert "rolling" not in res.paths and "少于滚动窗口" in res.skipped["rolling"]


def test_markdown_embeds_relative_paths(mpl, tmp_path):
    out = tmp_path / "sub" / "我的 报告.md"
    out.parent.mkdir()
    text = quiet(to_markdown, _monthly_report(), out, charts=True)
    files = tmp_path / "sub" / "我的 报告_files"
    assert (files / "wealth.png").stat().st_size > 0
    assert "## 图表" in text
    assert "](%E6%88%91%E7%9A%84%20%E6%8A%A5%E5%91%8A_files/wealth.png)" in text  # 相对路径，按 URL 规则转义
    assert str(tmp_path) not in text


def test_markdown_charts_need_path(mpl):
    with pytest.raises(ValueError, match="须给出 path"):
        to_markdown(_monthly_report(style=False), charts=True)


def test_excel_chart_sheet(mpl, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    out = quiet(to_excel, _monthly_report(), tmp_path / "r.xlsx", charts=True)
    wb = openpyxl.load_workbook(out)
    assert "图表" in wb.sheetnames and len(wb["图表"]._images) == 4
    assert wb["图表"]["A1"].value == "财富指数（期初 = 1）"


def test_no_cjk_font_falls_back_to_english(mpl, monkeypatch, tmp_path):
    from fundeval.report import charts

    monkeypatch.setattr(charts, "find_cjk_font", lambda: None)
    with pytest.warns(RuntimeWarning, match="未找到中文字体.*英文标签"):
        res = charts.report_charts(_monthly_report(), tmp_path)
    assert res.lang == "en" and res.paths["style"].stat().st_size > 0
    fig = charts.wealth_figure(_monthly_report(style=False).data, "en")
    assert fig.axes[0].get_title(loc="left") == "Wealth index (start = 1)"
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == ["Portfolio", "Benchmark"]
    rep = _monthly_report()
    fig = charts.style_figure(rep.style_rolling.weights, 24, "en", charts._style_codes(rep))
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == ["H00918", "H00919", "cash"]


def test_cjk_font_is_used_when_available(mpl, monkeypatch):
    from matplotlib import font_manager

    from fundeval.report import charts

    fake_font = type("F", (), {"name": "Noto Sans CJK SC"})()
    monkeypatch.setattr(font_manager.fontManager, "ttflist", [fake_font])
    assert charts.find_cjk_font() == "Noto Sans CJK SC" and charts.chart_language() == ("zh", "Noto Sans CJK SC")


def test_compare_charts_markdown(mpl, fake, cache, tmp_path):
    res = quiet(compare, ["110020", "000001"], START, END, freq="W", rf="0.018", **cache)
    text = quiet(res.to_markdown, tmp_path / "cmp.md", charts=True)
    assert "](cmp_files/compare.png)" in text and (tmp_path / "cmp_files" / "compare.png").stat().st_size > 0


def test_cli_charts_flag(mpl, fake, tmp_path):
    code, text = _cli(tmp_path, "--fund", "110020", "--charts")
    assert code == 0 and "](r_files/wealth.png)" in text and (tmp_path / "r_files" / "drawdown.png").exists()


def test_charts_without_matplotlib_gives_install_hint(monkeypatch, capsys, tmp_path):
    from fundeval.report import charts

    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(ImportError, match=r'pip install "fundeval\[plot\]"'):
        charts.require_matplotlib()
    code = cli.main(["report", "--input", str(tmp_path / "x.csv"), "--charts", "--out", str(tmp_path / "r.md")])
    assert code == 2 and "fundeval[plot]" in capsys.readouterr().err
    # 不加 --charts 时不受影响
    assert "matplotlib" not in cli.__dict__


def test_charts_need_out(capsys, tmp_path):
    code = cli.main(["report", "--input", str(tmp_path / "x.csv"), "--charts"])
    assert code == 2 and "--charts 须与 --out 一起使用" in capsys.readouterr().err


# ------------------------------ 联网 ------------------------------


@pytest.mark.network
def test_network_fx_rates_hkd(tmp_path):
    """国家外汇管理局人民币汇率中间价（currency_boc_safe）：港元 2021-01-04 约 0.84363，2025-12-31 约 0.90322。"""
    pytest.importorskip("akshare")
    with skip_if_unreachable("国家外汇管理局 currency_boc_safe"):
        s = aks.fx_rates("港元", "2021-01-01", "2025-12-31", cache_dir=tmp_path)
    assert s.loc["2021-01-04"] == pytest.approx(0.84363, abs=1e-4)
    assert s.loc["2025-12-31"] == pytest.approx(0.90322, abs=1e-4)
