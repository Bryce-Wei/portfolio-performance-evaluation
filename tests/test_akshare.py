"""akshare 数据源：用 monkeypatch 替换接口，返回 tests/data/akshare/ 中按真实返回格式构造的小样本。

CI 不访问网络；标记为 network 的冒烟测试默认跳过，本地用 ``pytest -m network`` 运行。
"""

import contextlib
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from fundeval.etl import benchmark as bm
from fundeval.etl import quality
from fundeval.etl.returns import price_to_returns
from fundeval.etl.sources import akshare as aks
from fake_akshare import DATA, FakeAkshare

START, END = "2024-01-01", "2024-02-29"
EX_DATE = pd.Timestamp("2024-01-15")
BAD_DATE = pd.Timestamp("2024-02-20")


@pytest.fixture
def fake(monkeypatch):
    ak = FakeAkshare()
    monkeypatch.setattr(aks, "_ak", lambda: ak)
    return ak


@pytest.fixture
def cache(tmp_path):
    return {"cache_dir": tmp_path / "cache"}


def raw_nav():
    df = pd.read_csv(DATA / "fund_open_fund_info_em_110011_单位净值走势.csv", parse_dates=["净值日期"])
    return df.set_index("净值日期")


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(*args, **kwargs)


# ------------------------------ 列名映射 ------------------------------


def test_fund_nav_maps_columns_and_units(fake, cache):
    nav = aks.fund_nav("110011", START, END, **cache)
    assert list(nav.columns) == ["nav", "reported_growth", "dividend"]
    assert isinstance(nav.index, pd.DatetimeIndex) and nav.index.name == "date"
    assert nav.index[0] >= pd.Timestamp(START) and nav.index[-1] <= pd.Timestamp(END)
    raw = raw_nav()
    assert nav["reported_growth"].loc["2024-01-03"] == pytest.approx(raw.loc["2024-01-03", "日增长率"] / 100)
    assert nav.loc[EX_DATE, "dividend"] == pytest.approx(0.05)
    assert nav["dividend"].drop(EX_DATE).eq(0).all()  # 2021 年的分红在区间外，不影响
    assert ("fund_open_fund_info_em", {"symbol": "110011", "indicator": "单位净值走势"}) in fake.calls


def test_index_and_rate_column_mapping(fake, cache):
    r = aks.index_returns("000300", START, END, **cache)
    raw = pd.read_csv(DATA / "index_zh_a_hist_000300.csv", parse_dates=["日期"]).set_index("日期")["收盘"]
    expected = (raw / raw.shift(1) - 1).loc[START:END]
    np.testing.assert_allclose(r.to_numpy(), expected.to_numpy())
    assert r.name == "000300" and r.index[0] == pd.Timestamp("2024-01-01")
    # 向前多取约 40 个自然日，保证区间首日有前收盘
    _, kwargs = next(c for c in fake.calls if c[0] == "index_zh_a_hist")
    assert kwargs["start_date"] == "20231122" and kwargs["end_date"] == "20240229"

    assert aks.index_source("H11001") == "csindex" and aks.index_source("cbond:composite") == "cbond"
    bond = aks.index_returns("H11001", START, END, freq="M", **cache)
    cb = aks.index_returns("cbond:composite", START, END, freq="M", **cache)
    assert list(bond.index) == list(cb.index) == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")]
    assert fake.count("stock_zh_index_hist_csindex") == 1 and fake.count("bond_composite_index_cbond") == 1


def test_risk_free_compound_conversion_uses_rate_known_at_period_start(fake, cache):
    rf = aks.risk_free_returns(START, END, "M", source="shibor3m", **cache)
    rates = pd.read_csv(DATA / "rate_interbank_Shibor人民币_3月.csv", parse_dates=["报告日"]).set_index("报告日")["利率"] / 100
    y_jan = rates.loc[:"2023-12-31"].iloc[-1]  # 1 月使用 12 月最后一个报价
    y_feb = rates.loc[:"2024-01-31"].iloc[-1]
    assert rf.loc["2024-01-31"] == pytest.approx((1 + y_jan) ** (1 / 12) - 1, abs=1e-15)
    assert rf.loc["2024-02-29"] == pytest.approx((1 + y_feb) ** (1 / 12) - 1, abs=1e-15)
    assert rf.name == "risk_free"

    cgb = aks.risk_free_returns(START, END, "M", source="cgb10y", **cache)
    assert len(cgb) == 2 and cgb.notna().all()
    daily = aks.risk_free_returns(START, "2024-01-05", "D", source="shibor3m", **cache)
    assert daily.loc["2024-01-02"] == pytest.approx((1 + rates.loc["2024-01-01"]) ** (1 / 252) - 1)

    # 常数年化利率
    const = aks.risk_free_returns("2024-01-01", "2024-03-31", "M", source=0.02)
    assert list(const.index) == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29"), pd.Timestamp("2024-03-31")]
    assert const.iloc[0] == pytest.approx(1.02 ** (1 / 12) - 1)
    # 按给定索引（如月末交易日）对齐
    idx = pd.DatetimeIndex(["2024-01-31", "2024-02-29"])
    aligned = aks.risk_free_returns(START, END, "M", source="shibor3m", index=idx, **cache)
    np.testing.assert_allclose(aligned.to_numpy(), rf.to_numpy())
    with pytest.raises(ValueError, match="未知的无风险利率来源"):
        aks.risk_free_returns(START, END, "M", source="libor", **cache)


# ------------------------------ 分红与累计净值 ------------------------------


def test_fund_returns_include_dividend_on_ex_date(fake, cache):
    r = quiet(aks.fund_returns, "110011", START, END, **cache)
    raw = raw_nav()["单位净值"]
    prev = raw.shift(1)
    assert r.loc[EX_DATE] == pytest.approx((raw.loc[EX_DATE] + 0.05) / prev.loc[EX_DATE] - 1)
    # 不计分红时除息日收益被低估约 2%
    assert r.loc[EX_DATE] - (raw.loc[EX_DATE] / prev.loc[EX_DATE] - 1) == pytest.approx(0.05 / prev.loc[EX_DATE])
    np.testing.assert_allclose(r.to_numpy(), price_to_returns(raw, pd.Series({EX_DATE: 0.05})).loc[START:END].to_numpy())
    assert r.name == "portfolio"


def test_accumulated_nav_is_not_a_total_return_index(fake, cache):
    total = quiet(aks.fund_returns, "110011", START, END, **cache)
    acc = aks.fund_accumulated_nav("110011", "2023-12-01", END, **cache)
    acc_returns = (acc / acc.shift(1) - 1).loc[START:END]
    # 累计净值 = 单位净值 + 累计分红：分红部分不再随净值涨跌（未再投资），
    # 按它算的收益每天都被压向零，与总收益逐日不同，区间累计收益也不一致
    moving = total.abs() > 1e-4
    assert (acc_returns[moving].abs() < total[moving].abs()).all()
    assert (acc_returns - total).abs().max() > quality.CROSS_CHECK_TOLERANCE
    assert (1 + acc_returns).prod() != pytest.approx((1 + total).prod(), abs=1e-3)


def test_fund_returns_does_not_use_accumulated_nav(fake, cache):
    quiet(aks.fund_returns, "110011", START, END, **cache)
    indicators = {kw["indicator"] for name, kw in fake.calls if name == "fund_open_fund_info_em"}
    assert indicators == {"单位净值走势", "分红送配详情"}


def test_cross_check_flags_mismatch_and_feeds_quality_report(fake, cache):
    with pytest.warns(RuntimeWarning, match="1 个日期"):
        r, check = aks.fund_returns("110011", START, END, return_check=True, **cache)
    assert list(check.index[check["flagged"]]) == [BAD_DATE]
    assert check.loc[BAD_DATE, "difference"] == pytest.approx(-0.003, abs=1e-4)
    assert not check.loc[EX_DATE, "flagged"]  # 分红计入后与日增长率一致
    rep = quality.data_quality_report(r, cross_check=check)
    assert list(rep.cross_check.index) == [BAD_DATE]
    assert any("基点" in i for i in rep.issues())


def test_fund_returns_monthly(fake, cache):
    daily = quiet(aks.fund_returns, "110011", START, END, **cache)
    monthly = quiet(aks.fund_returns, "110011", START, END, freq="M", **cache)
    assert monthly.loc["2024-01-31"] == pytest.approx((1 + daily.loc["2024-01"]).prod() - 1)


def test_dividend_amount_parsing():
    # 真实格式：每 10 份的金额，换算为每份
    assert aks._parse_dividend_amount("每10份派现金9.0000元") == pytest.approx(0.9)
    assert aks._parse_dividend_amount("每10份派现金0.5000元", 10) == pytest.approx(0.05)
    assert aks._parse_dividend_amount("每份派现金0.0500元") == pytest.approx(0.05)  # 旧格式
    assert aks._parse_dividend_amount(9.0, 10) == pytest.approx(0.9)
    assert aks._parse_dividend_amount(0.12, 1) == 0.12
    for bad in ("每10份派现金若干", "派现金9.0000元"):  # 缺金额或缺份数：报错，不猜测
        with pytest.raises(ValueError):
            aks._parse_dividend_amount(bad, 10)
    with pytest.raises(ValueError, match="每多少份"):
        aks._parse_dividend_amount(0.12)


# 110011 在 2021-02-25 除息，每10份派现金9.0000元（本地用 akshare 1.18.97 核实的真实数据）
REAL_EX_DATE = pd.Timestamp("2021-02-25")
REAL_TOTAL = (8.4677 + 0.9) / 9.4782 - 1  # ≈ −1.1658%


def test_real_dividend_per_10_units(fake, cache):
    r, check = aks.fund_returns("110011", "2021-02-25", "2021-02-25", return_check=True, **cache)
    assert r.loc[REAL_EX_DATE] == pytest.approx(REAL_TOTAL, abs=1e-12)
    assert r.loc[REAL_EX_DATE] == pytest.approx(-0.011658, abs=0.5e-6)
    diff = check.loc[REAL_EX_DATE, "difference"]
    assert abs(diff) < 1e-4  # 与接口日增长率 −1.17% 相差不到 1 个基点
    assert not check.loc[REAL_EX_DATE, "flagged"]
    assert aks.fund_nav("110011", "2021-02-25", "2021-02-25", **cache).loc[REAL_EX_DATE, "dividend"] == pytest.approx(0.9)


def test_per_10_amount_mistaken_as_per_unit_is_flagged(monkeypatch, cache):
    # 反例：把“每10份 9 元”误当每份 9 元，当日推算收益约 +84%，交叉核对必须标记
    ak = FakeAkshare()
    wrong = pd.read_csv(DATA / "fund_open_fund_info_em_110011_分红送配详情.csv")
    wrong = wrong.rename(columns={"每10份分红": "每份分红"})
    wrong["每份分红"] = wrong["每份分红"].str.replace("每10份", "每份")
    original = ak.fund_open_fund_info_em
    ak.fund_open_fund_info_em = lambda symbol, indicator, period="成立来": (
        wrong if indicator == "分红送配详情" else original(symbol=symbol, indicator=indicator, period=period)
    )
    monkeypatch.setattr(aks, "_ak", lambda: ak)
    with pytest.warns(RuntimeWarning, match="日增长率"):
        r, check = aks.fund_returns("110011", "2021-02-25", "2021-02-25", return_check=True, **cache)
    assert r.loc[REAL_EX_DATE] == pytest.approx((8.4677 + 9.0) / 9.4782 - 1)
    assert r.loc[REAL_EX_DATE] > 0.84
    assert check.loc[REAL_EX_DATE, "flagged"]


def test_dividend_without_amount_column_raises(monkeypatch, cache):
    ak = FakeAkshare()
    bad = pd.read_csv(DATA / "fund_open_fund_info_em_110011_分红送配详情.csv").rename(columns={"每10份分红": "分红"})
    original = ak.fund_open_fund_info_em
    ak.fund_open_fund_info_em = lambda symbol, indicator, period="成立来": (
        bad if indicator == "分红送配详情" else original(symbol=symbol, indicator=indicator, period=period)
    )
    monkeypatch.setattr(aks, "_ak", lambda: ak)
    with pytest.raises(aks.AkshareInterfaceError, match="每10份分红"):
        aks.fund_nav("110011", START, END, **cache)


# ------------------------------ 缓存 ------------------------------


def test_cache_hit_skips_second_request(fake, cache):
    first = aks.index_returns("000300", START, END, **cache)
    second = aks.index_returns("000300", START, END, **cache)
    assert fake.count("index_zh_a_hist") == 1
    pd.testing.assert_series_equal(first, second)
    path = aks.cache_path(cache["cache_dir"], "index_zh_a_hist", "000300", pd.Timestamp(START) - pd.Timedelta(days=40), END)
    assert path.exists()


def test_cache_refresh_and_disable(fake, cache, tmp_path):
    quiet(aks.fund_returns, "110011", START, END, **cache)
    quiet(aks.fund_returns, "110011", START, END, **cache)
    assert fake.count("fund_open_fund_info_em") == 2  # 净值与分红各请求一次
    quiet(aks.fund_returns, "110011", START, END, refresh=True, **cache)
    assert fake.count("fund_open_fund_info_em") == 4
    before = fake.count("rate_interbank")
    aks.risk_free_returns(START, END, "M", use_cache=False, cache_dir=tmp_path / "unused")
    aks.risk_free_returns(START, END, "M", use_cache=False, cache_dir=tmp_path / "unused")
    assert fake.count("rate_interbank") == before + 2
    assert not (tmp_path / "unused").exists()


def test_different_date_range_is_a_different_cache_key(fake, cache):
    aks.index_returns("000300", START, END, **cache)
    aks.index_returns("000300", "2024-02-01", END, **cache)
    assert fake.count("index_zh_a_hist") == 2


# ------------------------------ 报错 ------------------------------


def test_missing_interface_raises_clear_error(monkeypatch, cache):
    ak = FakeAkshare()
    monkeypatch.setattr(aks, "_ak", lambda: ak)
    monkeypatch.delattr(FakeAkshare, "index_zh_a_hist")
    with pytest.raises(aks.AkshareInterfaceError, match="index_zh_a_hist.*改名"):
        aks.index_returns("000300", START, END, **cache)


def test_renamed_columns_raise_clear_error(monkeypatch, cache):
    ak = FakeAkshare()
    renamed = pd.read_csv(DATA / "rate_interbank_Shibor人民币_3月.csv").rename(columns={"利率": "rate"})
    ak.rate_interbank = lambda **kwargs: renamed
    monkeypatch.setattr(aks, "_ak", lambda: ak)
    with pytest.raises(aks.AkshareInterfaceError, match="缺少列"):
        aks.risk_free_returns(START, END, "M", **cache)


def test_akshare_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "akshare", None)
    with pytest.raises(ImportError, match=r'fundeval\[data\]'):
        aks._ak()


# ------------------------------ 联网冒烟测试（默认跳过） ------------------------------


# 联网冒烟测试按数据源拆开：一个数据源不可达不影响其他项。
NET_START, NET_END = "2024-01-01", "2024-03-31"


@pytest.fixture
def net_kw(tmp_path):
    pytest.importorskip("akshare")
    return {"cache_dir": tmp_path}


def _unreachable_errors():
    """联网测试统一的“数据源不可达”判定：连接类与超时类异常时 skip，其余（列名、格式、数值）仍判失败。

    requests.ConnectionError 含 ConnectTimeout、ProxyError、SSLError；requests.Timeout 含 ReadTimeout；
    内置 ConnectionError 含连接重置与 DataSourceUnavailable；UpstreamTimeout 为总时限超时。
    """
    import requests

    return (requests.exceptions.ConnectionError, requests.exceptions.Timeout, ConnectionError, aks.UpstreamTimeout)


@contextlib.contextmanager
def skip_if_unreachable(what: str):
    try:
        yield
    except _unreachable_errors() as exc:
        pytest.skip(f"{what} 无法连接或超时（{type(exc).__name__}: {exc}），跳过")


@pytest.mark.network
def test_network_fund(net_kw):
    """基金净值与分红（东方财富 fund_open_fund_info_em）。"""
    r, check = aks.fund_returns("110011", NET_START, NET_END, return_check=True, **net_kw)
    assert len(r) > 40 and check["flagged"].mean() < 0.05


@pytest.mark.network
def test_network_index_eastmoney(net_kw):
    """东方财富指数行情 index_zh_a_hist；指定 source="em"，不让自动切换掩盖故障。

    连接类或超时类异常时 skip 并注明原因；列名、格式或数值错误仍然算失败。
    """
    with skip_if_unreachable("东方财富 index_zh_a_hist"):
        r = aks.index_returns("000300", NET_START, NET_END, source="em", **net_kw)
    assert len(r) > 40 and r.attrs["source"] == "em"


@pytest.mark.network
def test_network_index_csindex(net_kw):
    """中证指数官网 stock_zh_index_hist_csindex：全收益指数 H00300 与中证全债 H11001。"""
    for code in ("H00300", "H11001"):
        r = aks.index_returns(code, NET_START, NET_END, source="csindex", **net_kw)
        assert len(r) > 40 and r.attrs["source"] == "csindex", code


@pytest.mark.network
def test_network_chinabond(net_kw):
    """中债综合财富指数（yield.chinabond.com.cn）。

    该网站在部分网络环境下无法建立连接：连接类或超时类异常时 skip 并注明原因；
    数据格式或列名错误（AkshareInterfaceError 等）仍然算失败。
    """
    with skip_if_unreachable("中债网站 yield.chinabond.com.cn"):
        r = aks.index_returns("cbond:composite", NET_START, NET_END, **net_kw)
    assert len(r) > 40 and r.attrs["source"] == "cbond"


@pytest.mark.network
def test_network_shibor(net_kw):
    """Shibor 3M（rate_interbank，一次调用约翻 10 页，按默认的单请求超时应能完成）。

    连接类或超时类异常时 skip 并注明原因；列名、格式或数值错误仍然算失败。
    """
    with skip_if_unreachable("Shibor rate_interbank"):
        rf = aks.risk_free_returns(NET_START, NET_END, "M", source="shibor3m", **net_kw)
    assert rf.notna().all() and rf.attrs["source"] == "shibor3m"


@pytest.mark.network
def test_network_government_bond(net_kw):
    """中国国债收益率（bond_zh_us_rate）：2 年期（自动切换的备选）与 10 年期。"""
    for source in ("cgb2y", "cgb10y"):
        rf = aks.risk_free_returns(NET_START, NET_END, "M", source=source, **net_kw)
        assert rf.notna().all() and rf.attrs["source"] == source, source


@pytest.mark.network
def test_network_total_return_benchmark_removes_dividend_alpha(tmp_path):
    """110020（易方达沪深300ETF联接A）对沪深 300 全收益指数：分红口径一致后 Alpha 接近零。"""
    pytest.importorskip("akshare")
    from fundeval.report import evaluate

    kw = {"cache_dir": tmp_path}
    start, end = "2021-01-01", "2025-12-31"
    fund = aks.fund_returns("110020", start, end, freq="M", **kw)
    bench = aks.index_returns("H00300", start, end, freq="M", **kw)
    rf = aks.risk_free_returns(start, end, "M", source="auto", index=fund.index, **kw)
    rep = evaluate(fund, bench, rf, periods_per_year=12)
    assert rep.n >= 55
    assert abs(rep.capm.annualized_alpha(12)) < 0.01
    assert abs(rep.capm.alpha_t) < 2


@pytest.mark.parametrize(
    "exc",
    [
        __import__("requests").exceptions.ConnectionError("proxy refused"),
        __import__("requests").exceptions.ReadTimeout("Read timed out"),
        ConnectionError("Remote end closed connection without response"),
        aks.UpstreamTimeout("超过总时限"),
    ],
)
def test_network_skip_rule_skips_connection_and_timeout_errors(exc):
    """联网测试的统一判定（离线核对）：连接类、超时类异常转为 skip。"""
    with pytest.raises(pytest.skip.Exception, match="无法连接或超时"):
        with skip_if_unreachable("测试数据源"):
            raise exc


@pytest.mark.parametrize("exc", [aks.AkshareInterfaceError("缺少列 收盘"), KeyError("data"), ValueError("格式")])
def test_network_skip_rule_keeps_format_errors_as_failures(exc):
    with pytest.raises(type(exc)):
        with skip_if_unreachable("测试数据源"):
            raise exc


@pytest.mark.network
def test_network_style_indices_csindex(net_kw):
    """风格指数（中证指数官网）：沪深300成长 / 价值、中证红利的全收益与价格指数，2024 年差异与分红相符；
    中证2000 价格指数可取。数值为 2026-09-27 在中证官网实测的 2024 年收益。"""
    expected = {  # 代码: (价格指数 2024 收益, 全收益 2024 收益)
        "000918": (0.0424, 0.0691),
        "000919": (0.2485, 0.3084),
        "000922": (0.1231, 0.1876),
    }
    kw = dict(freq="A", source="csindex", **net_kw)
    for price_code, (price_ret, total_ret) in expected.items():
        total_code = bm.total_return_code(price_code)
        price = aks.index_returns(price_code, "2024-01-01", "2024-12-31", **kw)
        total = aks.index_returns(total_code, "2024-01-01", "2024-12-31", **kw)
        assert float(price.iloc[-1]) == pytest.approx(price_ret, abs=0.002), price_code
        assert float(total.iloc[-1]) == pytest.approx(total_ret, abs=0.002), total_code
    r = aks.index_returns("932000", NET_START, NET_END, source="csindex", **net_kw)
    assert len(r) > 40


@pytest.mark.network
def test_network_timing_gamma_sensitive_to_2024_09(tmp_path):
    """110020 对 H00300，2021-01 至 2025-12 月度，HAC 滞后 3、t 分布：TM γ 由 2024-09 一期主导。

    本地实测（2026-09-27）：全样本 γ = −0.141（t = −6.22）；剔除 2024-09（基金 +19.28%，指数 +21.11%）
    后 γ = −0.042（t = −2.02）。默认的稳健 z 值阈值 5 不一定把 2024-09 标为异常，因此这里显式剔除该月；
    γ 仍为负且 |t| 未跨过 1.96，按规则不算“显著性改变”，但幅度缩小约七成。
    """
    pytest.importorskip("akshare")
    from fundeval.alpha.robustness import exclusion_sensitivity

    kw = {"cache_dir": tmp_path}
    start, end = "2021-01-01", "2025-12-31"
    fund = aks.fund_returns("110020", start, end, freq="M", **kw)
    bench = aks.index_returns("H00300", start, end, freq="M", **kw)
    rf = aks.risk_free_returns(start, end, "M", source="auto", index=fund.index, **kw)
    idx = fund.index.intersection(bench.index)
    fund, bench, rf = fund[idx], bench[idx], rf.reindex(idx)
    sep = idx[idx.to_period("M") == pd.Period("2024-09", "M")]
    assert len(sep) == 1
    assert float(fund[sep[0]]) == pytest.approx(0.1928, abs=0.002)
    assert float(bench[sep[0]]) == pytest.approx(0.2111, abs=0.002)
    rob = exclusion_sensitivity(fund, bench, rf, sep, hac_lags=3, use_t=True)
    full, trimmed = rob.full["TM"], rob.trimmed["TM"]
    assert full.gamma == pytest.approx(-0.141, abs=0.01) and full.gamma_t == pytest.approx(-6.22, abs=0.3)
    assert trimmed.gamma == pytest.approx(-0.042, abs=0.01) and trimmed.gamma_t == pytest.approx(-2.02, abs=0.3)


@pytest.mark.network
@pytest.mark.parametrize("fund_code", ["110011", "110020"])
def test_network_factor_decomposition_cn_index_proxy(tmp_path, fund_code):
    """110011 与 110020 对 cn_index_proxy 因子（H00300、H00852、H00919、H00918，中证指数官网全收益）的
    多因子回归，2021-01 至 2025-12 月度，HAC 滞后 3、t 分布：能正常运行，因子暴露为有限值，R² 在 (0, 1) 内。
    不断言具体数值。"""
    pytest.importorskip("akshare")
    from fundeval.attribution.factor import factor_decomposition, index_proxy_factors, preset_codes

    kw = {"cache_dir": tmp_path}
    start, end = "2021-01-01", "2025-12-31"
    fund = aks.fund_returns(fund_code, start, end, freq="M", **kw)
    indices = pd.concat(
        {code: aks.index_returns(code, start, end, freq="M", source="csindex", **kw) for code in preset_codes("cn_index_proxy")},
        axis=1,
    ).dropna()
    idx = fund.index.intersection(indices.index)
    rf = aks.risk_free_returns(start, end, "M", source="auto", index=idx, **kw)
    factors = index_proxy_factors(indices.loc[idx], rf)
    d = factor_decomposition(fund[idx], factors, rf, hac_lags=3, use_t=True)
    assert d.n >= 55
    assert np.isfinite(d.betas).all() and np.isfinite(d.regression.tvalues).all()
    assert 0 < d.rsquared < 1
    rec = d.reconciliation(12)
    assert rec["合计"] == pytest.approx(rec["平均超额收益"])
