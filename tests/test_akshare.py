"""akshare 数据源：用 monkeypatch 替换接口，返回 tests/data/akshare/ 中按真实返回格式构造的小样本。

CI 不访问网络；标记为 network 的冒烟测试默认跳过，本地用 ``pytest -m network`` 运行。
"""

import sys
import warnings

import numpy as np
import pandas as pd
import pytest

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
    assert aks._parse_dividend_amount("每份派现金0.0500元") == 0.05
    assert aks._parse_dividend_amount(0.12) == 0.12
    with pytest.raises(ValueError):
        aks._parse_dividend_amount("每10份派现金若干")


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


@pytest.mark.network
def test_network_smoke(tmp_path):
    pytest.importorskip("akshare")
    kw = {"cache_dir": tmp_path}
    r, check = aks.fund_returns("110011", "2024-01-01", "2024-03-31", return_check=True, **kw)
    assert len(r) > 40 and check["flagged"].mean() < 0.05
    assert len(aks.index_returns("000300", "2024-01-01", "2024-03-31", **kw)) > 40
    assert len(aks.index_returns("H11001", "2024-01-01", "2024-03-31", **kw)) > 40
    assert len(aks.index_returns("cbond:composite", "2024-01-01", "2024-03-31", **kw)) > 40
    assert aks.risk_free_returns("2024-01-01", "2024-03-31", "M", source="shibor3m", **kw).notna().all()
    assert aks.risk_free_returns("2024-01-01", "2024-03-31", "M", source="cgb10y", **kw).notna().all()


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
