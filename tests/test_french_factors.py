"""Fama–French 因子库（etl.sources.french）与四因子预设（attribution.factor：cn_index_proxy4、ff3_us、carhart_us）。

离线测试按真实格式（2026-10-01 本地实测）制作 zip 样本，不访问网络；akshare 用 tests/fake_akshare.py 模拟。
联网测试（@pytest.mark.network）核对 French 因子库 202512 与 202608 的数值，以及 H30260 与 000300 的年度收益；
连接类与超时类异常时 skip。
"""

import contextlib
import io
import urllib.error
import warnings
import zipfile

import numpy as np
import pandas as pd
import pytest

from fundeval import cli
from fundeval.attribution.factor import (
    FACTOR_DESCRIPTIONS,
    FACTOR_PRESET_NAMES,
    INDEX_PROXY,
    PROXY_CAVEAT,
    US_FACTOR_CAVEAT,
    US_MARKET,
    factor_preset,
    index_proxy_factors,
    preset_codes,
    us_market_factors,
)
from fundeval.etl.sources import akshare as aks
from fundeval.etl.sources import french
from fundeval.report import evaluate, to_markdown
from fundeval.report import inputs as report_inputs
from fake_akshare import FakeAkshare

# ------------------------------ 真实格式的样本 ------------------------------

FF3_CSV = """This file was created by CMPT_ME_BEME_RETS using the 202608 CRSP database.
The 1-month TBill rate data until 202405 are from Ibbotson and Associates Inc.
The 1-month TBill rate data starting from 202406 are from ICE BofA US 1-Month Treasury Bill Index.

,Mkt-RF,SMB,HML,RF
192607,    2.89,   -2.55,   -2.39,    0.22
192608,    2.64,   -1.14,    3.81,    0.25
202510,    1.20,   -0.50,  -99.99,    0.37
202511,    0.10,    0.20,    0.30,    0.35
202512,   -0.36,   -1.04,    2.40,    0.34
202601,    1.00,    0.50,   -0.50,    0.33
202608,    2.56,    0.34,   -3.54,    0.29

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RF
1927,   29.47,   -2.04,   -4.54,    3.12
2025,   15.00,   -3.00,    4.00,    4.20

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""

MOM_CSV = """This file was created by CMPT_ME_PRIOR_RETS using the 202608 CRSP database.
It contains a momentum factor, constructed from six value-weight portfolios formed using independent sorts on size and prior return of NYSE, AMEX, and NASDAQ stocks.
Missing data are indicated by -99.99 or -999.

,Mom
192701,    0.57
202510,    1.10
202511,   -0.80
202512,   -2.40
202601,    0.60
202608,   -5.70

 Annual Factors: January-December
,Mom
1928,   25.77
2025,  -999

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


def make_zip(text: str, name: str = "F-F_Research_Data_Factors.CSV") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, text.replace("\n", "\r\n"))  # 真实文件为 CRLF 换行
    return buf.getvalue()


@pytest.fixture
def fake_download(monkeypatch):
    """替换 french._download：按数据集返回样本 zip，并记录请求的 URL。"""
    calls = []
    files = {
        french.dataset_url(french.FF3_DATASET): make_zip(FF3_CSV),
        french.dataset_url(french.MOMENTUM_DATASET): make_zip(MOM_CSV, "F-F_Momentum_Factor.CSV"),
    }

    def download(url, timeout):
        calls.append((url, timeout))
        return files[url]

    monkeypatch.setattr(french, "_download", download)
    monkeypatch.setattr(aks, "_sleep", lambda s: None)
    return calls


# ------------------------------ 解析 ------------------------------


def test_parse_monthly_units_month_end_and_missing(fake_download, tmp_path):
    f = french.french_factors(french.FF3_DATASET, cache_dir=tmp_path)
    assert list(f.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert isinstance(f.index, pd.DatetimeIndex) and f.index.name == "date"
    # 月末索引：202512 → 2025-12-31，192602 不存在
    assert f.index[0] == pd.Timestamp("1926-07-31") and pd.Timestamp("2025-12-31") in f.index
    assert all(d == d + pd.offsets.MonthEnd(0) for d in f.index)
    # 百分数 → 小数
    row = f.loc["2025-12-31"]
    assert row["Mkt-RF"] == pytest.approx(-0.0036) and row["SMB"] == pytest.approx(-0.0104)
    assert row["HML"] == pytest.approx(0.0240) and row["RF"] == pytest.approx(0.0034)
    # −99.99 → NaN，不填补
    assert np.isnan(f.loc["2025-10-31", "HML"]) and f.loc["2025-10-31", "Mkt-RF"] == pytest.approx(0.012)
    # 年度段被排除：只有 7 个月度行，没有 1927 或 2025 年度值
    assert len(f) == 7 and pd.Timestamp("1927-12-31") not in f.index
    assert f.attrs["currency"] == "USD" and f.attrs["freq"] == "M"
    assert fake_download[0][1] == aks.DEFAULT_TIMEOUT  # 单请求超时沿用默认 30 秒


def test_parse_annual_section(fake_download, tmp_path):
    a = french.french_factors(french.FF3_DATASET, "A", cache_dir=tmp_path)
    assert list(a.index) == [pd.Timestamp("1927-12-31"), pd.Timestamp("2025-12-31")]
    assert a.loc["1927-12-31", "Mkt-RF"] == pytest.approx(0.2947)
    mom = french.french_factors(french.MOMENTUM_DATASET, "A", cache_dir=tmp_path)
    assert np.isnan(mom.loc["2025-12-31", "Mom"])  # −999 → NaN


def test_parse_header_with_trailing_spaces_and_bad_formats():
    df = french.parse_french_csv(MOM_CSV)
    assert list(df.columns) == ["freq", "date", "Mom"]
    assert set(df["freq"]) == {"M", "A"}
    with pytest.raises(ValueError, match="没有找到月度数据段"):
        french.parse_french_csv("说明\n\nCopyright\n")
    with pytest.raises(ValueError, match="2 个数值"):
        french.parse_french_csv(",A\n202501, 1.0, 2.0\n")
    with pytest.raises(ValueError, match="不是 zip"):
        french.read_zip_csv(b"<html>blocked</html>")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a.csv", "x")
        zf.writestr("b.csv", "y")
    with pytest.raises(ValueError, match="只有一个 CSV"):
        french.read_zip_csv(buf.getvalue())


def test_freq_validation(fake_download, tmp_path):
    with pytest.raises(ValueError, match="只提供月度"):
        french.french_factors(french.FF3_DATASET, "D", cache_dir=tmp_path)


def test_carhart_factors_merge_and_columns(fake_download, tmp_path):
    c = french.carhart_factors(cache_dir=tmp_path)
    assert list(c.columns) == ["MKT", "SMB", "HML", "UMD", "RF"]
    # 交集：Mom 从 192701 开始，所以 1926 年的两个月不在结果中
    assert c.index[0] == pd.Timestamp("2025-10-31") and len(c) == 5
    assert c.loc["2025-12-31", "UMD"] == pytest.approx(-0.024)
    assert c.loc["2026-08-31", "MKT"] == pytest.approx(0.0256) and c.loc["2026-08-31", "UMD"] == pytest.approx(-0.057)
    assert np.isnan(c.loc["2025-10-31", "HML"])  # 缺失原样保留
    assert c.attrs["currency"] == "USD"


def test_cache_7_days_and_no_cache(fake_download, tmp_path):
    french.french_factors(french.FF3_DATASET, cache_dir=tmp_path)
    french.french_factors(french.FF3_DATASET, cache_dir=tmp_path)
    assert len(fake_download) == 1  # 第二次读缓存
    assert french.FRENCH_MAX_AGE == pd.Timedelta(days=7)
    french.french_factors(french.FF3_DATASET, cache_dir=tmp_path, use_cache=False)
    assert len(fake_download) == 2
    cached = list(tmp_path.glob("french__*.csv"))
    assert len(cached) == 1


def test_network_errors_are_retried(monkeypatch, tmp_path):
    attempts = []

    def flaky(url, timeout):
        attempts.append(url)
        if len(attempts) < 3:
            raise ConnectionError("Remote end closed connection")
        return make_zip(FF3_CSV)

    monkeypatch.setattr(french, "_download", flaky)
    monkeypatch.setattr(aks, "_sleep", lambda s: None)
    f = french.french_factors(french.FF3_DATASET, cache_dir=tmp_path)
    assert len(attempts) == 3 and len(f) == 7


def test_download_maps_http_errors(monkeypatch):
    def raiser(code):
        def urlopen(request, timeout=None):
            raise urllib.error.HTTPError(request.full_url, code, "x", {}, None)
        return urlopen

    monkeypatch.setattr(french.urllib.request, "urlopen", raiser(403))
    with pytest.raises(ConnectionError, match="HTTP 403"):
        french._download("https://example.invalid/x.zip", 5)
    monkeypatch.setattr(french.urllib.request, "urlopen", raiser(404))
    with pytest.raises(ValueError, match="HTTP 404"):
        french._download("https://example.invalid/x.zip", 5)

    def unreachable(request, timeout=None):
        raise urllib.error.URLError(OSError("Name or service not known"))

    monkeypatch.setattr(french.urllib.request, "urlopen", unreachable)
    with pytest.raises(ConnectionError):
        french._download("https://example.invalid/x.zip", 5)


# ------------------------------ 预设 ------------------------------


def test_cn_index_proxy4_preset_and_column_order():
    spec = factor_preset("cn_index_proxy4")
    assert list(spec) == ["MKT", "SMB", "HML", "UMD"]
    assert spec["UMD"] == ("H30260", "000300")
    # 前三个因子与 cn_index_proxy 相同
    assert {k: spec[k] for k in ("MKT", "SMB", "HML")} == factor_preset("cn_index_proxy")
    assert preset_codes("cn_index_proxy4") == ["H00300", "H00852", "H00919", "H00918", "H30260", "000300"]
    assert FACTOR_PRESET_NAMES == ("cn_index_proxy", "cn_index_proxy4", "ff3_us", "carhart_us")
    assert "价格指数" in FACTOR_DESCRIPTIONS["cn_index_proxy4"]["UMD"]


def test_cn_index_proxy4_construction():
    idx = pd.date_range("2024-01-31", periods=4, freq="ME")
    data = pd.DataFrame(
        {
            "H00300": [0.01, 0.02, -0.01, 0.03], "H00852": [0.02, 0.01, 0.0, 0.05],
            "H00919": [0.0, 0.01, 0.02, 0.01], "H00918": [0.01, 0.03, -0.02, 0.02],
            "H30260": [0.015, 0.01, -0.03, 0.04], "000300": [0.008, 0.018, -0.012, 0.028],
        },
        index=idx,
    )
    f = index_proxy_factors(data, 0.001, "cn_index_proxy4")
    assert list(f.columns) == ["MKT", "SMB", "HML", "UMD"]
    assert f["UMD"].to_numpy() == pytest.approx((data["H30260"] - data["000300"]).to_numpy())
    assert f["MKT"].to_numpy() == pytest.approx((data["H00300"] - 0.001).to_numpy())
    assert f.attrs["factor_type"] == INDEX_PROXY
    with pytest.raises(ValueError, match="缺少指数：H30260、000300"):
        index_proxy_factors(data.drop(columns=["H30260", "000300"]), 0.001, "cn_index_proxy4")


def test_us_market_factors_column_mapping(fake_download, tmp_path):
    lib = french.carhart_factors(cache_dir=tmp_path).loc["2025-11-30":]
    f3, rf = us_market_factors(lib, "ff3_us")
    assert list(f3.columns) == ["MKT", "SMB", "HML"] and f3.attrs["factor_type"] == US_MARKET
    assert f3.loc["2025-12-31", "MKT"] == pytest.approx(-0.0036)  # MKT 即 Mkt-RF，不再减 RF
    assert rf.loc["2025-12-31"] == pytest.approx(0.0034)
    f4, _ = us_market_factors(lib, "carhart_us")
    assert list(f4.columns) == ["MKT", "SMB", "HML", "UMD"]
    assert f4.loc["2025-12-31", "UMD"] == pytest.approx(-0.024)
    with pytest.raises(ValueError, match="缺失"):
        us_market_factors(french.carhart_factors(cache_dir=tmp_path), "ff3_us")  # 2025-10 的 HML 缺失
    with pytest.raises(KeyError):
        us_market_factors(lib, "cn_index_proxy")


def test_evaluate_with_us_factors_uses_factor_rf_and_caveat():
    idx = pd.date_range("2021-01-31", periods=48, freq="ME")
    rng = np.random.default_rng(11)
    lib = pd.DataFrame(rng.normal(0.005, 0.04, (48, 4)), index=idx, columns=["MKT", "SMB", "HML", "UMD"])
    lib["RF"] = 0.003
    factors, rf = us_market_factors(lib, "carhart_us")
    p = pd.Series(0.003 + 0.001 + factors.to_numpy() @ [1.0, 0.2, -0.1, 0.3] + rng.normal(0, 0.002, 48), index=idx)
    rep = evaluate(p, None, 0.0015, 12, factor_returns=factors, factor_risk_free=rf,
                   labels={"factor_risk_free": "French 因子库的 RF"})
    assert rep.factor.caveat == US_FACTOR_CAVEAT
    assert rep.factor.alpha == pytest.approx(0.001, abs=0.001)
    assert rep.factor.betas["MKT"] == pytest.approx(1.0, abs=0.05)
    assert rep.scope["因子类型"] == US_FACTOR_CAVEAT and rep.scope["因子回归无风险收益"] == "French 因子库的 RF"
    assert "美元计价" in rep.conclusion() and PROXY_CAVEAT not in rep.conclusion()
    assert "美国市场因子（美元）" in to_markdown(rep)
    # 因子已是超额收益，只有组合的超额收益随无风险收益变化：报告无风险收益 0.0015 与因子 RF 0.003 相差
    # 0.0015，两次回归的 Alpha 恰好相差 0.0015，说明第一次用的是因子 RF
    rep2 = evaluate(p, None, 0.0015, 12, factor_returns=factors)
    assert rep2.factor.alpha - rep.factor.alpha == pytest.approx(0.0015, abs=1e-12)
    with pytest.raises(ValueError, match="因子回归的无风险收益未覆盖"):
        evaluate(p, None, 0.0015, 12, factor_returns=factors, factor_risk_free=rf.iloc[2:])


# ------------------------------ CLI 与报告输入 ------------------------------


def test_cli_cn_index_proxy4_fetches_price_legs(monkeypatch, tmp_path):
    """UMD 的两条腿按原代码取数：H30260 与 000300（价格指数），不被全收益替换成 H00300。"""
    fake = FakeAkshare()
    monkeypatch.setattr(aks, "_ak", lambda: fake)
    monkeypatch.setattr(aks, "_sleep", lambda s: None)
    out = tmp_path / "r.md"
    args = ["report", "--fund", "110011", "--start", "2024-01-01", "--end", "2024-02-29", "--benchmark", "000300",
            "--freq", "D", "--rf", "0.018", "--factors", "cn_index_proxy4", "--index-return-type", "total",
            "--cache-dir", str(tmp_path / "cache"), "--out", str(out)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert cli.main(args) == 0
    assert ("stock_zh_index_hist_csindex", {"symbol": "H30260"}) in fake.calls
    em = [kw["symbol"] for name, kw in fake.calls if name == "index_zh_a_hist"]
    assert em == ["000300"]  # 000300 只为 UMD 取一次价格指数（基准已换成 H00300）
    text = out.read_text(encoding="utf-8")
    assert "UMD = 沪深300动量 H30260 − 沪深300 000300（均为价格指数，分红口径一致）" in text
    assert "| 因子类型 | 因子为指数代理因子" in text and "因子暴露：UMD" in text


def test_cli_carhart_us(monkeypatch, tmp_path, fake_download):
    """--factors carhart_us：从 French 因子库取因子与 RF，口径写明美国市场与美元计价。"""
    idx = pd.date_range("2024-01-31", periods=24, freq="ME")
    rng = np.random.default_rng(3)
    lib = pd.DataFrame(rng.normal(0.004, 0.04, (24, 4)), index=idx, columns=["MKT", "SMB", "HML", "UMD"])
    lib["RF"] = 0.004
    lib.attrs["source_label"] = "Kenneth R. French Data Library（测试）"
    seen = {}

    def carhart(freq="M", **kw):
        seen.update(freq=freq, **kw)
        return lib

    monkeypatch.setattr(french, "carhart_factors", carhart)
    p = pd.Series(0.004 + lib[["MKT", "SMB", "HML", "UMD"]].to_numpy() @ [0.9, 0.1, 0.0, 0.2]
                  + rng.normal(0, 0.003, 24), index=idx, name="portfolio")
    path = tmp_path / "r.csv"
    pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "portfolio": p.to_numpy()}).to_csv(path, index=False)
    out = tmp_path / "r.md"
    assert cli.main(["report", "--input", str(path), "--factors", "carhart_us", "--rf", "0.02",
                     "--cache-dir", str(tmp_path / "c"), "--out", str(out)]) == 0
    assert seen["freq"] == "M" and seen["cache_dir"] == str(tmp_path / "c")
    text = out.read_text(encoding="utf-8")
    assert "| 因子 | carhart_us：MKT = Mkt-RF（美国市场超额收益）" in text
    assert "UMD = Mom（动量" in text and "以美元计价" in text
    assert "| 因子回归无风险收益 | French 因子库的 RF" in text
    assert "人民币兑美元汇率" in text


def test_us_factors_require_monthly(fake_download, tmp_path):
    with pytest.raises(ValueError, match="只支持 --freq M"):
        report_inputs._factors_from_french("ff3_us", None, None, "Q", {"cache_dir": tmp_path}, pd.DatetimeIndex([]))


def test_cli_rejects_unknown_factor_preset():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["report", "--input", "x.csv", "--factors", "ff5_us"])


# ------------------------------ 联网测试 ------------------------------


@contextlib.contextmanager
def skip_if_unreachable(what: str):
    """连接类与超时类异常时 skip；格式与数值错误仍判失败。"""
    errors = [ConnectionError, TimeoutError]
    try:
        import requests
    except ImportError:  # pragma: no cover
        pass
    else:
        errors += [requests.exceptions.ConnectionError, requests.exceptions.Timeout]
    try:
        yield
    except tuple(errors) as exc:
        pytest.skip(f"{what} 无法连接或超时（{type(exc).__name__}: {exc}），跳过")


@pytest.mark.network
def test_network_french_values(tmp_path):
    """French 因子库 2026-10-01 本地实测值：202512 与 202608。"""
    with skip_if_unreachable("French 因子库 F-F_Research_Data_Factors / F-F_Momentum_Factor"):
        c = french.carhart_factors(cache_dir=tmp_path)
    expected = {
        "2025-12-31": {"MKT": -0.36, "SMB": -1.04, "HML": 2.40, "RF": 0.34, "UMD": -2.40},
        "2026-08-31": {"MKT": 2.56, "SMB": 0.34, "HML": -3.54, "RF": 0.29, "UMD": -5.70},
    }
    for date, values in expected.items():
        for col, pct in values.items():
            assert c.loc[date, col] == pytest.approx(pct / 100, abs=1e-9), (date, col)


@pytest.mark.network
def test_network_h30260_vs_000300_annual_returns(tmp_path):
    """H30260（沪深300动量）与 000300 的年度收益，与 2026-10-01 本地实测一致（容差 0.05 个百分点）。"""
    pytest.importorskip("akshare")
    expected = {
        "H30260": [-9.63, -25.47, -11.45, 14.56, 21.52],
        "000300": [-5.20, -21.63, -11.38, 14.68, 17.66],
    }
    for code, values in expected.items():
        with skip_if_unreachable(f"中证指数官网 {code}"):
            r = aks.index_returns(code, "2021-01-01", "2025-12-31", source="csindex", freq="A", cache_dir=tmp_path)
        assert list(r.index.year) == [2021, 2022, 2023, 2024, 2025]
        assert (r.to_numpy() * 100) == pytest.approx(values, abs=0.05), code
