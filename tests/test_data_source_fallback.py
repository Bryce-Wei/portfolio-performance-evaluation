"""真实数据端到端运行暴露的问题：全收益基准、重试与自动切换、缓存覆盖、CLI 网络报错。

akshare 行为一律用 tests/fake_akshare.py 模拟，不访问网络。
"""

import os
import time
import warnings

import pandas as pd
import pytest
import requests

from fundeval import cli
from fundeval.etl import benchmark as bm
from fundeval.etl.sources import akshare as aks
from fake_akshare import FakeAkshare

START, END = "2024-01-01", "2024-02-29"
EM_DOWN = ConnectionError("Remote end closed connection without response")
SHIBOR_DOWN = requests.exceptions.ChunkedEncodingError("Connection broken: IncompleteRead")


@pytest.fixture
def sleeps(monkeypatch):
    waited = []
    monkeypatch.setattr(aks, "_sleep", waited.append)
    return waited


def use(monkeypatch, fake):
    monkeypatch.setattr(aks, "_ak", lambda: fake)
    return fake


@pytest.fixture
def cache(tmp_path):
    return {"cache_dir": tmp_path / "cache"}


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(*args, **kwargs)


# ------------------------------ 一、全收益指数 ------------------------------


def test_total_return_codes_and_return_types():
    expected = {"000300": "H00300", "000905": "H00905", "000906": "H00906", "000852": "H00852", "000016": "H00016"}
    for price, total in expected.items():
        assert bm.total_return_code(price) == total
        assert bm.total_return_code(total) == total  # 本身已是全收益指数
        assert bm.index_return_type(price) == "price" and bm.index_return_type(total) == "total"
    assert bm.total_return_code("cbond:composite") == "cbond:composite"
    assert bm.index_return_type("cbond:new_composite") == "total"
    # 中证债券指数口径未核实：标 unknown，不给全收益代码
    assert bm.index_return_type("H11001") == "unknown" and bm.total_return_code("H11001") is None
    assert bm.total_return_code("399006") is None and bm.index_return_type("399006") == "price"
    assert bm.total_return_code("999999") is None and bm.index_return_type("999999") == "unknown"
    rec = bm.index_record("H00300")
    assert (rec.name, rec.price_code, rec.source) == ("沪深300", "000300", "em")
    # 名称对照保持不变
    assert bm.lookup_index_code("沪深300指数") == "000300" and bm.lookup_index_code("中债综合") == "cbond:composite"


def test_benchmark_return_type_labels():
    assert bm.benchmark_return_type(["H00300", "cbond:composite"]) == "全收益"
    assert bm.benchmark_return_type(["000300", "000905"]) == "价格指数"
    assert bm.benchmark_return_type(["000300", "cbond:composite"]) == "含价格指数成分"
    assert bm.benchmark_return_type(["H00300", "H11001"]) == "未知"
    assert not bm.needs_price_caveat("全收益")
    assert all(bm.needs_price_caveat(x) for x in ("价格指数", "含价格指数成分", "未知"))


def test_resolve_benchmark_codes_warns_when_no_total_index():
    parts = [("000300", 0.8), ("H11001", 0.2)]
    with pytest.warns(RuntimeWarning, match="H11001 没有对应的全收益指数"):
        codes, notes = cli.resolve_benchmark_codes(parts, "total")
    assert codes == [("H00300", 0.8), ("H11001", 0.2)] and len(notes) == 1
    assert cli.resolve_benchmark_codes(parts, "price") == (parts, [])


def _fund_report(monkeypatch, tmp_path, *extra, fake=None):
    fake = use(monkeypatch, fake or FakeAkshare())
    out = tmp_path / "fund.md"
    args = [
        "report", "--fund", "110011", "--start", START, "--end", END, "--freq", "W",
        "--cache-dir", str(tmp_path / "cache"), "--out", str(out), *extra,
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        code = cli.main(args)
    return code, (out.read_text(encoding="utf-8") if out.exists() else ""), fake, caught


def test_cli_uses_total_return_index_by_default(monkeypatch, tmp_path, sleeps):
    code, text, fake, _ = _fund_report(monkeypatch, tmp_path, "--benchmark", "000300", "--rf", "0.018")
    assert code == 0
    assert ("stock_zh_index_hist_csindex", {"symbol": "H00300"}) in fake.calls
    assert fake.count("index_zh_a_hist") == 0
    assert "000300（实际使用 H00300）" in text
    assert "| 基准收益类型 | 全收益 |" in text
    assert "价格指数不含成分股分红" not in text


def test_cli_price_index_caveat_in_scope_notes_and_conclusion(monkeypatch, tmp_path, sleeps):
    code, text, fake, _ = _fund_report(
        monkeypatch, tmp_path, "--benchmark", "000300", "--rf", "0.018", "--index-return-type", "price"
    )
    assert code == 0 and fake.count("index_zh_a_hist") == 1
    caveat = bm.PRICE_INDEX_CAVEAT
    assert f"| 基准收益类型 | 价格指数（{caveat}） |" in text
    notes = text.split("## 附注")[1].split("## 结论")[0]
    assert caveat in notes
    paragraphs = text.split("## 结论")[1].split("## 未完成的检验")[0].strip().split("\n\n")
    assert paragraphs[1].startswith("可以支持的解释") and caveat in paragraphs[1]


def test_evaluate_benchmark_return_type_label(worked_example):
    from fundeval.report import conclusion, evaluate

    rep = evaluate(worked_example, labels={"benchmark_return_type": "含价格指数成分"})
    assert rep.scope["基准收益类型"].startswith("含价格指数成分（价格指数不含成分股分红")
    assert bm.PRICE_INDEX_CAVEAT in conclusion(rep).split("\n\n")[1]
    rep = evaluate(worked_example, labels={"benchmark_return_type": "全收益"})
    assert rep.scope["基准收益类型"] == "全收益" and bm.PRICE_INDEX_CAVEAT not in conclusion(rep)
    assert evaluate(worked_example).scope["基准收益类型"] == "未说明"


# ------------------------------ 二、重试与自动切换 ------------------------------


def test_index_auto_falls_back_to_csindex_after_retries(monkeypatch, cache, sleeps):
    fake = use(monkeypatch, FakeAkshare(fail={"index_zh_a_hist": EM_DOWN}))
    with aks.collect_notes() as notes:
        with pytest.warns(RuntimeWarning, match="已改用中证指数官网"):
            r = aks.index_returns("000300", START, END, **cache)
    assert fake.count("index_zh_a_hist") == aks.MAX_ATTEMPTS == 3
    assert sleeps == [1.0, 2.0]  # 指数退避
    assert fake.count("stock_zh_index_hist_csindex") == 1
    assert r.attrs["source"] == "csindex" and "中证指数官网" in r.attrs["source_label"]
    assert len(notes) == 1 and "000300" in notes[0] and "ConnectionError" in notes[0]
    assert r.notna().all() and len(r) > 30


def test_index_auto_routes_csindex_codes_directly(monkeypatch, cache, sleeps):
    fake = use(monkeypatch, FakeAkshare(fail={"index_zh_a_hist": EM_DOWN}))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # 不发生切换
        r = aks.index_returns("H00300", START, END, **cache)
    assert fake.count("index_zh_a_hist") == 0 and r.attrs["source"] == "csindex"


def test_explicit_source_does_not_fall_back(monkeypatch, cache, sleeps):
    fake = use(monkeypatch, FakeAkshare(fail={"index_zh_a_hist": EM_DOWN}))
    with pytest.raises(ConnectionError):
        aks.index_returns("000300", START, END, source="em", **cache)
    assert fake.count("index_zh_a_hist") == 3 and fake.count("stock_zh_index_hist_csindex") == 0


def test_non_network_errors_are_not_retried(monkeypatch, cache, sleeps):
    fake = use(monkeypatch, FakeAkshare(fail={"index_zh_a_hist": KeyError("data")}))
    with pytest.raises(KeyError):
        aks.index_returns("000300", START, END, **cache)
    assert fake.count("index_zh_a_hist") == 1 and sleeps == []
    assert fake.count("stock_zh_index_hist_csindex") == 0


def test_retry_recovers_from_transient_failure(monkeypatch, cache, sleeps):
    fake = use(monkeypatch, FakeAkshare())
    original = fake.index_zh_a_hist
    state = {"n": 0}

    def flaky(**kwargs):
        state["n"] += 1
        if state["n"] == 1:
            raise requests.exceptions.ConnectionError("reset")
        return original(**kwargs)

    fake.index_zh_a_hist = flaky
    r = aks.index_returns("000300", START, END, **cache)
    assert state["n"] == 2 and sleeps == [1.0] and r.attrs["source"] == "em"


def test_risk_free_auto_falls_back_to_cgb2y(monkeypatch, cache, sleeps):
    fake = use(monkeypatch, FakeAkshare(fail={"rate_interbank": SHIBOR_DOWN}))
    with aks.collect_notes() as notes:
        with pytest.warns(RuntimeWarning, match="已改用中国国债 2 年期收益率"):
            rf = aks.risk_free_returns(START, END, "M", source="auto", **cache)
    assert fake.count("rate_interbank") == 3 and fake.count("bond_zh_us_rate") == 1
    assert rf.attrs["source"] == "cgb2y" and rf.attrs["failed"] == ["shibor3m"]
    assert "中国国债 2 年期收益率（Shibor 3M 获取失败后改用）" in rf.attrs["description"]
    assert len(notes) == 1 and "ChunkedEncodingError" in notes[0]
    expected = aks.risk_free_returns(START, END, "M", source="cgb2y", **cache)
    pd.testing.assert_series_equal(rf, expected)


def test_risk_free_auto_all_fail_raises_and_suggests_constant(monkeypatch, cache, sleeps):
    use(monkeypatch, FakeAkshare(fail={"rate_interbank": SHIBOR_DOWN, "bond_zh_us_rate": TimeoutError("timed out")}))
    with pytest.raises(aks.DataSourceUnavailable, match=r"--rf 0\.018"):
        aks.risk_free_returns(START, END, "M", source="auto", **cache)


def test_risk_free_named_source_does_not_fall_back(monkeypatch, cache, sleeps):
    fake = use(monkeypatch, FakeAkshare(fail={"rate_interbank": SHIBOR_DOWN}))
    with pytest.raises(requests.RequestException):
        aks.risk_free_returns(START, END, "M", source="shibor3m", **cache)
    assert fake.count("bond_zh_us_rate") == 0


def test_cli_records_fallback_sources_in_report(monkeypatch, tmp_path, sleeps):
    fake = FakeAkshare(fail={"index_zh_a_hist": EM_DOWN, "rate_interbank": SHIBOR_DOWN})
    code, text, fake, caught = _fund_report(
        monkeypatch, tmp_path, "--benchmark", "000300:0.8,H11001:0.2", "--index-return-type", "price", fake=fake
    )
    assert code == 0
    assert "| 基准数据源 | 000300：中证指数官网 stock_zh_index_hist_csindex；H11001：中证指数官网" in text
    assert "中国国债 2 年期收益率（Shibor 3M 获取失败后改用）" in text
    notes = text.split("## 附注")[1].split("## 结论")[0]
    assert "东方财富 index_zh_a_hist 请求失败" in notes and "已改用中国国债 2 年期收益率" in notes
    messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any("已改用中证指数官网" in m for m in messages) and any("已改用中国国债" in m for m in messages)


# ------------------------------ 三、CLI 网络错误提示 ------------------------------


def test_cli_network_error_prints_one_chinese_line(monkeypatch, tmp_path, capsys, sleeps):
    use(monkeypatch, FakeAkshare(fail={"index_zh_a_hist": EM_DOWN}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        code = cli.main([
            "report", "--fund", "110011", "--benchmark", "000300", "--index-source", "em", "--rf", "0.018",
            "--start", START, "--end", END, "--cache-dir", str(tmp_path / "cache"),
        ])
    err = capsys.readouterr().err
    assert code == 2
    assert "Traceback" not in err and len(err.strip().splitlines()) == 1
    assert err.startswith("错误：网络请求失败")
    for hint in ("--index-source csindex", "--rf 0.018", "稍后重试", "--refresh"):
        assert hint in err


def test_cli_risk_free_failure_is_a_network_error(monkeypatch, tmp_path, capsys, sleeps):
    use(monkeypatch, FakeAkshare(fail={"rate_interbank": SHIBOR_DOWN, "bond_zh_us_rate": EM_DOWN}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        code = cli.main(["report", "--fund", "110011", "--start", START, "--end", END, "--cache-dir", str(tmp_path / "c")])
    err = capsys.readouterr().err
    assert code == 2 and "Traceback" not in err and "网络请求失败" in err and "--rf 0.018" in err


# ------------------------------ 四、缓存覆盖与回退 ------------------------------


def test_cache_refetches_when_coverage_is_short(monkeypatch, cache, sleeps):
    fake = use(monkeypatch, FakeAkshare())
    monkeypatch.setattr(aks, "_today", lambda: pd.Timestamp("2024-06-28"))
    quiet(aks.fund_returns, "110011", START, END, **cache)
    quiet(aks.fund_returns, "110011", START, END, **cache)
    assert fake.count("fund_open_fund_info_em") == 2  # 覆盖到 END：命中缓存（净值与分红各一次）
    # 缓存数据截至 2024-02-29，请求到 2024-03-31 时覆盖不足，重新拉取净值
    quiet(aks.fund_returns, "110011", START, "2024-03-31", **cache)
    assert sum(1 for n, kw in fake.calls if kw.get("indicator") == "单位净值走势") == 2
    # 允许 cache_lag_days 的滞后：放宽到 40 天时不再重新拉取
    quiet(aks.fund_returns, "110011", START, "2024-03-31", cache_lag_days=40, **cache)
    assert sum(1 for n, kw in fake.calls if kw.get("indicator") == "单位净值走势") == 2


def test_cache_end_none_uses_coverage_and_file_age(monkeypatch, cache, sleeps):
    fake = use(monkeypatch, FakeAkshare())
    monkeypatch.setattr(aks, "_today", lambda: pd.Timestamp("2024-03-01"))
    aks.rate_series("shibor3m", START, None, **cache)
    aks.rate_series("shibor3m", START, None, **cache)
    assert fake.count("rate_interbank") == 1  # 覆盖到 2024-03-01 前 7 天，且文件是新的
    path = aks.cache_path(cache["cache_dir"], "rate_interbank-3月", "shibor3m", None, None)
    old = time.time() - 2 * 86400
    os.utime(path, (old, old))
    aks.rate_series("shibor3m", START, None, **cache)
    assert fake.count("rate_interbank") == 2  # 超过 cache_max_age（1 天）
    aks.rate_series("shibor3m", START, None, **cache)
    assert fake.count("rate_interbank") == 2
    monkeypatch.setattr(aks, "_today", lambda: pd.Timestamp("2024-06-03"))
    aks.rate_series("shibor3m", START, None, **cache)
    assert fake.count("rate_interbank") == 3  # 缓存数据未覆盖到今天之前的最后一个工作日


def test_failed_refetch_falls_back_to_old_cache_with_warning(monkeypatch, cache, sleeps):
    fake = use(monkeypatch, FakeAkshare())
    monkeypatch.setattr(aks, "_today", lambda: pd.Timestamp("2024-06-28"))
    first = quiet(aks.fund_returns, "110011", START, END, **cache)
    fake.fail["fund_open_fund_info_em"] = requests.exceptions.ConnectionError("reset by peer")
    path = aks.cache_path(cache["cache_dir"], "fund_open_fund_info_em-单位净值走势", "110011", None, None)
    stamp = pd.Timestamp.fromtimestamp(path.stat().st_mtime)
    with aks.collect_notes() as notes:
        with pytest.warns(RuntimeWarning, match=f"使用 {stamp:%Y-%m-%d} 的缓存数据"):
            again = aks.fund_returns("110011", START, "2024-03-31", **cache)
    pd.testing.assert_series_equal(first, again)
    assert any(f"使用 {stamp:%Y-%m-%d} 的缓存数据" in n and "未覆盖到" in n for n in notes)
    # 没有旧缓存时照常报错
    with pytest.raises(requests.RequestException):
        aks.fund_returns("110011", START, END, use_cache=False)


def test_refetch_failure_without_network_cause_is_not_masked(monkeypatch, cache, sleeps):
    fake = use(monkeypatch, FakeAkshare())
    monkeypatch.setattr(aks, "_today", lambda: pd.Timestamp("2024-06-28"))
    quiet(aks.fund_returns, "110011", START, END, **cache)
    fake.fail["fund_open_fund_info_em"] = KeyError("data")
    with pytest.raises(KeyError):
        quiet(aks.fund_returns, "110011", START, "2024-03-31", **cache)


def test_cli_fallback_to_stale_cache_is_noted_in_report(monkeypatch, tmp_path, sleeps):
    monkeypatch.setattr(aks, "_today", lambda: pd.Timestamp("2024-06-28"))
    fake = FakeAkshare()
    _fund_report(monkeypatch, tmp_path, "--rf", "0.018", fake=fake)
    fake.fail["fund_open_fund_info_em"] = requests.exceptions.ConnectionError("reset by peer")
    code, text, _, _ = _fund_report(monkeypatch, tmp_path, "--rf", "0.018", "--refresh", fake=fake)
    assert code == 0
    notes = text.split("## 附注")[1].split("## 结论")[0]
    assert "的缓存数据" in notes and "重新拉取失败" in notes


# ------------------------------ 请求超时 ------------------------------


@pytest.fixture
def hanging(monkeypatch):
    fakes = []

    def make(**kwargs):
        fake = use(monkeypatch, FakeAkshare(**kwargs))
        fakes.append(fake)
        return fake

    yield make
    for fake in fakes:
        fake.release()


def test_hanging_request_times_out_and_is_retried(hanging, cache, sleeps):
    fake = hanging(hang={"fund_open_fund_info_em": 30})
    t0 = time.monotonic()
    with pytest.raises(aks.UpstreamTimeout, match="超过 0.2 秒未返回"):
        aks.fund_returns("110011", START, END, timeout=0.2, **cache)
    assert time.monotonic() - t0 < 5
    assert fake.count("fund_open_fund_info_em") == 3 and sleeps == [1.0, 2.0]
    assert aks.is_network_error(aks.UpstreamTimeout("x"))


def test_hanging_index_falls_back_to_csindex(hanging, cache, sleeps):
    fake = hanging(hang={"index_zh_a_hist": 30})
    with pytest.warns(RuntimeWarning, match="UpstreamTimeout.*已改用中证指数官网"):
        r = aks.index_returns("000300", START, END, timeout=0.2, **cache)
    assert r.attrs["source"] == "csindex" and fake.count("index_zh_a_hist") == 3


def test_timeout_none_and_invalid(monkeypatch, cache):
    use(monkeypatch, FakeAkshare())
    assert len(aks.index_returns("000300", START, END, timeout=None, **cache)) > 30
    with pytest.raises(ValueError, match="timeout"):
        aks.index_returns("000300", START, END, timeout=0, use_cache=False)


def test_requests_default_timeout_is_injected_only_during_calls(monkeypatch):
    seen = {}
    original = requests.Session.request

    def fake_request(self, method, url, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        return "ok"

    monkeypatch.setattr(requests.Session, "request", fake_request)
    assert aks._call_with_timeout(lambda: requests.get("https://example.invalid"), 7.5, "test") == "ok"
    assert seen["timeout"] == 7.5
    # 调用方已给 timeout 时不覆盖
    aks._call_with_timeout(lambda: requests.get("https://example.invalid", timeout=3), 7.5, "test")
    assert seen["timeout"] == 3
    # 调用结束后恢复原方法
    assert requests.Session.request is fake_request and fake_request is not original


def test_cli_timeout_gives_friendly_error(hanging, tmp_path, capsys, sleeps):
    hanging(hang={"fund_open_fund_info_em": 30})
    t0 = time.monotonic()
    code = cli.main([
        "report", "--fund", "110011", "--rf", "0.018", "--start", START, "--end", END,
        "--timeout", "0.2", "--cache-dir", str(tmp_path / "c"),
    ])
    err = capsys.readouterr().err
    assert code == 2 and time.monotonic() - t0 < 5
    assert "Traceback" not in err and "网络请求失败" in err and "超过 0.2 秒未返回" in err
