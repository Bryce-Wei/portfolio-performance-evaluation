"""测试用的 akshare 替身：按接口签名返回 tests/data/akshare/ 中的样本，并记录调用。

``fail`` 把接口名映射到要抛出的异常（实例或类），用于模拟上游请求中途断开，例如
``FakeAkshare(fail={"index_zh_a_hist": ConnectionError("Remote end closed connection")})``；
失败的调用同样记入 ``calls``，便于核对重试次数。

``hang`` 把接口名映射到挂起秒数，模拟上游请求迟迟不返回（akshare 内部 requests 未设超时）；
测试结束时调用 ``release()`` 让挂起的后台线程立即退出。

``pages`` 把接口名映射到每页耗时（秒）的列表，模拟需要翻页的接口（如 rate_interbank 约 10 页）：
每页经 requests.get 发出一个真实的请求对象，由 ``patch_http`` 安装的假传输层按耗时响应。
假传输层遵守请求的 timeout：耗时不小于读取超时时，等待满超时后抛出 requests.ReadTimeout。
值也可以是“每次调用一个列表”的列表，第 k 次调用取第 k 个（超出时取最后一个），
例如 ``[[30], [0.01]]`` 表示首次调用的请求挂起、之后的调用立即返回。
"""

import threading
import types
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).parent / "data" / "akshare"


class FakeAkshare(types.SimpleNamespace):
    """按接口签名返回样本，并记录调用次数。"""

    def __init__(self, fail=None, hang=None, pages=None):
        super().__init__(
            __version__="fake", calls=[], fail=dict(fail or {}), hang=dict(hang or {}), pages=dict(pages or {}),
            http=[], _released=threading.Event(),
        )

    def release(self):
        self._released.set()

    def _log(self, name, **kwargs):
        self.calls.append((name, kwargs))
        if name in self.hang:
            self._released.wait(self.hang[name])
        if name in self.fail:
            exc = self.fail[name]
            raise exc() if isinstance(exc, type) else exc
        if name in self.pages:
            delays = self.pages[name]
            if delays and isinstance(delays[0], (list, tuple)):
                delays = delays[min(self.count(name), len(delays)) - 1]
            for i, delay in enumerate(delays, 1):
                if self._released.is_set():  # 测试已结束：被总时限放弃的后台线程不再发请求
                    break
                # 与 akshare 一样不传 timeout
                requests.get(
                    f"http://fake-akshare.invalid/{name}?page={i}",
                    headers={"X-Fake-Delay": str(delay), "X-Fake-Id": str(id(self))},
                )

    def fund_open_fund_info_em(self, symbol="710001", indicator="单位净值走势", period="成立来"):
        self._log("fund_open_fund_info_em", symbol=symbol, indicator=indicator)
        return pd.read_csv(DATA / f"fund_open_fund_info_em_{symbol}_{indicator}.csv")

    def index_zh_a_hist(self, symbol="000859", period="daily", start_date="19700101", end_date="22220101"):
        self._log("index_zh_a_hist", symbol=symbol, start_date=start_date, end_date=end_date)
        return pd.read_csv(DATA / f"index_zh_a_hist_{symbol}.csv", dtype={"日期": str})

    def stock_zh_index_hist_csindex(self, symbol="000928", start_date="20180526", end_date="20240604"):
        self._log("stock_zh_index_hist_csindex", symbol=symbol)
        return pd.read_csv(DATA / f"stock_zh_index_hist_csindex_{symbol}.csv")

    def bond_composite_index_cbond(self, indicator="财富", period="总值"):
        self._log("bond_composite_index_cbond", indicator=indicator, period=period)
        return pd.read_csv(DATA / f"bond_composite_index_cbond_{indicator}_{period}.csv")

    def fund_overview_em(self, symbol="015641"):
        self._log("fund_overview_em", symbol=symbol)
        path = DATA / f"fund_overview_em_{symbol}.csv"
        if not path.exists():  # 与真实接口一样：查无此基金时返回空表
            return pd.DataFrame([])
        return pd.read_csv(path, dtype=str)

    def index_csindex_all(self):
        self._log("index_csindex_all")
        return pd.read_csv(DATA / "index_csindex_all.csv", dtype={"指数代码": str})

    def rate_interbank(self, market="上海银行同业拆借市场", symbol="Shibor人民币", indicator="隔夜"):
        self._log("rate_interbank", market=market, symbol=symbol, indicator=indicator)
        return pd.read_csv(DATA / f"rate_interbank_{symbol}_{indicator}.csv")

    def bond_zh_us_rate(self, start_date="19901219"):
        self._log("bond_zh_us_rate", start_date=start_date)
        return pd.read_csv(DATA / "bond_zh_us_rate.csv")

    def count(self, name):
        return sum(1 for n, _ in self.calls if n == name)


def patch_http(monkeypatch, fake):
    """把 requests 的传输层替换为按 X-Fake-Delay 头耗时响应的假实现，记录 ``fake`` 发出的每个请求收到的 timeout。"""

    def send(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None):
        delay = float(request.headers.get("X-Fake-Delay", 0))
        read_timeout = timeout[1] if isinstance(timeout, tuple) else timeout
        # 只记录本测试的请求：前一个测试被放弃的后台线程可能仍在发请求
        if request.headers.get("X-Fake-Id") == str(id(fake)):
            fake.http.append((request.url, read_timeout))
        if read_timeout is not None and delay >= read_timeout:
            fake._released.wait(read_timeout)
            raise requests.exceptions.ReadTimeout(f"Read timed out. (read timeout={read_timeout})", request=request)
        fake._released.wait(delay)
        response = requests.Response()
        response.status_code = 200
        response._content = b"{}"
        response.url = request.url
        response.request = request
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
