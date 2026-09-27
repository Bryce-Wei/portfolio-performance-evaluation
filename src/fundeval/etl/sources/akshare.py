"""akshare 数据源：国内公募基金净值、指数行情与无风险利率（正文第一部分“数据准备”）。

akshare 是可选依赖，按需延迟导入；未安装时报错并提示 ``pip install "fundeval[data]"``。
上游接口名、参数与返回列名集中在本模块顶部的映射里，上游改名时只需改这一层。
以下接口按 akshare 1.18.97 的源码核对：

- 基金：``fund_open_fund_info_em(symbol, indicator, period)``，东方财富（天天基金）源。
  indicator="单位净值走势" 返回 净值日期、单位净值、日增长率（百分数）；
  indicator="分红送配详情" 返回 年份、权益登记日、除息日、每10份分红（如“每10份派现金9.0000元”，
  即每 10 份的金额，已用真实返回核对）、分红发放日；旧格式的“每份分红”列也接受；
  indicator="累计净值走势" 返回 净值日期、累计净值
- 指数：``index_zh_a_hist(symbol, period, start_date, end_date)``（东方财富，中证、国证、上证等 A 股指数），
  ``stock_zh_index_hist_csindex(symbol, start_date, end_date)``（中证指数官网，如 H11001 中证全债），
  ``bond_composite_index_cbond(indicator, period)`` 与 ``bond_new_composite_index_cbond``（中债综合 / 新综合）
- 利率：``rate_interbank(market, symbol, indicator)`` 返回 报告日、利率（%）、涨跌（Shibor 各期限）；
  ``bond_zh_us_rate(start_date)`` 返回 日期、中国国债收益率2年 / 5年 / 10年 / 30年（%）等

注意：累计净值 = 单位净值 + 累计分红，没有做分红再投资，不能当作复权净值计算收益。
本模块以单位净值加除息日分红计算总收益（etl.returns.price_to_returns），
再与接口给出的日增长率交叉核对，差异超过容差的日期进入数据质量报告。
份额拆分会使单位净值跳变，本模块不做拆分调整，这类日期会在交叉核对中暴露。

原始数据按 (接口, 代码, 日期区间) 以 CSV 缓存到 ``cache_dir``（默认 ``~/.fundeval/cache``），
``use_cache=False`` 关闭缓存，``refresh=True`` 强制重新请求并覆盖缓存。不接受日期参数、
总是返回全历史的接口（基金净值与分红、Shibor、中债）不按日期区间分键。缓存命中时检查
覆盖范围与新旧（见 _stale_reason），过期则重新拉取；重新拉取失败时回退到旧缓存，
但一定发出警告并注明“使用 YYYY-MM-DD 的缓存数据”。

网络类异常（requests.RequestException 及其子类、ConnectionError、TimeoutError）按指数退避
共尝试 MAX_ATTEMPTS 次，其他异常不重试。每次请求有 ``timeout`` 秒（默认 DEFAULT_TIMEOUT = 30）
的总时限：akshare 内部调用 requests 时多未设超时，挂起的请求不会抛异常，因此在调用期间为
requests 注入默认超时，并以守护线程限定总时长，超时按网络类异常重试或切换数据源。指数行情默认 ``source="auto"``：纯数字代码先走
东方财富，网络失败后改用中证指数官网；无风险利率 ``source="auto"`` 按 RF_AUTO_ORDER
（Shibor 3M → 国债 2 年）依次尝试，全部失败时报错，不静默改用常数。所有自动切换与缓存
回退都发出 RuntimeWarning，并写入 collect_notes() 收集的附注，供报告使用。
"""

from __future__ import annotations

import contextlib
import re
import threading
import time
import warnings
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fundeval.etl import schema
from fundeval.etl.quality import CROSS_CHECK_TOLERANCE, nav_growth_check
from fundeval.etl.returns import period_code, period_end_index, price_to_returns, to_frequency

INSTALL_HINT = 'pip install "fundeval[data]"'

#: 按源码核对过接口名、参数与返回列名的 akshare 版本
VERIFIED_VERSION = "1.18.97"

DEFAULT_CACHE_DIR = Path.home() / ".fundeval" / "cache"

# ---------------------------------------------------------------------------
# 上游接口映射：函数名、参数与列名。akshare 改名时只改这里。
# ---------------------------------------------------------------------------

FUND_NAV = {
    "func": "fund_open_fund_info_em",
    "kwargs": {"indicator": "单位净值走势", "period": "成立来"},
    "date": "净值日期",
    "nav": "单位净值",
    "growth": "日增长率",  # 百分数
}
FUND_ACCUMULATED_NAV = {
    "func": "fund_open_fund_info_em",
    "kwargs": {"indicator": "累计净值走势", "period": "成立来"},
    "date": "净值日期",
    "value": "累计净值",
}
FUND_DIVIDEND = {
    "func": "fund_open_fund_info_em",
    "kwargs": {"indicator": "分红送配详情"},
    "ex_date": "除息日",
    "amount": "每10份分红",  # 值如“每10份派现金9.0000元”，按每 10 份计
    "legacy_amount": "每份分红",  # 旧格式，按每份计
}
INDEX_EM = {
    "func": "index_zh_a_hist",
    "kwargs": {"period": "daily"},
    "date": "日期",
    "close": "收盘",
}
INDEX_CSINDEX = {
    "func": "stock_zh_index_hist_csindex",
    "kwargs": {},
    "date": "日期",
    "close": "收盘",
}
INDEX_CBOND = {
    "composite": {"func": "bond_composite_index_cbond", "label": "中债-综合指数（财富）"},
    "new_composite": {"func": "bond_new_composite_index_cbond", "label": "中债-新综合指数（财富）"},
}
INDEX_CBOND_KWARGS = {"indicator": "财富", "period": "总值"}
INDEX_CBOND_COLUMNS = {"date": "date", "value": "value"}

#: 无风险利率来源。rate 列为年化百分数。
RATE_SOURCES = {
    "shibor3m": {
        "func": "rate_interbank",
        "kwargs": {"market": "上海银行同业拆借市场", "symbol": "Shibor人民币", "indicator": "3月"},
        "date": "报告日",
        "rate": "利率",
        "label": "Shibor 3M",
    },
    "shibor1m": {
        "func": "rate_interbank",
        "kwargs": {"market": "上海银行同业拆借市场", "symbol": "Shibor人民币", "indicator": "1月"},
        "date": "报告日",
        "rate": "利率",
        "label": "Shibor 1M",
    },
    "shibor_on": {
        "func": "rate_interbank",
        "kwargs": {"market": "上海银行同业拆借市场", "symbol": "Shibor人民币", "indicator": "隔夜"},
        "date": "报告日",
        "rate": "利率",
        "label": "Shibor 隔夜",
    },
    "cgb2y": {
        "func": "bond_zh_us_rate",
        "start_arg": "start_date",
        "date": "日期",
        "rate": "中国国债收益率2年",
        "label": "中国国债 2 年期收益率",
    },
    "cgb10y": {
        "func": "bond_zh_us_rate",
        "start_arg": "start_date",
        "date": "日期",
        "rate": "中国国债收益率10年",
        "label": "中国国债 10 年期收益率",
    },
}


#: 无风险利率 source="auto" 时依次尝试的来源
RF_AUTO_ORDER = ("shibor3m", "cgb2y")

#: 指数数据源的中文说明
INDEX_SOURCE_LABELS = {
    "em": "东方财富 index_zh_a_hist",
    "csindex": "中证指数官网 stock_zh_index_hist_csindex",
    "cbond": "中债财富指数（bond_composite_index_cbond / bond_new_composite_index_cbond）",
}

#: 网络类异常的最多尝试次数（含首次请求）与退避基数（秒）：第 k 次失败后等待 RETRY_BACKOFF × 2^(k−1)
MAX_ATTEMPTS = 3
RETRY_BACKOFF = 1.0

#: 每次上游请求的默认超时（秒）。akshare 内部调用 requests 时多未设 timeout，请求可能无限挂起
DEFAULT_TIMEOUT = 30.0

#: 缓存覆盖范围允许的滞后（自然日）与无日期接口、end 为 None 时的缓存有效期
CACHE_LAG_DAYS = 7
CACHE_MAX_AGE = pd.Timedelta(days=1)


class AkshareInterfaceError(RuntimeError):
    """akshare 缺少所需接口，或返回的列名与适配层的映射不一致。"""


class DataSourceUnavailable(ConnectionError):
    """自动切换的全部候选数据源都因网络原因失败。"""


class UpstreamTimeout(TimeoutError):
    """上游请求超过超时秒数仍未返回；按网络类异常处理（重试或切换数据源）。"""


# ---------------------------------------------------------------------------
# 网络异常、重试与附注
# ---------------------------------------------------------------------------


def network_errors() -> tuple[type[BaseException], ...]:
    """视为网络类的异常：requests.RequestException（未安装 requests 时略过）、ConnectionError、TimeoutError。"""
    errors: list[type[BaseException]] = [ConnectionError, TimeoutError]
    try:
        import requests
    except ImportError:  # pragma: no cover - requests 随 akshare 安装
        pass
    else:
        errors.append(requests.RequestException)
    return tuple(errors)


def is_network_error(exc: BaseException) -> bool:
    """是否为网络类异常（可重试、可切换数据源）。"""
    return isinstance(exc, network_errors())


def _sleep(seconds: float) -> None:
    """退避等待；测试中用 monkeypatch 替换。"""
    time.sleep(seconds)


_TIMEOUT_LOCK = threading.Lock()
_TIMEOUT_STATE: dict = {"depth": 0, "timeout": None, "original": None}


@contextlib.contextmanager
def _requests_default_timeout(timeout: float) -> Iterator[None]:
    """调用期间临时包装 requests.Session.request：调用方未给 timeout 时注入默认超时。

    akshare 用 requests.get 等函数发请求，最终都经过 Session.request；注入后挂起的连接会在
    超时后抛出 requests.Timeout，工作线程随之结束。未安装 requests 时不做任何事。
    """
    try:
        import requests
    except ImportError:  # pragma: no cover - requests 随 akshare 安装
        yield
        return
    with _TIMEOUT_LOCK:
        if _TIMEOUT_STATE["depth"] == 0:
            original = requests.Session.request

            def request(self, method, url, **kwargs):
                if kwargs.get("timeout") is None and _TIMEOUT_STATE["timeout"] is not None:
                    kwargs["timeout"] = _TIMEOUT_STATE["timeout"]
                return original(self, method, url, **kwargs)

            _TIMEOUT_STATE["original"] = original
            requests.Session.request = request
        _TIMEOUT_STATE["depth"] += 1
        previous = _TIMEOUT_STATE["timeout"]
        _TIMEOUT_STATE["timeout"] = timeout
    try:
        yield
    finally:
        with _TIMEOUT_LOCK:
            _TIMEOUT_STATE["timeout"] = previous
            _TIMEOUT_STATE["depth"] -= 1
            if _TIMEOUT_STATE["depth"] == 0:
                requests.Session.request = _TIMEOUT_STATE["original"]
                _TIMEOUT_STATE["original"] = None


def _call_with_timeout(fetch: Callable[[], pd.DataFrame], timeout: float | None, name: str) -> pd.DataFrame:
    """在守护线程中调用 fetch()，超过 timeout 秒未返回时抛出 UpstreamTimeout。

    两层保护：requests 层注入默认超时，让挂起的连接自行结束；线程层以 timeout 为总时限，
    不依赖上游是否使用 requests。超时后工作线程若仍未结束，作为守护线程留在后台，
    不阻塞程序退出。``timeout`` 为 None 时直接调用。
    """
    if timeout is None:
        return fetch()
    if timeout <= 0:
        raise ValueError(f"timeout 须为正数或 None，收到 {timeout!r}")
    box: dict = {}

    def run():
        try:
            box["value"] = fetch()
        except BaseException as exc:  # 在调用线程中重新抛出
            box["error"] = exc

    with _requests_default_timeout(timeout):
        worker = threading.Thread(target=run, name=f"fundeval-{name}", daemon=True)
        worker.start()
        worker.join(timeout)
    if worker.is_alive():
        raise UpstreamTimeout(f"{name} 超过 {timeout:g} 秒未返回")
    if "error" in box:
        raise box["error"]
    return box["value"]


def _with_retry(fetch: Callable[[], pd.DataFrame], *, timeout: float | None = DEFAULT_TIMEOUT, name: str = "上游接口") -> pd.DataFrame:
    """网络类异常（含超时）按指数退避共尝试 MAX_ATTEMPTS 次，最后一次仍失败时抛出；其他异常立即抛出。

    每次尝试都受 ``timeout`` 秒的总时限约束（见 _call_with_timeout），超时视为网络类异常。
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return _call_with_timeout(fetch, timeout, name)
        except Exception as exc:
            if not is_network_error(exc) or attempt == MAX_ATTEMPTS:
                raise
            _sleep(RETRY_BACKOFF * 2 ** (attempt - 1))
    raise AssertionError("unreachable")  # pragma: no cover


_NOTE_SINKS: list[list[str]] = []


@contextlib.contextmanager
def collect_notes() -> Iterator[list[str]]:
    """收集块内发生的数据源自动切换与缓存回退说明，供写入报告附注::

        with collect_notes() as notes:
            r = index_returns("000300", ...)
        evaluate(..., notes=notes)
    """
    notes: list[str] = []
    _NOTE_SINKS.append(notes)
    try:
        yield notes
    finally:
        _NOTE_SINKS.remove(notes)


def _notify(message: str) -> None:
    """发出 RuntimeWarning，并写入当前所有 collect_notes() 收集器。"""
    warnings.warn(message, RuntimeWarning, stacklevel=3)
    for sink in _NOTE_SINKS:
        sink.append(message)


def _describe_error(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _ak():
    """延迟导入 akshare；测试中用 monkeypatch 替换本函数。"""
    try:
        import akshare
    except ImportError as exc:
        raise ImportError(f"需要可选依赖 akshare，请先安装：{INSTALL_HINT}") from exc
    return akshare


def _interface(name: str):
    ak = _ak()
    fn = getattr(ak, name, None)
    if fn is None:
        version = getattr(ak, "__version__", "未知版本")
        raise AkshareInterfaceError(
            f"当前 akshare（{version}）没有接口 {name}，上游可能已改名；"
            f"请更新 fundeval.etl.sources.akshare 顶部的接口映射（已核对版本 {VERIFIED_VERSION}）"
        )
    return fn


def _require_columns(df: pd.DataFrame, columns, interface: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise AkshareInterfaceError(
            f"{interface} 返回的数据缺少列 {missing}（实际列：{list(df.columns)}）；"
            f"上游列名可能已变化，请更新 fundeval.etl.sources.akshare 的映射（已核对版本 {VERIFIED_VERSION}）"
        )


# ---------------------------------------------------------------------------
# 缓存
# ---------------------------------------------------------------------------


def _date_key(value) -> str:
    return "all" if value is None else pd.Timestamp(value).strftime("%Y%m%d")


def cache_path(cache_dir, interface: str, code: str, start, end) -> Path:
    """缓存文件路径：<cache_dir>/<接口>__<代码>__<起>__<止>.csv。"""
    safe = re.sub(r"[^\w.\-一-鿿]", "_", f"{interface}__{code}__{_date_key(start)}__{_date_key(end)}")
    return Path(cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR).expanduser() / f"{safe}.csv"


def _today() -> pd.Timestamp:
    """当天日期；测试中用 monkeypatch 替换。"""
    return pd.Timestamp.today().normalize()


def _coverage_target(end, lag_days: int) -> pd.Timestamp:
    """缓存至少应覆盖到的日期：min(end, 今天) 之前（含）的最后一个工作日，再减去 lag_days 个自然日。"""
    upto = _today() if end is None else min(pd.Timestamp(end).normalize(), _today())
    return pd.offsets.BDay().rollback(upto) - pd.Timedelta(days=int(lag_days))


def _stale_reason(df: pd.DataFrame, path: Path, *, date_col, end, dated: bool, lag_days: int, max_age) -> str | None:
    """判断缓存是否过期，过期时返回原因，否则返回 None。

    - 覆盖范围：给出日期列时，缓存数据的最后日期早于 _coverage_target(end) 即过期
      （end 为 None 时按今天计算）
    - 有效期：不接受日期参数的接口（dated=False）或 end 为 None 的请求，缓存文件修改时间
      早于 max_age 之前也视为过期
    """
    if date_col is not None and date_col in df.columns:
        dates = _to_dates(df[date_col])
        last = dates.max() if len(dates) else pd.NaT
        target = _coverage_target(end, lag_days)
        if pd.isna(last) or last < target:
            last_text = "无有效日期" if pd.isna(last) else f"截至 {last:%Y-%m-%d}"
            return f"缓存数据{last_text}，未覆盖到 {target:%Y-%m-%d}"
    if not dated or end is None:
        age = pd.Timedelta(seconds=time.time() - path.stat().st_mtime)
        if age > _as_timedelta(max_age):
            return f"缓存文件已超过 {_as_timedelta(max_age)} 未更新"
    return None


def _as_timedelta(value) -> pd.Timedelta:
    """cache_max_age 可为 Timedelta 或天数。"""
    if isinstance(value, pd.Timedelta):
        return value
    return pd.Timedelta(days=float(value))


def _read_cache(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _cached_call(
    interface: str,
    code: str,
    start,
    end,
    fetch,
    *,
    cache_dir=None,
    use_cache: bool = True,
    refresh: bool = False,
    cache_lag_days: int = CACHE_LAG_DAYS,
    cache_max_age=CACHE_MAX_AGE,
    timeout: float | None = DEFAULT_TIMEOUT,
    date_col: str | None = None,
    coverage_end=None,
    dated: bool = True,
) -> pd.DataFrame:
    """按 (接口, 代码, 日期区间) 读取缓存；未命中、过期、关闭缓存或强制刷新时调用 fetch()。

    ``coverage_end`` 为调用方请求的截止日期，用于覆盖范围检查；``dated=False`` 表示接口不接受
    日期参数（总是返回全历史），此时另按 ``cache_max_age`` 检查缓存文件的新旧。
    ``timeout`` 为每次上游请求的超时秒数（见 _with_retry）。
    重新拉取遇到网络类异常且已有旧缓存时回退到旧缓存，发出警告并写入附注。
    """
    path = cache_path(cache_dir, interface, code, start, end)
    cached = _read_cache(path) if use_cache and path.exists() else None
    reason = "指定了强制刷新"
    if cached is not None and not refresh:
        reason = _stale_reason(
            cached, path, date_col=date_col, end=coverage_end, dated=dated,
            lag_days=cache_lag_days, max_age=cache_max_age,
        )
        if reason is None:
            return cached
    try:
        df = _with_retry(fetch, timeout=timeout, name=f"{interface}（{code}）")
    except Exception as exc:
        if cached is None or not is_network_error(exc):
            raise
        cached_on = datetime.fromtimestamp(path.stat().st_mtime)
        _notify(
            f"{interface}（{code}）重新拉取失败（{_describe_error(exc)}），{reason}；"
            f"使用 {cached_on:%Y-%m-%d} 的缓存数据"
        )
        return cached
    if not isinstance(df, pd.DataFrame):
        raise AkshareInterfaceError(f"{interface} 返回的不是 DataFrame：{type(df)!r}")
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8")
    return df


def _call(
    spec: dict, code: str, start, end, cache: dict, *, date_col: str | None = None, coverage_end=None,
    dated: bool = True, **kwargs,
) -> pd.DataFrame:
    fn_name = spec["func"]
    indicator = spec.get("kwargs", {}).get("indicator")
    interface = fn_name if indicator is None else f"{fn_name}-{indicator}"
    call_kwargs = {**spec.get("kwargs", {}), **kwargs}

    def fetch():
        return _interface(fn_name)(**call_kwargs)

    return _cached_call(
        interface, code, start, end, fetch, date_col=date_col, coverage_end=coverage_end, dated=dated, **cache
    )


def _cache_opts(cache_dir, use_cache, refresh, cache_lag_days, cache_max_age, timeout) -> dict:
    return dict(
        cache_dir=cache_dir, use_cache=use_cache, refresh=refresh,
        cache_lag_days=cache_lag_days, cache_max_age=cache_max_age, timeout=timeout,
    )


def _to_dates(col: pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(col.astype(str), errors="coerce"))


def _series(df: pd.DataFrame, date_col: str, value_col: str, name: str) -> pd.Series:
    s = pd.Series(pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float), index=_to_dates(df[date_col]))
    s = s[s.index.notna()].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.rename(name).rename_axis(schema.DATE)


def _slice(obj, start, end):
    return obj.loc[pd.Timestamp(start) if start is not None else None : pd.Timestamp(end) if end is not None else None]


# ---------------------------------------------------------------------------
# 基金
# ---------------------------------------------------------------------------

_AMOUNT = re.compile(r"(\d+(?:\.\d+)?)\s*元")
_UNITS = re.compile(r"每\s*(\d*)\s*份")


def _parse_dividend_amount(value, units: int | None = None) -> float:
    """把分红记录换算为每份现金分红（元）。

    - 文字：“每10份派现金9.0000元” → 9.0 / 10 = 0.9；“每份派现金0.0500元” → 0.05。
      文字中写明的份数优先；写不出份数或金额的一律报错，不猜测
    - 数值：须由 ``units`` 给出每多少份（来自列名，如“每10份分红”为 10），否则报错
    """
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        if not np.isfinite(value):
            raise schema.SchemaError(f"无法解析分红金额：{value!r}")
        if units is None:
            raise schema.SchemaError(f"分红金额 {value!r} 未说明按每多少份计，无法换算为每份分红")
        return float(value) / units
    text = str(value)
    amount = _AMOUNT.search(text)
    unit_match = _UNITS.search(text)
    if amount is None or unit_match is None:
        raise schema.SchemaError(f"无法解析分红金额或份数：{text!r}（应形如“每10份派现金9.0000元”）")
    n = int(unit_match.group(1)) if unit_match.group(1) else 1
    if n <= 0:
        raise schema.SchemaError(f"分红份数无效：{text!r}")
    return float(amount.group(1)) / n


def _dividends(raw: pd.DataFrame, nav_index: pd.DatetimeIndex) -> pd.Series:
    """除息日每份现金分红，映射到除息日当天（非净值日则顺延到下一个净值日）。"""
    out = pd.Series(0.0, index=nav_index)
    if raw.empty:
        return out
    if FUND_DIVIDEND["amount"] in raw.columns:
        amount_col, units = FUND_DIVIDEND["amount"], 10
    elif FUND_DIVIDEND["legacy_amount"] in raw.columns:
        amount_col, units = FUND_DIVIDEND["legacy_amount"], 1
    else:
        raise AkshareInterfaceError(
            f"{FUND_DIVIDEND['func']} 返回的分红数据缺少金额列 {FUND_DIVIDEND['amount']!r} 或 "
            f"{FUND_DIVIDEND['legacy_amount']!r}（实际列：{list(raw.columns)}）；上游列名可能已变化，"
            f"请更新 fundeval.etl.sources.akshare 的映射（已核对版本 {VERIFIED_VERSION}）"
        )
    _require_columns(raw, [FUND_DIVIDEND["ex_date"]], FUND_DIVIDEND["func"])
    ex_dates = _to_dates(raw[FUND_DIVIDEND["ex_date"]])
    amounts = [_parse_dividend_amount(v, units) for v in raw[amount_col]]
    for date, amount in zip(ex_dates, amounts):
        if pd.isna(date):
            raise schema.SchemaError("分红记录的除息日无法解析")
        pos = nav_index.searchsorted(date)
        if pos < len(nav_index):
            out.iloc[pos] += amount
    return out


def fund_nav(
    code: str,
    start=None,
    end=None,
    *,
    cache_dir=None,
    use_cache: bool = True,
    refresh: bool = False,
    cache_lag_days: int = CACHE_LAG_DAYS,
    cache_max_age=CACHE_MAX_AGE,
    timeout: float | None = DEFAULT_TIMEOUT,
) -> pd.DataFrame:
    """开放式基金单位净值、日增长率与除息日分红。

    返回以日期为索引的 DataFrame：``nav``（单位净值）、``reported_growth``（接口给出的
    日增长率，已由百分数换算为小数）、``dividend``（除息日每份现金分红，其余日期为 0）。
    结果截取到 [start, end]。累计净值没有做分红再投资，这里不使用。
    """
    frame = _fund_frame(code, end, _cache_opts(cache_dir, use_cache, refresh, cache_lag_days, cache_max_age, timeout))
    return _slice(frame, start, end)


def _fund_frame(code: str, end, cache: dict) -> pd.DataFrame:
    # 接口总是返回成立以来的全历史，缓存不按日期区间分键
    raw = _call(FUND_NAV, code, None, None, cache, date_col=FUND_NAV["date"], coverage_end=end, dated=False, symbol=code)
    _require_columns(raw, [FUND_NAV["date"], FUND_NAV["nav"], FUND_NAV["growth"]], FUND_NAV["func"])
    nav = _series(raw, FUND_NAV["date"], FUND_NAV["nav"], schema.NAV)
    growth = _series(raw, FUND_NAV["date"], FUND_NAV["growth"], "reported_growth") / 100
    div_raw = _call(FUND_DIVIDEND, code, None, None, cache, dated=False, symbol=code)
    frame = pd.DataFrame({schema.NAV: nav, "reported_growth": growth.reindex(nav.index)})
    frame[schema.DIVIDEND] = _dividends(div_raw, frame.index)
    return frame


def fund_returns(
    code: str,
    start=None,
    end=None,
    *,
    freq: str | None = None,
    tolerance: float = CROSS_CHECK_TOLERANCE,
    return_check: bool = False,
    cache_dir=None,
    use_cache: bool = True,
    refresh: bool = False,
    cache_lag_days: int = CACHE_LAG_DAYS,
    cache_max_age=CACHE_MAX_AGE,
    timeout: float | None = DEFAULT_TIMEOUT,
):
    """基金单期总收益（小数），由单位净值加分红计算：r_t = (NAV_t + D_t) / NAV_{t-1} − 1。

    累计净值 = 单位净值 + 累计分红，没有做分红再投资，不能直接当复权净值用；
    这里在除息日把每份分红加回，得到分红再投资口径的总收益。

    计算结果与接口给出的“日增长率”逐日交叉核对（etl.quality.nav_growth_check），
    差异超过 ``tolerance``（默认 5 个基点）的日期发出警告；``return_check=True`` 时返回
    ``(收益, 核对表)``，核对表可传给 etl.quality.data_quality_report 的 ``cross_check``。

    ``freq`` 为 W/M/Q/A 时用 etl.returns.to_frequency 合成为该频率（期内有缺失则为 NaN），
    核对表仍为日度。首个收益需要 start 之前一个净值，本函数取全历史计算后再截取区间。
    """
    frame = _fund_frame(code, end, _cache_opts(cache_dir, use_cache, refresh, cache_lag_days, cache_max_age, timeout))
    nav = frame[schema.NAV].dropna()
    if len(nav) < len(frame):
        warnings.warn(f"基金 {code} 有 {len(frame) - len(nav)} 个日期单位净值缺失，已跳过这些日期", RuntimeWarning, stacklevel=2)
    div = frame[schema.DIVIDEND].reindex(nav.index)
    total = price_to_returns(nav, div[div > 0])
    total = _slice(total, start, end).rename(schema.PORTFOLIO)
    check = nav_growth_check(total, frame["reported_growth"].reindex(total.index), tolerance)
    n_flagged = int(check["flagged"].sum())
    if n_flagged:
        warnings.warn(
            f"基金 {code} 有 {n_flagged} 个日期净值推算收益与日增长率相差超过 {tolerance * 1e4:.0f} 个基点，"
            "请查看数据质量报告（return_check=True）",
            RuntimeWarning,
            stacklevel=2,
        )
    out = to_frequency(total, freq).rename(schema.PORTFOLIO) if freq else total
    return (out, check) if return_check else out


def fund_accumulated_nav(
    code: str,
    start=None,
    end=None,
    *,
    cache_dir=None,
    use_cache: bool = True,
    refresh: bool = False,
    cache_lag_days: int = CACHE_LAG_DAYS,
    cache_max_age=CACHE_MAX_AGE,
    timeout: float | None = DEFAULT_TIMEOUT,
) -> pd.Series:
    """累计净值（单位净值 + 累计分红），仅供核对与展示。

    累计净值没有做分红再投资，分红后按它计算的收益会低估总收益，不能用于绩效计算；
    请用 fund_returns。
    """
    spec = FUND_ACCUMULATED_NAV
    raw = _call(
        spec, code, None, None, _cache_opts(cache_dir, use_cache, refresh, cache_lag_days, cache_max_age, timeout),
        date_col=spec["date"], coverage_end=end, dated=False, symbol=code,
    )
    _require_columns(raw, [spec["date"], spec["value"]], spec["func"])
    return _slice(_series(raw, spec["date"], spec["value"], "accumulated_nav"), start, end)


# ---------------------------------------------------------------------------
# 指数
# ---------------------------------------------------------------------------

#: 取指数行情时向前多取的自然日数，保证区间首日有前一交易日收盘价
_LOOKBACK_DAYS = 40


def index_source(code: str) -> str:
    """按代码推断默认数据源：``cbond:`` 前缀为中债，含字母（如 H11001、H00300）为中证指数官网，其余为东方财富。"""
    if code.startswith("cbond:"):
        return "cbond"
    if not code.isdigit():
        return "csindex"
    return "em"


def _index_prices_from(source: str, code: str, fetch_start, end, cache: dict) -> pd.Series:
    s_arg = "19700101" if fetch_start is None else fetch_start.strftime("%Y%m%d")
    e_arg = "22220101" if end is None else pd.Timestamp(end).strftime("%Y%m%d")
    if source in ("em", "csindex"):
        spec = INDEX_EM if source == "em" else INDEX_CSINDEX
        raw = _call(
            spec, code, fetch_start, end, cache, date_col=spec["date"], coverage_end=end,
            symbol=code, start_date=s_arg, end_date=e_arg,
        )
        _require_columns(raw, [spec["date"], spec["close"]], spec["func"])
        return _series(raw, spec["date"], spec["close"], code)
    if source == "cbond":
        key = code.removeprefix("cbond:")
        if key not in INDEX_CBOND:
            raise ValueError(f"未知中债指数 {code!r}，可选：{['cbond:' + k for k in INDEX_CBOND]}")
        spec = {"func": INDEX_CBOND[key]["func"], "kwargs": INDEX_CBOND_KWARGS}
        raw = _call(spec, code, None, None, cache, date_col=INDEX_CBOND_COLUMNS["date"], coverage_end=end, dated=False)
        _require_columns(raw, list(INDEX_CBOND_COLUMNS.values()), spec["func"])
        return _series(raw, INDEX_CBOND_COLUMNS["date"], INDEX_CBOND_COLUMNS["value"], code)
    raise ValueError(f"source 须为 'auto'、'em'、'csindex' 或 'cbond'，收到 {source!r}")


def index_prices(
    code: str,
    start=None,
    end=None,
    *,
    source: str | None = "auto",
    cache_dir=None,
    use_cache: bool = True,
    refresh: bool = False,
    cache_lag_days: int = CACHE_LAG_DAYS,
    cache_max_age=CACHE_MAX_AGE,
    timeout: float | None = DEFAULT_TIMEOUT,
) -> pd.Series:
    """指数日收盘点位（中债为财富指数值），含 start 之前约 40 个自然日，供计算首日收益。

    ``source="auto"``（默认，None 同义）：``cbond:`` 代码走中债，H 开头等中证代码直接走中证
    指数官网；纯数字代码先试东方财富，网络类异常（重试后仍失败）时自动改用中证指数官网，
    发出 RuntimeWarning 并写入 collect_notes() 的附注。实际使用的数据源记录在结果的
    ``attrs["source"]``（em / csindex / cbond）与 ``attrs["source_label"]``。
    """
    cache = _cache_opts(cache_dir, use_cache, refresh, cache_lag_days, cache_max_age, timeout)
    fetch_start = None if start is None else pd.Timestamp(start) - pd.Timedelta(days=_LOOKBACK_DAYS)
    fallback = None
    if source in (None, "auto"):
        used = index_source(code)
        if used == "em":
            try:
                price = _index_prices_from("em", code, fetch_start, end, cache)
            except Exception as exc:
                if not is_network_error(exc):
                    raise
                fallback = f"{INDEX_SOURCE_LABELS['em']} 请求失败（{_describe_error(exc)}）"
                _notify(f"指数 {code}：{fallback}，已改用{INDEX_SOURCE_LABELS['csindex']}")
                used = "csindex"
                price = _index_prices_from("csindex", code, fetch_start, end, cache)
        else:
            price = _index_prices_from(used, code, fetch_start, end, cache)
    else:
        used = source
        price = _index_prices_from(source, code, fetch_start, end, cache)
    price = _slice(price, fetch_start, end)
    if price.isna().any():
        warnings.warn(f"指数 {code} 有 {int(price.isna().sum())} 个交易日收盘价缺失，已跳过这些日期", RuntimeWarning, stacklevel=2)
        price = price.dropna()
    price.attrs.update(source=used, source_label=INDEX_SOURCE_LABELS.get(used, used), fallback=fallback)
    return price


def index_returns(
    code: str,
    start=None,
    end=None,
    *,
    source: str | None = "auto",
    freq: str | None = None,
    cache_dir=None,
    use_cache: bool = True,
    refresh: bool = False,
    cache_lag_days: int = CACHE_LAG_DAYS,
    cache_max_age=CACHE_MAX_AGE,
    timeout: float | None = DEFAULT_TIMEOUT,
) -> pd.Series:
    """指数单期收益（小数）：r_t = P_t / P_{t-1} − 1，P 为收盘点位。

    ``source`` 见 index_prices：默认 "auto"，沪深 300（000300）等纯数字代码先走东方财富
    ``index_zh_a_hist``，网络失败后改用中证指数官网；H00300（沪深 300 全收益）、H11001 等
    走中证指数官网；``cbond:composite`` / ``cbond:new_composite`` 为中债综合 / 新综合财富指数。
    价格指数不含成分股分红，与含分红的基金净值比较时会高估超额收益，宜用全收益指数
    （etl.benchmark.total_return_code）。``freq`` 为 W/M/Q/A 时合成为该频率。结果以代码命名，
    ``attrs`` 记录实际数据源。
    """
    price = index_prices(
        code, start, end, source=source, cache_dir=cache_dir, use_cache=use_cache, refresh=refresh,
        cache_lag_days=cache_lag_days, cache_max_age=cache_max_age, timeout=timeout,
    )
    r = _slice(price_to_returns(price).rename(code), start, end)
    out = to_frequency(r, freq).rename(code) if freq else r
    out.attrs = dict(price.attrs)
    return out


# ---------------------------------------------------------------------------
# 无风险收益
# ---------------------------------------------------------------------------


def rate_series(
    source: str,
    start=None,
    end=None,
    *,
    cache_dir=None,
    use_cache: bool = True,
    refresh: bool = False,
    cache_lag_days: int = CACHE_LAG_DAYS,
    cache_max_age=CACHE_MAX_AGE,
    timeout: float | None = DEFAULT_TIMEOUT,
) -> pd.Series:
    """年化利率序列（小数），来源见 RATE_SOURCES，如 "shibor3m"、"cgb10y"。"""
    if source not in RATE_SOURCES:
        raise ValueError(
            f"未知的无风险利率来源 {source!r}，可选：{sorted(RATE_SOURCES)} 或 'auto'，或直接传入常数年化利率"
        )
    spec = RATE_SOURCES[source]
    kwargs = {}
    fetch_start = None if start is None else pd.Timestamp(start) - pd.DateOffset(years=1)
    dated = "start_arg" in spec
    if dated:
        kwargs[spec["start_arg"]] = "19901219" if fetch_start is None else fetch_start.strftime("%Y%m%d")
    cache = _cache_opts(cache_dir, use_cache, refresh, cache_lag_days, cache_max_age, timeout)
    # 不接受日期参数的接口总是返回全历史，缓存不按日期区间分键
    raw = _call(
        spec, source, fetch_start if dated else None, end if dated else None, cache,
        date_col=spec["date"], coverage_end=end, dated=dated, **kwargs,
    )
    _require_columns(raw, [spec["date"], spec["rate"]], spec["func"])
    rate = _series(raw, spec["date"], spec["rate"], source).dropna() / 100
    return rate.loc[: pd.Timestamp(end) if end is not None else None]


def _per_period(y, k: int):
    return (1 + y) ** (1 / k) - 1


def risk_free_returns(
    start=None,
    end=None,
    freq: str = "M",
    source: str | float = "shibor3m",
    *,
    index: pd.DatetimeIndex | None = None,
    cache_dir=None,
    use_cache: bool = True,
    refresh: bool = False,
    cache_lag_days: int = CACHE_LAG_DAYS,
    cache_max_age=CACHE_MAX_AGE,
    timeout: float | None = DEFAULT_TIMEOUT,
) -> pd.Series:
    """把年化利率换算为每期无风险收益，采用复利口径 r_f = (1 + y)^(1/K) − 1。

    - y 为年化利率（小数），K 为一年期数（D=252、W=52、M=12、Q=4、A=1）
    - 每期使用期初已知的利率：周、月、季、年频取上一期最后一个报价，日频取前一个报价日
      的利率；更早没有报价的期为 NaN，不填补
    - ``source`` 为 RATE_SOURCES 中的名称（如 "shibor3m"）、"auto"，或常数年化利率（如 0.018）
    - ``source="auto"`` 按 RF_AUTO_ORDER（shibor3m → cgb2y）依次尝试，网络类异常时改用下一个，
      发出 RuntimeWarning 并写入附注；全部失败时抛出 DataSourceUnavailable，提示改用常数，
      不静默改用常数
    - Shibor 等货币市场利率本身是单利报价，这里按复利换算只是近似，报告中须写明口径

    ``index`` 给出时，结果按其日期所属的期对齐（例如月末交易日映射到同月）；
    否则周、月、季、年频以日历期末为索引，日频以报价日为索引（常数时为工作日）。
    结果的 ``attrs`` 记录实际来源（``source``）、失败后被跳过的来源（``failed``）与口径说明
    （``description``）。
    """
    k = schema.periods_per_year(freq)
    daily = str(freq).upper()[:1] == "D"
    if isinstance(source, (int, float, np.integer, np.floating)) and not isinstance(source, bool):
        value = _per_period(float(source), k)
        if index is None:
            if start is None or end is None:
                raise ValueError("常数无风险利率须给出 start 与 end，或给出 index")
            if daily:
                index = pd.bdate_range(start, end)
            else:
                index = pd.period_range(start, end, freq=period_code(freq)).to_timestamp(how="end").normalize()
        out = pd.Series(value, index=pd.DatetimeIndex(index).rename(schema.DATE), name=schema.RISK_FREE)
        out.attrs.update(source=float(source), failed=[], description=describe_rate_source(float(source), freq))
        return out

    candidates = RF_AUTO_ORDER if source == "auto" else (source,)
    failed: list[str] = []
    errors: list[str] = []
    rate = None
    used = None
    for name in candidates:
        try:
            rate = rate_series(
                name, start, end, cache_dir=cache_dir, use_cache=use_cache, refresh=refresh,
                cache_lag_days=cache_lag_days, cache_max_age=cache_max_age, timeout=timeout,
            )
        except Exception as exc:
            if source != "auto" or not is_network_error(exc):
                raise
            failed.append(name)
            errors.append(f"{RATE_SOURCES[name]['label']} 获取失败（{_describe_error(exc)}）")
            continue
        used = name
        break
    if rate is None:
        raise DataSourceUnavailable(
            f"无风险利率 {' → '.join(candidates)} 均获取失败：{'；'.join(errors)}；"
            "请改用常数年化利率，如命令行 --rf 0.018 或 risk_free_returns(source=0.018)"
        )
    if failed:
        _notify(f"无风险利率：{'；'.join(errors)}，已改用{RATE_SOURCES[used]['label']}")

    if daily:
        prior = rate.shift(1)  # 前一个报价日的利率
        if index is not None:
            target = pd.DatetimeIndex(index)
            prior = pd.Series(
                rate.asof(target - pd.Timedelta(days=1)).to_numpy() if len(rate) else np.nan, index=target
            )
        out = _per_period(prior, k)
    else:
        labels = period_end_index(rate.index, freq)
        last = rate.groupby(labels).last()
        full = pd.period_range(last.index.min(), last.index.max(), freq=period_code(freq))
        last = last.reindex(full.to_timestamp(how="end").normalize())
        out = _per_period(last.shift(1), k)  # 期初已知 = 上一期末报价；中间无报价的期为 NaN
        if index is not None:
            target = pd.DatetimeIndex(index)
            out = pd.Series(out.reindex(period_end_index(target, freq)).to_numpy(), index=target)
    out = out.rename(schema.RISK_FREE).rename_axis(schema.DATE)
    if index is None:
        out = _slice(out, start, end)
    out.attrs.update(source=used, failed=failed, description=describe_rate_source(used, freq, failed))
    return out


def describe_rate_source(source: str | float, freq: str, failed: list[str] | tuple[str, ...] = ()) -> str:
    """无风险收益口径的文字说明，供报告“口径”一节使用。

    ``failed`` 为自动切换时先失败的来源，例如写成“中国国债 2 年期收益率（Shibor 3M 获取失败后改用）”。
    """
    k = schema.periods_per_year(freq)
    if isinstance(source, (int, float, np.integer, np.floating)) and not isinstance(source, bool):
        if float(source) == 0:
            return "未提供，按 0 计"
        return f"常数年化 {float(source):.2%}，按 (1 + y)^(1/{k}) − 1 换算为每期"
    label = RATE_SOURCES.get(source, {}).get("label", str(source))
    if failed:
        label += f"（{'、'.join(RATE_SOURCES.get(f, {}).get('label', f) for f in failed)} 获取失败后改用）"
    return f"{label}（akshare），取期初已知报价，按 (1 + y)^(1/{k}) − 1 复利换算为每期"
