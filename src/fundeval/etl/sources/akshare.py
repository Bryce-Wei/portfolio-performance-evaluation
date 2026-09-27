"""akshare 数据源：国内公募基金净值、指数行情与无风险利率（正文第一部分“数据准备”）。

akshare 是可选依赖，按需延迟导入；未安装时报错并提示 ``pip install "fundeval[data]"``。
上游接口名、参数与返回列名集中在本模块顶部的映射里，上游改名时只需改这一层。
以下接口按 akshare 1.18.97 的源码核对：

- 基金：``fund_open_fund_info_em(symbol, indicator, period)``，东方财富（天天基金）源。
  indicator="单位净值走势" 返回 净值日期、单位净值、日增长率（百分数）；
  indicator="分红送配详情" 返回 年份、权益登记日、除息日、每份分红（如“每份派现金0.0500元”）、分红发放日；
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
``use_cache=False`` 关闭缓存，``refresh=True`` 强制重新请求并覆盖缓存。
"""

from __future__ import annotations

import re
import warnings
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
    "amount": "每份分红",
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


class AkshareInterfaceError(RuntimeError):
    """akshare 缺少所需接口，或返回的列名与适配层的映射不一致。"""


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


def _cached_call(
    interface: str, code: str, start, end, fetch, *, cache_dir=None, use_cache: bool = True, refresh: bool = False
) -> pd.DataFrame:
    """按 (接口, 代码, 日期区间) 读取缓存；未命中、关闭缓存或强制刷新时调用 fetch()。"""
    path = cache_path(cache_dir, interface, code, start, end)
    if use_cache and not refresh and path.exists():
        try:
            return pd.read_csv(path, encoding="utf-8")
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    df = fetch()
    if not isinstance(df, pd.DataFrame):
        raise AkshareInterfaceError(f"{interface} 返回的不是 DataFrame：{type(df)!r}")
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8")
    return df


def _call(spec: dict, code: str, start, end, cache: dict, **kwargs) -> pd.DataFrame:
    fn_name = spec["func"]
    indicator = spec.get("kwargs", {}).get("indicator")
    interface = fn_name if indicator is None else f"{fn_name}-{indicator}"
    call_kwargs = {**spec.get("kwargs", {}), **kwargs}

    def fetch():
        return _interface(fn_name)(**call_kwargs)

    return _cached_call(interface, code, start, end, fetch, **cache)


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


def _parse_dividend_amount(value) -> float:
    """解析“每份派现金0.0500元”或数值形式的每份分红；无法解析时报错，不猜测。"""
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
        return float(value)
    text = str(value)
    match = _AMOUNT.search(text)
    if match is None:
        raise schema.SchemaError(f"无法解析每份分红：{text!r}")
    return float(match.group(1))


def _dividends(raw: pd.DataFrame, nav_index: pd.DatetimeIndex) -> pd.Series:
    """除息日每份现金分红，映射到除息日当天（非净值日则顺延到下一个净值日）。"""
    out = pd.Series(0.0, index=nav_index)
    if raw.empty:
        return out
    _require_columns(raw, [FUND_DIVIDEND["ex_date"], FUND_DIVIDEND["amount"]], FUND_DIVIDEND["func"])
    ex_dates = _to_dates(raw[FUND_DIVIDEND["ex_date"]])
    amounts = [_parse_dividend_amount(v) for v in raw[FUND_DIVIDEND["amount"]]]
    for date, amount in zip(ex_dates, amounts):
        if pd.isna(date):
            raise schema.SchemaError("分红记录的除息日无法解析")
        pos = nav_index.searchsorted(date)
        if pos < len(nav_index):
            out.iloc[pos] += amount
    return out


def fund_nav(code: str, start=None, end=None, *, cache_dir=None, use_cache: bool = True, refresh: bool = False) -> pd.DataFrame:
    """开放式基金单位净值、日增长率与除息日分红。

    返回以日期为索引的 DataFrame：``nav``（单位净值）、``reported_growth``（接口给出的
    日增长率，已由百分数换算为小数）、``dividend``（除息日每份现金分红，其余日期为 0）。
    结果截取到 [start, end]。累计净值没有做分红再投资，这里不使用。
    """
    frame = _fund_frame(code, start, end, dict(cache_dir=cache_dir, use_cache=use_cache, refresh=refresh))
    return _slice(frame, start, end)


def _fund_frame(code: str, start, end, cache: dict) -> pd.DataFrame:
    raw = _call(FUND_NAV, code, start, end, cache, symbol=code)
    _require_columns(raw, [FUND_NAV["date"], FUND_NAV["nav"], FUND_NAV["growth"]], FUND_NAV["func"])
    nav = _series(raw, FUND_NAV["date"], FUND_NAV["nav"], schema.NAV)
    growth = _series(raw, FUND_NAV["date"], FUND_NAV["growth"], "reported_growth") / 100
    div_raw = _call(FUND_DIVIDEND, code, start, end, cache, symbol=code)
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
    frame = _fund_frame(code, start, end, dict(cache_dir=cache_dir, use_cache=use_cache, refresh=refresh))
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


def fund_accumulated_nav(code: str, start=None, end=None, *, cache_dir=None, use_cache: bool = True, refresh: bool = False) -> pd.Series:
    """累计净值（单位净值 + 累计分红），仅供核对与展示。

    累计净值没有做分红再投资，分红后按它计算的收益会低估总收益，不能用于绩效计算；
    请用 fund_returns。
    """
    raw = _call(FUND_ACCUMULATED_NAV, code, start, end, dict(cache_dir=cache_dir, use_cache=use_cache, refresh=refresh), symbol=code)
    _require_columns(raw, [FUND_ACCUMULATED_NAV["date"], FUND_ACCUMULATED_NAV["value"]], FUND_ACCUMULATED_NAV["func"])
    return _slice(_series(raw, FUND_ACCUMULATED_NAV["date"], FUND_ACCUMULATED_NAV["value"], "accumulated_nav"), start, end)


# ---------------------------------------------------------------------------
# 指数
# ---------------------------------------------------------------------------

#: 取指数行情时向前多取的自然日数，保证区间首日有前一交易日收盘价
_LOOKBACK_DAYS = 40


def index_source(code: str) -> str:
    """按代码推断数据源：``cbond:`` 前缀为中债，含字母（如 H11001）为中证指数官网，其余为东方财富。"""
    if code.startswith("cbond:"):
        return "cbond"
    if not code.isdigit():
        return "csindex"
    return "em"


def index_prices(code: str, start=None, end=None, *, source: str | None = None, cache_dir=None, use_cache: bool = True, refresh: bool = False) -> pd.Series:
    """指数日收盘点位（中债为财富指数值），含 start 之前约 40 个自然日，供计算首日收益。"""
    cache = dict(cache_dir=cache_dir, use_cache=use_cache, refresh=refresh)
    source = source or index_source(code)
    fetch_start = None if start is None else pd.Timestamp(start) - pd.Timedelta(days=_LOOKBACK_DAYS)
    s_arg = "19700101" if fetch_start is None else fetch_start.strftime("%Y%m%d")
    e_arg = "22220101" if end is None else pd.Timestamp(end).strftime("%Y%m%d")
    if source == "em":
        raw = _call(INDEX_EM, code, fetch_start, end, cache, symbol=code, start_date=s_arg, end_date=e_arg)
        _require_columns(raw, [INDEX_EM["date"], INDEX_EM["close"]], INDEX_EM["func"])
        price = _series(raw, INDEX_EM["date"], INDEX_EM["close"], code)
    elif source == "csindex":
        raw = _call(INDEX_CSINDEX, code, fetch_start, end, cache, symbol=code, start_date=s_arg, end_date=e_arg)
        _require_columns(raw, [INDEX_CSINDEX["date"], INDEX_CSINDEX["close"]], INDEX_CSINDEX["func"])
        price = _series(raw, INDEX_CSINDEX["date"], INDEX_CSINDEX["close"], code)
    elif source == "cbond":
        key = code.removeprefix("cbond:")
        if key not in INDEX_CBOND:
            raise ValueError(f"未知中债指数 {code!r}，可选：{['cbond:' + k for k in INDEX_CBOND]}")
        spec = {"func": INDEX_CBOND[key]["func"], "kwargs": INDEX_CBOND_KWARGS}
        raw = _call(spec, code, None, None, cache)
        _require_columns(raw, list(INDEX_CBOND_COLUMNS.values()), spec["func"])
        price = _series(raw, INDEX_CBOND_COLUMNS["date"], INDEX_CBOND_COLUMNS["value"], code)
    else:
        raise ValueError(f"source 须为 'em'、'csindex' 或 'cbond'，收到 {source!r}")
    price = _slice(price, fetch_start, end)
    if price.isna().any():
        warnings.warn(f"指数 {code} 有 {int(price.isna().sum())} 个交易日收盘价缺失，已跳过这些日期", RuntimeWarning, stacklevel=2)
        price = price.dropna()
    return price


def index_returns(
    code: str,
    start=None,
    end=None,
    *,
    source: str | None = None,
    freq: str | None = None,
    cache_dir=None,
    use_cache: bool = True,
    refresh: bool = False,
) -> pd.Series:
    """指数单期收益（小数）：r_t = P_t / P_{t-1} − 1，P 为收盘点位。

    ``source`` 缺省时按代码推断（见 index_source）：沪深 300（000300）、中证 500（000905）
    等走东方财富 ``index_zh_a_hist``；H11001（中证全债）等走中证指数官网；
    ``cbond:composite`` / ``cbond:new_composite`` 为中债综合 / 新综合财富指数。
    价格指数不含成分股分红，与含分红的基金净值比较时会高估超额收益，报告中须注明。
    ``freq`` 为 W/M/Q/A 时合成为该频率。结果以代码命名。
    """
    price = index_prices(code, start, end, source=source, cache_dir=cache_dir, use_cache=use_cache, refresh=refresh)
    r = _slice(price_to_returns(price).rename(code), start, end)
    return to_frequency(r, freq).rename(code) if freq else r


# ---------------------------------------------------------------------------
# 无风险收益
# ---------------------------------------------------------------------------


def rate_series(source: str, start=None, end=None, *, cache_dir=None, use_cache: bool = True, refresh: bool = False) -> pd.Series:
    """年化利率序列（小数），来源见 RATE_SOURCES，如 "shibor3m"、"cgb10y"。"""
    if source not in RATE_SOURCES:
        raise ValueError(f"未知的无风险利率来源 {source!r}，可选：{sorted(RATE_SOURCES)}，或直接传入常数年化利率")
    spec = RATE_SOURCES[source]
    kwargs = {}
    fetch_start = None if start is None else pd.Timestamp(start) - pd.DateOffset(years=1)
    if "start_arg" in spec:
        kwargs[spec["start_arg"]] = "19901219" if fetch_start is None else fetch_start.strftime("%Y%m%d")
        key_start = fetch_start
    else:
        key_start = None  # 接口不接受日期参数，总是返回全历史
    cache = dict(cache_dir=cache_dir, use_cache=use_cache, refresh=refresh)
    raw = _call(spec, source, key_start, end if "start_arg" in spec else None, cache, **kwargs)
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
) -> pd.Series:
    """把年化利率换算为每期无风险收益，采用复利口径 r_f = (1 + y)^(1/K) − 1。

    - y 为年化利率（小数），K 为一年期数（D=252、W=52、M=12、Q=4、A=1）
    - 每期使用期初已知的利率：周、月、季、年频取上一期最后一个报价，日频取前一个报价日
      的利率；更早没有报价的期为 NaN，不填补
    - ``source`` 为 RATE_SOURCES 中的名称（如 "shibor3m"），或常数年化利率（如 0.018）
    - Shibor 等货币市场利率本身是单利报价，这里按复利换算只是近似，报告中须写明口径

    ``index`` 给出时，结果按其日期所属的期对齐（例如月末交易日映射到同月）；
    否则周、月、季、年频以日历期末为索引，日频以报价日为索引（常数时为工作日）。
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
        return pd.Series(value, index=pd.DatetimeIndex(index).rename(schema.DATE), name=schema.RISK_FREE)

    rate = rate_series(source, start, end, cache_dir=cache_dir, use_cache=use_cache, refresh=refresh)
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
    return out


def describe_rate_source(source: str | float, freq: str) -> str:
    """无风险收益口径的文字说明，供报告“口径”一节使用。"""
    k = schema.periods_per_year(freq)
    if isinstance(source, (int, float, np.integer, np.floating)) and not isinstance(source, bool):
        return f"常数年化 {float(source):.2%}，按 (1 + y)^(1/{k}) − 1 换算为每期"
    label = RATE_SOURCES.get(source, {}).get("label", str(source))
    return f"{label}（akshare），取期初已知报价，按 (1 + y)^(1/{k}) − 1 复利换算为每期"
