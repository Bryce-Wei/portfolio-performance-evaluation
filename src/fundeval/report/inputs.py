"""报告输入的组装：按命令行或 Python 选项联网取数（akshare）或读本地文件，得到 evaluate 的参数。

``fundeval report`` 与 ``fundeval compare``（report.compare）共用本模块：基金收益、基准（代码与权重、
合同文字，或 ``contract`` 取基金合同业绩比较基准）、无风险收益、风格指数与因子。选项见 ReportOptions，
字段与命令行参数一一对应。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd

from fundeval.attribution.factor import (
    FACTOR_DESCRIPTIONS,
    FACTOR_PRESET_NAMES,
    FACTOR_PRESETS,
    INDEX_PROXY,
    US_FACTOR_PRESETS,
    US_MARKET,
    index_proxy_factors,
    preset_codes,
    us_market_factors,
)
from fundeval.etl import fx as fx_mod
from fundeval.etl import schema
from fundeval.etl.benchmark import (
    CASH,
    CNY,
    FX_CAVEAT,
    RETURN_TYPE_LABELS,
    STYLE_AUTO,
    STYLE_PRESETS,
    auto_style_preset,
    cbond_alternative_hint,
    index_record,
    benchmark_return_type,
    composite_benchmark,
    index_return_type,
    lookup_index_code,
    parse_benchmark,
    style_preset,
    total_return_code,
)
from fundeval.etl.quality import CROSS_CHECK_TOLERANCE
from fundeval.etl.returns import period_code, period_end_index, price_to_returns, to_frequency
from fundeval.etl.sources import files
from fundeval.etl.sources.akshare import DEFAULT_TIMEOUT, DEFAULT_TOTAL_TIMEOUT
from fundeval.report.summary import DEFAULT_FEE_BASIS


def parse_benchmark_spec(spec: str) -> list[tuple[str, float]]:
    """解析 --benchmark：``"000300:0.8,H11001:0.2"``（代码:权重）、单个代码，或合同文字
    （如 ``"沪深300指数收益率*80%+中债综合指数收益率*20%"``，名称按 etl.benchmark.INDEX_CODES 换算）。"""
    spec = spec.strip()
    if ":" in spec or "：" in spec:
        out = []
        for part in spec.replace("：", ":").replace("，", ",").split(","):
            code, _, weight = part.partition(":")
            if not code.strip() or not weight.strip():
                raise ValueError(f"无法解析基准成分 {part!r}，格式应为 代码:权重")
            out.append((code.strip(), float(weight)))
        return out
    if any(ch in spec for ch in "*×%％+＋"):
        return [(lookup_index_code(name), w) for name, w in parse_benchmark(spec)]
    return [(spec, 1.0)]


def _parse_columns(text: str | None) -> dict[str, str] | None:
    if not text:
        return None
    mapping = {}
    for part in text.replace("，", ",").split(","):
        src, _, dst = part.partition("=")
        if not src.strip() or not dst.strip():
            raise ValueError(f"无法解析列映射 {part!r}，格式应为 原列名=标准列名")
        mapping[src.strip()] = dst.strip()
    return mapping


def _trim_partial(obj, freq: str, start, end):
    """去掉首末不完整的期（期初早于 start 或期末晚于 end）。日频不处理。"""
    if str(freq).upper().startswith("D") or obj is None:
        return obj
    periods = obj.index.to_period(period_code(freq))
    keep = pd.Series(True, index=obj.index)
    if start is not None:
        keep &= periods.start_time >= pd.Timestamp(start)
    if end is not None:
        keep &= periods.end_time.normalize() <= pd.Timestamp(end)
    return obj[keep.to_numpy()]


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def resolve_benchmark_codes(parts, return_type: str = "total") -> tuple[list[tuple[str, float]], list[str]]:
    """按 --index-return-type 换算基准成分代码，返回 (新成分, 说明)。

    ``total``：有对应全收益指数的成分换成全收益代码（000300 → H00300），没有的沿用原代码
    并发出警告；``price``：沿用原代码。
    """
    if return_type not in ("total", "price"):
        raise ValueError(f"--index-return-type 须为 total 或 price，收到 {return_type!r}")
    out, notes = [], []
    for code, weight in parts:
        new = code
        if return_type == "total":
            total = total_return_code(code)
            if total is None:
                kind = {"price": "价格指数", "unknown": "收益类型未知"}.get(index_return_type(code), "")
                msg = f"基准成分 {code} 没有对应的全收益指数，沿用原代码（{kind}）"
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                notes.append(msg)
            else:
                new = total
        out.append((new, weight))
    return out, notes


def _format_parts(parts) -> str:
    return "、".join(f"{code}×{weight:.0%}" if len(parts) > 1 else code for code, weight in parts)


#: 汇率取数失败时的替代办法
FX_NONE_HINT = "可用 --fx none 不做汇率换算（口径将注明“未做汇率换算，基准收益含汇率差异”）"

#: 汇率中间价向前多取的自然日数，保证价格序列首日之前有中间价可取（asof）
_FX_LOOKBACK_DAYS = 15


def _is_foreign(currency) -> bool:
    return currency not in (CNY, "", None)


def _fx_converted_returns(code, currency, start, end, freq, cache, *, source="auto") -> pd.Series:
    """外币指数的人民币收益：日度价格 → 日度外币收益 → 按人民币汇率中间价换算（etl.fx.convert_returns，
    asof 取不晚于当日的最近一个中间价）→ 截取区间 → 合成为 freq。

    汇率取数失败或无法对应币种时报错，并提示可用 --fx none，不静默跳过。
    """
    from fundeval.etl.sources import akshare as aks

    price = aks.index_prices(code, start, end, source=source, **cache)
    if len(price) < 2:
        raise ValueError(f"指数 {code} 在区间内的收盘价不足两个交易日，无法计算收益")
    local = price_to_returns(price).rename(code)
    rates_start = price.index[0] - pd.Timedelta(days=_FX_LOOKBACK_DAYS)
    try:
        rates = aks.fx_rates(currency, rates_start, end, **cache)
    except Exception as exc:
        if aks.is_network_error(exc):
            raise aks.DataSourceUnavailable(
                f"{fx_mod.FX_SOURCE_LABEL}（{currency}）取数失败（{aks._describe_error(exc)}），"
                f"基准成分 {code} 无法换算为人民币收益；{FX_NONE_HINT}"
            ) from exc
        raise ValueError(f"基准成分 {code} 无法换算为人民币收益：{exc}；{FX_NONE_HINT}") from exc
    daily = fx_mod.convert_returns(local, rates, base_date=price.index[0])
    daily = daily.loc[pd.Timestamp(start) if start is not None else None : pd.Timestamp(end) if end is not None else None]
    if daily.isna().any():
        first = daily.index[daily.isna()][0]
        raise ValueError(
            f"基准成分 {code} 在 {first:%Y-%m-%d} 及之前没有{currency}的{fx_mod.FX_SOURCE_LABEL}，无法换算；{FX_NONE_HINT}"
        )
    out = to_frequency(daily, freq).rename(code) if freq != "D" else daily
    label = price.attrs.get("source_label", "未知")
    out.attrs = {**price.attrs, "source_label": f"{label}，按{fx_mod.FX_SOURCE_LABEL}（{currency}）换算为人民币"}
    return out


def _component_returns(code, start, end, freq, cache, *, source="auto", currency=None, fx="convert", name=None):
    """取一个指数成分的每期收益（去掉首末不完整的期）。

    - 币种不是人民币且 ``fx="convert"`` 时按 _fx_converted_returns 换算为人民币收益
    - 中债成分（``cbond:``）遇网络类异常时，在错误信息后补充 --benchmark-map 替代提示（不自动替换）
    """
    from fundeval.etl.sources import akshare as aks

    try:
        if fx == fx_mod.FX_CONVERT and _is_foreign(currency):
            r = _fx_converted_returns(code, currency, start, end, freq, cache, source=source)
        else:
            r = aks.index_returns(code, start, end, source=source, freq=None if freq == "D" else freq, **cache)
    except Exception as exc:
        if str(code).startswith("cbond:") and aks.is_network_error(exc):
            raise aks.DataSourceUnavailable(
                f"基准成分“{name or code}”（{code}）取数失败：{aks._describe_error(exc)}。{cbond_alternative_hint(name or code)}"
            ) from exc
        raise
    return _trim_partial(r, freq, start, end)


def _currency_texts(foreign, fx: str, k: int) -> tuple[str, list[str]]:
    """非人民币成分的口径文字与附注。foreign 为 [(名称, 代码, 币种), ...]。"""
    listed = "、".join(f"{name}（{code}，{cur}）" for name, code, cur in foreign)
    if fx == fx_mod.FX_CONVERT:
        caveat = fx_mod.converted_caveat(cur for _, _, cur in foreign)
        note = (
            f"基准含非人民币成分 {listed}：{caveat}。先在日度上按 r_CNY = (1 + r_外币) × S_t / S_t−1 − 1 换算"
            "（S 为每单位外币折合人民币，指数交易日取不晚于当日的最近一个中间价，不用未来数据），"
            f"再按期内复利合成为每期收益（K = {k}）。"
        )
        return f"含非人民币成分：{listed}；{caveat}", [note]
    return f"含非人民币成分：{listed}；{FX_CAVEAT}", [f"基准含非人民币成分 {listed}：{FX_CAVEAT}（--fx none）。"]


def _catalog_loader(cache: dict):
    """返回按需获取中证指数目录的无参函数（只取一次）。"""
    from fundeval.etl.sources import akshare as aks

    holder: dict[str, pd.DataFrame] = {}

    def catalog() -> pd.DataFrame:
        if "df" not in holder:
            holder["df"] = aks.index_catalog(**_cache_kwargs(cache))
        return holder["df"]

    return catalog


def _code_currency(code: str, catalog) -> tuple[str | None, str | None]:
    """指数代码的币种：指数表与中债代码为人民币；其余在中证指数目录中查“指数币种”。
    返回 (币种, 附注)；目录取不到或查无此代码时币种为 None，附注写明未能确认。"""
    if index_record(code) is not None or code.startswith("cbond:"):
        return CNY, None
    try:
        frame = catalog()
    except Exception as exc:
        msg = f"基准成分 {code} 的币种未能确认（中证指数目录获取失败：{type(exc).__name__}），按人民币计价处理，未做汇率换算"
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        return None, msg
    rows = frame[frame["指数代码"] == code]
    if rows.empty:
        return None, f"基准成分 {code} 不在中证指数目录中，币种未能确认，按人民币计价处理，未做汇率换算"
    return str(rows.iloc[0]["指数币种"]) or None, None


def _benchmark_from_akshare(spec, start, end, freq, cache, *, return_type="total", source="auto", fx="convert"):
    """联网取基准成分并合成，返回 (基准收益, labels 补充, 附注)。

    指数表以外的代码在中证指数目录中查币种（需要时才联网获取目录）；非人民币成分按 ``fx`` 换算或注明未换算。
    """
    parts = parse_benchmark_spec(spec)
    codes, notes = resolve_benchmark_codes(parts, return_type)
    catalog = _catalog_loader(cache)
    comps, foreign = {}, []
    for code, _ in codes:
        currency, note = _code_currency(code, catalog)
        if note:
            notes.append(note)
        if _is_foreign(currency):
            foreign.append((code, code, currency))
        comps[code] = _component_returns(code, start, end, freq, cache, source=source, currency=currency, fx=fx)
    bench = composite_benchmark(comps, {code: w for code, w in codes})
    label = spec if codes == parts else f"{spec}（实际使用 {_format_parts(codes)}）"
    labels = {
        "benchmark": label,
        "benchmark_return_type": benchmark_return_type([c for c, _ in codes]),
        "benchmark_source": "；".join(f"{code}：{r.attrs.get('source_label', '未知')}" for code, r in comps.items()),
    }
    if foreign:
        labels["benchmark_currency"], fx_notes = _currency_texts(foreign, fx, schema.periods_per_year(freq))
        notes += fx_notes
    return bench, labels, notes


def parse_style_spec(spec: str) -> dict[str, str]:
    """解析 --style：预设名（如 ``cn_equity``，见 etl.benchmark.style_preset）或逗号分隔的代码列表
    （``cash`` 表示无风险收益）。返回 {列名: 代码}；代码列表的列名即代码本身。
    ``auto`` 须先按基金类型换成预设名（etl.benchmark.auto_style_preset，build_report_inputs 中完成）。"""
    spec = spec.strip()
    if spec.lower() == STYLE_AUTO:
        raise ValueError("--style auto 须先按基金类型选择预设（需要 --fund）")
    if spec in STYLE_PRESETS:
        return style_preset(spec)
    codes = [c.strip() for c in spec.replace("，", ",").split(",")]
    if not codes or any(not c for c in codes):
        raise ValueError(f"无法解析 --style {spec!r}：应为预设名（{'、'.join(STYLE_PRESETS)}）或逗号分隔的代码列表")
    if len(set(codes)) != len(codes):
        raise ValueError(f"--style 代码重复：{spec!r}")
    if len(codes) < 2:
        raise ValueError("--style 至少需要两个风格资产（可含 cash）")
    return {c: c for c in codes}


def _style_from_akshare(spec, start, end, freq, cache, risk_free, index, *, return_type="total", source="auto"):
    """联网取风格指数收益，返回 (风格收益表, labels 补充, 附注)。cash 列取无风险收益。"""
    from fundeval.etl.sources import akshare as aks

    columns = parse_style_spec(spec)
    notes: list[str] = []
    frames, described, sources = {}, [], []
    for col, code in columns.items():
        if code == CASH:
            rf = risk_free if isinstance(risk_free, pd.Series) else pd.Series(float(risk_free), index=index)
            frames[col] = rf.reindex(index)
            described.append(f"{col}：无风险收益（与报告无风险收益同口径）")
            continue
        used = code
        if return_type == "total":
            used = total_return_code(code) or code
            if used == code and index_return_type(code) != "total":
                msg = f"风格指数 {code} 没有对应的全收益指数，沿用原代码"
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                notes.append(msg)
        r = aks.index_returns(used, start, end, source=source, freq=None if freq == "D" else freq, **cache)
        r = _trim_partial(r, freq, start, end)
        frames[col] = r
        kind = RETURN_TYPE_LABELS.get(index_return_type(used), "未知")
        described.append(f"{col}：{used}（{kind}）")
        sources.append(f"{used}：{r.attrs.get('source_label', '未知')}")
        if index_return_type(used) != "total":
            notes.append(f"风格指数 {col}（{used}）收益类型为{kind}，与含分红的基金净值口径可能不一致，权重与残差须谨慎解读")
    style = pd.concat(frames, axis=1)
    labels = {"style": "；".join(described), "style_source": "；".join(sources) or "—"}
    return style, labels, notes


def _factors_from_akshare(preset, start, end, freq, cache, risk_free, index, *, source="auto"):
    """联网取因子预设用到的指数并构造指数代理因子，返回 (因子收益表, labels 补充)。

    预设中的代码按原样取数，不做全收益替换（--index-return-type 只作用于基准与风格指数）：
    cn_index_proxy4 的 UMD 两条腿 H30260 与 000300 都是价格指数，保证分红口径一致。
    """
    from fundeval.etl.sources import akshare as aks

    if preset not in FACTOR_PRESETS:
        raise ValueError(f"未知的因子预设 {preset!r}，可选：{'、'.join(FACTOR_PRESET_NAMES)}")
    comps, sources = {}, []
    for code in preset_codes(preset):
        r = aks.index_returns(code, start, end, source=source, freq=None if freq == "D" else freq, **cache)
        comps[code] = _trim_partial(r, freq, start, end)
        sources.append(f"{code}：{r.attrs.get('source_label', '未知')}")
    frame = pd.concat(comps, axis=1).dropna()
    frame = frame.loc[frame.index.intersection(index)]
    rf = risk_free.reindex(frame.index) if isinstance(risk_free, pd.Series) else risk_free
    factors = index_proxy_factors(frame, rf, preset)
    desc = FACTOR_DESCRIPTIONS.get(preset, {})
    labels = {
        "factors": f"{preset}：" + "；".join(f"{name} = {desc.get(name, name)}" for name in factors.columns),
        "factor_type": INDEX_PROXY,
        "factor_source": "；".join(sources),
    }
    return factors, labels


def _factors_from_french(preset, start, end, freq, cache, index):
    """从 French 因子库取美国市场因子（ff3_us、carhart_us），返回 (因子收益表, 因子回归无风险收益 RF, labels 补充)。

    因子库只有月度与年度数据，``freq`` 须为 M 或 A；因子以日历期末为索引，与基金收益的期末日期对齐。
    """
    from fundeval.etl.sources import french

    key = str(freq).upper()[:1]
    if key not in ("M", "A"):
        raise ValueError(f"--factors {preset} 只支持 --freq M（或 A）：French 因子库只提供月度与年度数据，收到 {freq!r}")
    library = french.carhart_factors(key, **_cache_kwargs(cache))
    library = library.loc[pd.Timestamp(start) if start else None : pd.Timestamp(end) if end else None]
    library = library.loc[library.index.intersection(pd.DatetimeIndex(index))]
    factors, rf = us_market_factors(library, preset)
    desc = FACTOR_DESCRIPTIONS.get(preset, {})
    labels = {
        "factors": f"{preset}：" + "；".join(f"{name} = {desc.get(name, name)}" for name in factors.columns),
        "factor_type": US_MARKET,
        "factor_source": library.attrs.get("source_label", french.SOURCE_LABEL),
        "factor_risk_free": "French 因子库的 RF（美国 1 个月国库券收益，美元，每期）",
    }
    return factors, rf, labels


#: --benchmark 取基金合同业绩比较基准的写法
CONTRACT = "contract"


def parse_benchmark_map(spec) -> dict[str, str]:
    """解析 --benchmark-map：``"中债总指数=cbond:composite,中证香港300指数=H11164"``；也接受字典。"""
    if spec is None or spec == "":
        return {}
    if isinstance(spec, dict):
        return {str(k).strip(): str(v).strip() for k, v in spec.items()}
    out = {}
    for part in str(spec).replace("，", ",").split(","):
        if not part.strip():
            continue
        name, sep, code = part.replace("＝", "=").partition("=")
        if not sep or not name.strip() or not code.strip():
            raise ValueError(f"无法解析 --benchmark-map 的 {part!r}，格式应为 名称=代码，如 \"中债总指数=cbond:composite\"")
        out[name.strip()] = code.strip()
    return out


def _cache_kwargs(cache: dict) -> dict:
    """基金概况与指数目录接口接受的缓存参数（不含 cache_lag_days 等按日期检查的参数）。"""
    return {k: cache[k] for k in ("cache_dir", "use_cache", "refresh", "timeout", "total_timeout") if k in cache}


def _benchmark_from_contract(profile, start, end, freq, cache, index, *, return_type="total", source="auto",
                             overrides=None, deposit_rate=None, time_deposit_rate=None, fx="convert"):
    """按基金合同业绩比较基准合成复合基准，返回 (基准收益, labels 补充, 附注, 解析结果表)。

    成分按 etl.benchmark.resolve_benchmark 解析（指数目录 index_catalog 只在需要时联网获取）；
    指数成分经 akshare 取收益，现金成分按常数年化利率以 (1 + y)^(1/K) − 1 换算为每期，每期再平衡合成。
    非人民币成分在 ``fx="convert"``（默认）时按人民币汇率中间价换算为人民币收益（etl.fx），
    ``fx="none"`` 时沿用原币收益并注明“未做汇率换算”。中债成分取数失败时，错误信息补充
    --benchmark-map 的替代提示（etl.benchmark.cbond_alternative_hint），不自动替换。
    """
    from fundeval.etl import benchmark as bm

    text = profile.benchmark_text
    if not text:
        raise ValueError(f"基金 {profile.code} 的概况中没有业绩比较基准，请用 --benchmark 指定")
    catalog = _catalog_loader(cache)
    rates = {}
    if deposit_rate is not None:
        rates["deposit_rate"] = float(deposit_rate)
    if time_deposit_rate is not None:
        rates["time_deposit_rate"] = float(time_deposit_rate)
    comps = bm.resolve_benchmark(text, catalog, overrides, return_type=return_type, fx=fx, **rates)
    k = schema.periods_per_year(freq)
    series: dict[str, pd.Series] = {}
    used_sources: dict[str, str] = {}
    for c in comps:
        if c.is_cash:
            continue
        src = source if c.source in ("em", "csindex") else "auto"
        r = _component_returns(c.code, start, end, freq, cache, source=src, currency=c.currency, fx=fx, name=c.name)
        series[c.name] = r
        used_sources[c.name] = r.attrs.get("source_label", "未知")
    common = None
    for r in series.values():
        common = r.index if common is None else common.intersection(r.index)
    if common is None:
        common = pd.DatetimeIndex(index)
    for c in comps:
        if c.is_cash:
            series[c.name] = bm.cash_returns(c.rate, common, k)
            used_sources[c.name] = "常数"
    bench = composite_benchmark(series, {c.name: c.weight for c in comps})
    table = bm.resolution_table(comps)
    table["数据源"] = [used_sources.get(c.name, s) for c, s in zip(comps, table["数据源"])]

    def used(c) -> str:
        return f"现金 {c.rate:.2%}" if c.is_cash else c.code

    labels = {
        "benchmark": f"合同业绩比较基准：{text}（实际使用 {'、'.join(f'{used(c)}×{c.weight:.0%}' for c in comps)}）",
        "benchmark_return_type": bm.components_return_type(comps),
        "benchmark_source": "；".join(f"{used(c)}：{used_sources[c.name]}" for c in comps),
    }
    notes = []
    for c in comps:
        if c.is_cash:
            notes.append(f"基准成分“{c.name}”按{c.note}，以 (1 + y)^(1/{k}) − 1 换算为每期收益。")
    foreign = [(c.name, c.code, c.currency) for c in comps if _is_foreign(c.currency)]
    if foreign:
        labels["benchmark_currency"], fx_notes = _currency_texts(foreign, fx, k)
        notes += fx_notes
    return bench, labels, notes, table


@dataclass
class ReportOptions:
    """报告输入选项，字段与 ``fundeval report`` 的命令行参数一一对应（见 cli.build_parser）。

    ``risk_free`` 为调用方已取得的每期无风险收益（以期末日期为索引），给出时不再按 ``rf`` 取数；
    compare 用它让各基金使用同一无风险收益。``profile`` 为已取得的基金概况，给出时不再联网获取。
    """

    fund: str | None = None
    input: str | None = None
    columns: str | None = None
    date_col: str = schema.DATE
    percent: bool = False
    input_freq: str | None = None
    benchmark: str | None = None
    benchmark_map: Any = None
    deposit_rate: float | None = None
    time_deposit_rate: float | None = None
    index_return_type: str = "total"
    index_source: str = "auto"
    fx: str = fx_mod.FX_CONVERT
    start: Any = None
    end: Any = None
    freq: str = "M"
    rf: str | None = "auto"
    style: str | None = None
    factors: str | None = None
    style_window: int | None = None
    mar: float | None = None
    hac_lags: int | None = None
    use_t: bool = False
    target_active: float | None = None
    target_te: float | None = None
    window: int | None = None
    tolerance: float = CROSS_CHECK_TOLERANCE
    fees: str = DEFAULT_FEE_BASIS
    title: str | None = None
    cache_dir: Any = None
    no_cache: bool = False
    refresh: bool = False
    timeout: float | None = DEFAULT_TIMEOUT
    total_timeout: float | None = DEFAULT_TOTAL_TIMEOUT
    risk_free: pd.Series | None = None
    profile: Any = None

    @classmethod
    def from_namespace(cls, ns, **overrides) -> "ReportOptions":
        """从 argparse 的 Namespace（或任何带同名属性的对象）构造；缺少的字段取默认值。"""
        values = {f.name: getattr(ns, f.name) for f in fields(cls) if hasattr(ns, f.name)}
        return cls(**{**values, **overrides})

    def cache(self) -> dict:
        return {
            "cache_dir": self.cache_dir, "use_cache": not self.no_cache, "refresh": self.refresh,
            "timeout": self.timeout, "total_timeout": self.total_timeout,
        }


def _align_risk_free(rf: pd.Series, index: pd.DatetimeIndex, freq: str) -> pd.Series:
    """把以期末日期为索引的每期无风险收益按所属期对齐到 index（日频按日期）。"""
    target = pd.DatetimeIndex(index)
    if str(freq).upper().startswith("D"):
        values = rf.reindex(target).to_numpy()
    else:
        by_period = pd.Series(rf.to_numpy(), index=period_end_index(pd.DatetimeIndex(rf.index), freq))
        by_period = by_period[~by_period.index.duplicated(keep="last")]
        values = by_period.reindex(period_end_index(target, freq)).to_numpy()
    out = pd.Series(values, index=target, name=schema.RISK_FREE)
    out.attrs = dict(rf.attrs)
    return out


def fund_profile_or_none(code: str, cache: dict, *, required: bool):
    """取基金概况；``required=False``（已指定基准，概况只用于口径说明）时失败只发警告并返回 None。"""
    from fundeval.etl.sources import akshare as aks

    try:
        return aks.fund_profile(code, **_cache_kwargs(cache))
    except Exception as exc:
        if required:
            raise
        warnings.warn(f"基金 {code} 的概况获取失败（{type(exc).__name__}: {exc}），报告不含基金概况", RuntimeWarning, stacklevel=2)
        return None


def build_report_inputs(opts) -> dict:
    """按选项组装 evaluate 的参数。``opts`` 为 ReportOptions，或带同名属性的对象（如 argparse 的 Namespace）。

    给出 ``fund`` 而未给 ``benchmark`` 时，基准默认为 ``contract``：取基金概况中的合同业绩比较基准，
    按 etl.benchmark.resolve_benchmark 解析（``benchmark_map`` 为调用方指定的 {名称: 代码}），
    沿用 ``index_return_type`` 换全收益指数，合成复合基准；解析结果表与基金概况写入报告口径。

    ``fx``（默认 convert）：非人民币基准成分按国家外汇管理局人民币汇率中间价换算为人民币收益；none 时不换算，
    口径注明“未做汇率换算”。``style="auto"`` 按基金概况的基金类型选择风格预设（etl.benchmark.auto_style_preset），
    选择依据写入口径“风格预设”与附注。
    """
    if not isinstance(opts, ReportOptions):
        opts = ReportOptions.from_namespace(opts)
    freq = opts.freq.upper()
    k = schema.periods_per_year(freq)
    for flag, value in (("--timeout", opts.timeout), ("--total-timeout", opts.total_timeout)):
        if value is not None and value <= 0:
            raise ValueError(f"{flag} 须为正数（秒），收到 {value}")
    for flag, value in (("--deposit-rate", opts.deposit_rate), ("--time-deposit-rate", opts.time_deposit_rate)):
        if value is not None and not 0 <= value < 0.2:
            raise ValueError(f"{flag} 为年化小数（0.35% 写作 0.0035），收到 {value}")
    fx = fx_mod.check_mode(opts.fx)
    style_auto = bool(opts.style) and opts.style.strip().lower() == STYLE_AUTO
    if style_auto and not opts.fund:
        raise ValueError("--style auto 须与 --fund 一起使用（按基金概况中的基金类型选择风格预设）")
    cache = opts.cache()
    labels = {"fees": opts.fees}
    if opts.title:
        labels["title"] = opts.title
    cross_check = None
    risk_free = None
    notes: list[str] = []
    bench_kw = {"return_type": opts.index_return_type, "source": opts.index_source}
    rf_given = opts.rf is not None and opts.rf != "auto"
    benchmark_spec = opts.benchmark
    overrides = parse_benchmark_map(opts.benchmark_map)
    profile = None
    components = None
    if benchmark_spec is not None and benchmark_spec.strip().lower() == CONTRACT:
        benchmark_spec = CONTRACT
    if not opts.fund and (benchmark_spec == CONTRACT or overrides):
        raise ValueError("--benchmark contract 与 --benchmark-map 须与 --fund 一起使用（需要基金合同中的业绩比较基准）")
    if opts.fund and benchmark_spec is None:
        benchmark_spec = CONTRACT
    if overrides and benchmark_spec != CONTRACT:
        raise ValueError("--benchmark-map 只用于合同基准（--benchmark contract），指定代码时直接写在 --benchmark 中")

    if opts.fund:
        from fundeval.etl.sources import akshare as aks

        profile = opts.profile or fund_profile_or_none(
            opts.fund, cache, required=benchmark_spec == CONTRACT or style_auto
        )
        portfolio, cross_check = aks.fund_returns(
            opts.fund, opts.start, opts.end, freq=None if freq == "D" else freq,
            tolerance=opts.tolerance, return_check=True, **cache,
        )
        portfolio = _trim_partial(portfolio, freq, opts.start, opts.end)
        name = f" {profile.short_name}" if profile is not None and profile.short_name else ""
        labels["portfolio"] = f"基金 {opts.fund}{name}（单位净值加分红的总收益）"
        if benchmark_spec == CONTRACT:
            benchmark, extra, bench_notes, components = _benchmark_from_contract(
                profile, opts.start, opts.end, freq, cache, portfolio.index, overrides=overrides,
                deposit_rate=opts.deposit_rate, time_deposit_rate=opts.time_deposit_rate, fx=fx, **bench_kw,
            )
        else:
            benchmark, extra, bench_notes = _benchmark_from_akshare(
                benchmark_spec, opts.start, opts.end, freq, cache, fx=fx, **bench_kw
            )
        labels.update(extra)
        notes += bench_notes
        benchmark = _trim_partial(benchmark, freq, opts.start, opts.end)
    else:
        df = files.load_returns(
            opts.input, columns=_parse_columns(opts.columns), date_col=opts.date_col, percent=opts.percent
        )
        in_freq = (opts.input_freq or freq).upper()
        if in_freq != freq:
            df = to_frequency(df, freq)
        df = df.loc[pd.Timestamp(opts.start) if opts.start else None : pd.Timestamp(opts.end) if opts.end else None]
        portfolio = df[schema.PORTFOLIO]
        labels["portfolio"] = Path(opts.input).stem
        benchmark = df[schema.BENCHMARK] if schema.BENCHMARK in df.columns else None
        if benchmark_spec:
            if benchmark is not None:
                raise ValueError("输入文件已含 benchmark 列，不能再用 --benchmark 联网获取")
            benchmark, extra, bench_notes = _benchmark_from_akshare(
                benchmark_spec, opts.start, opts.end, freq, cache, fx=fx, **bench_kw
            )
            labels.update(extra)
            notes += bench_notes
        elif benchmark is not None:
            labels["benchmark"] = "输入文件的 benchmark 列"
        if schema.RISK_FREE in df.columns:
            if rf_given or opts.risk_free is not None:
                raise ValueError("输入文件已含 risk_free 列，不能再指定 --rf")
            risk_free = df[schema.RISK_FREE]
            labels["risk_free"] = "输入文件的 risk_free 列（每期）"
    # 组合与基准的对齐交给 evaluate：口径写明各自期数与共同期数，丢弃的期列入附注

    if risk_free is None and opts.risk_free is not None:
        risk_free = _align_risk_free(opts.risk_free, portfolio.index, freq)
        labels["risk_free"] = opts.risk_free.attrs.get("description") or "调用方给出的每期无风险收益"
        if risk_free.isna().any():
            raise ValueError("给出的无风险收益未覆盖全部观测期，请检查区间")
    if risk_free is None:
        from fundeval.etl.sources import akshare as aks

        # --rf auto：--fund 时按 Shibor 3M → 国债 2 年联网获取；本地文件不联网，按 0 计
        rf_arg = opts.rf if rf_given else ("auto" if opts.fund else "0")
        if _is_number(rf_arg):
            y = float(rf_arg)
            risk_free = ((1 + y) ** (1 / k) - 1) if y else 0.0
            labels["risk_free"] = aks.describe_rate_source(y, freq)
        else:
            risk_free = aks.risk_free_returns(opts.start, opts.end, freq, source=rf_arg, index=portfolio.index, **cache)
            labels["risk_free"] = risk_free.attrs.get("description") or aks.describe_rate_source(rf_arg, freq)
            if risk_free.isna().any():
                raise ValueError(f"无风险利率 {rf_arg} 未覆盖全部观测期，请检查区间或改用常数 --rf")

    style_returns = None
    if opts.style_window is not None and not opts.style:
        raise ValueError("--style-window 须与 --style 一起给出")
    if opts.style:
        style_spec = opts.style
        if style_auto:
            if profile is None:  # pragma: no cover - required=True 时取不到概况会直接报错
                raise ValueError("--style auto 需要基金概况中的基金类型")
            style_spec, basis = auto_style_preset(profile.fund_type)
            labels["style_preset"] = f"{style_spec}（--style auto：{basis}）"
            notes.append(f"风格预设按基金类型自动选择：{basis}。")
        style_returns, style_labels, style_notes = _style_from_akshare(
            style_spec, opts.start, opts.end, freq, cache, risk_free, portfolio.index, **bench_kw
        )
        labels.update(style_labels)
        notes += style_notes

    factor_returns = None
    factor_risk_free = None
    if opts.factors:
        if opts.factors in US_FACTOR_PRESETS:
            factor_returns, factor_risk_free, factor_labels = _factors_from_french(
                opts.factors, opts.start, opts.end, freq, cache, portfolio.index
            )
            notes.append(
                f"多因子分解使用美国市场因子（{opts.factors}），回归的超额收益按因子库 RF（美元）计算；"
                "基金收益为人民币计价，回归残差与 Alpha 含人民币兑美元汇率变动的影响。"
            )
        else:
            factor_returns, factor_labels = _factors_from_akshare(
                opts.factors, opts.start, opts.end, freq, cache, risk_free, portfolio.index, source=opts.index_source
            )
        labels.update(factor_labels)

    targets = None
    if opts.target_te is not None:
        if opts.target_active is None or opts.window is None:
            raise ValueError("--target-te 须与 --target-active、--window 一起给出")
        targets = {"target_active_return": opts.target_active, "target_te": opts.target_te, "window": opts.window}

    return dict(
        returns=portfolio,
        benchmark=benchmark,
        risk_free=risk_free,
        periods_per_year=k,
        mar=opts.mar,
        hac_lags=opts.hac_lags,
        monitor_targets=targets,
        labels=labels,
        cross_check=cross_check,
        use_t=True if opts.use_t else None,
        notes=notes,
        style_returns=style_returns,
        style_window=opts.style_window,
        factor_returns=factor_returns,
        factor_risk_free=factor_risk_free,
        profile=profile,
        benchmark_components=components,
    )
