"""命令行入口：``fundeval report`` 一键生成评价报告（正文第九、十部分）。

示例::

    fundeval report --fund 110011 --benchmark "000300:0.8,H11001:0.2" \\
        --start 2021-01-01 --end 2025-12-31 --freq M --rf shibor3m --out report.md

    fundeval report --input returns.csv --freq M --out report.xlsx

``--fund`` 通过 akshare 取数（需 ``pip install "fundeval[data]"``）；``--input`` 读取本地
CSV / Excel（走 etl.sources.files），不依赖网络。

基准成分默认换成全收益指数（``--index-return-type total``，如 000300 → H00300），因为基金净值
含分红而价格指数不含；指数默认 ``--index-source auto``（东方财富失败时改用中证指数官网），
无风险利率默认 ``--rf auto``（Shibor 3M 失败时改用国债 2 年）。自动切换与旧缓存回退写入
报告附注；网络失败时输出一行中文提示与替代办法，返回码 2。
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import warnings
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from fundeval.etl import schema
from fundeval.etl.benchmark import (
    benchmark_return_type,
    composite_benchmark,
    index_return_type,
    lookup_index_code,
    parse_benchmark,
    total_return_code,
)
from fundeval.etl.quality import CROSS_CHECK_TOLERANCE
from fundeval.etl.returns import period_code, to_frequency
from fundeval.etl.sources import files
from fundeval.report import evaluate, to_excel, to_markdown
from fundeval.report.summary import DEFAULT_FEE_BASIS

#: 网络失败时提示的替代办法
NETWORK_HINT = "可改用 --index-source csindex（中证指数官网）、常数无风险利率 --rf 0.018，稍后重试，或加 --refresh 重新拉取缓存"


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


def _benchmark_from_akshare(spec, start, end, freq, cache, *, return_type="total", source="auto"):
    """联网取基准成分并合成，返回 (基准收益, labels 补充, 附注)。"""
    from fundeval.etl.sources import akshare as aks

    parts = parse_benchmark_spec(spec)
    codes, notes = resolve_benchmark_codes(parts, return_type)
    comps = {
        code: aks.index_returns(code, start, end, source=source, freq=None if freq == "D" else freq, **cache)
        for code, _ in codes
    }
    bench = composite_benchmark(comps, {code: w for code, w in codes})
    label = spec if codes == parts else f"{spec}（实际使用 {_format_parts(codes)}）"
    labels = {
        "benchmark": label,
        "benchmark_return_type": benchmark_return_type([c for c, _ in codes]),
        "benchmark_source": "；".join(f"{code}：{r.attrs.get('source_label', '未知')}" for code, r in comps.items()),
    }
    return bench, labels, notes


def _build_report_inputs(args):
    freq = args.freq.upper()
    k = schema.periods_per_year(freq)
    cache = {"cache_dir": args.cache_dir, "use_cache": not args.no_cache, "refresh": args.refresh}
    labels = {"fees": args.fees}
    if args.title:
        labels["title"] = args.title
    cross_check = None
    risk_free = None
    notes: list[str] = []
    bench_kw = {"return_type": args.index_return_type, "source": args.index_source}
    rf_given = args.rf is not None and args.rf != "auto"

    if args.fund:
        from fundeval.etl.sources import akshare as aks

        portfolio, cross_check = aks.fund_returns(
            args.fund, args.start, args.end, freq=None if freq == "D" else freq,
            tolerance=args.tolerance, return_check=True, **cache,
        )
        labels["portfolio"] = f"基金 {args.fund}（单位净值加分红的总收益）"
        benchmark = None
        if args.benchmark:
            benchmark, extra, bench_notes = _benchmark_from_akshare(args.benchmark, args.start, args.end, freq, cache, **bench_kw)
            labels.update(extra)
            notes += bench_notes
    else:
        df = files.load_returns(
            args.input, columns=_parse_columns(args.columns), date_col=args.date_col, percent=args.percent
        )
        in_freq = (args.input_freq or freq).upper()
        if in_freq != freq:
            df = to_frequency(df, freq)
        df = df.loc[pd.Timestamp(args.start) if args.start else None : pd.Timestamp(args.end) if args.end else None]
        portfolio = df[schema.PORTFOLIO]
        labels["portfolio"] = Path(args.input).stem
        benchmark = df[schema.BENCHMARK] if schema.BENCHMARK in df.columns else None
        if args.benchmark:
            if benchmark is not None:
                raise ValueError("输入文件已含 benchmark 列，不能再用 --benchmark 联网获取")
            benchmark, extra, bench_notes = _benchmark_from_akshare(args.benchmark, args.start, args.end, freq, cache, **bench_kw)
            labels.update(extra)
            notes += bench_notes
        elif benchmark is not None:
            labels["benchmark"] = "输入文件的 benchmark 列"
        if schema.RISK_FREE in df.columns:
            if rf_given:
                raise ValueError("输入文件已含 risk_free 列，不能再指定 --rf")
            risk_free = df[schema.RISK_FREE]
            labels["risk_free"] = "输入文件的 risk_free 列（每期）"

    if args.fund:
        portfolio = _trim_partial(portfolio, freq, args.start, args.end)
        benchmark = _trim_partial(benchmark, freq, args.start, args.end)
    # 组合与基准的对齐交给 evaluate：口径写明各自期数与共同期数，丢弃的期列入附注

    if risk_free is None:
        from fundeval.etl.sources import akshare as aks

        # --rf auto：--fund 时按 Shibor 3M → 国债 2 年联网获取；本地文件不联网，按 0 计
        rf_arg = args.rf if rf_given else ("auto" if args.fund else "0")
        if _is_number(rf_arg):
            y = float(rf_arg)
            risk_free = ((1 + y) ** (1 / k) - 1) if y else 0.0
            labels["risk_free"] = aks.describe_rate_source(y, freq)
        else:
            risk_free = aks.risk_free_returns(args.start, args.end, freq, source=rf_arg, index=portfolio.index, **cache)
            labels["risk_free"] = risk_free.attrs.get("description") or aks.describe_rate_source(rf_arg, freq)
            if risk_free.isna().any():
                raise ValueError(f"无风险利率 {rf_arg} 未覆盖全部观测期，请检查区间或改用常数 --rf")

    targets = None
    if args.target_te is not None:
        if args.target_active is None or args.window is None:
            raise ValueError("--target-te 须与 --target-active、--window 一起给出")
        targets = {"target_active_return": args.target_active, "target_te": args.target_te, "window": args.window}

    return dict(
        returns=portfolio,
        benchmark=benchmark,
        risk_free=risk_free,
        periods_per_year=k,
        mar=args.mar,
        hac_lags=args.hac_lags,
        monitor_targets=targets,
        labels=labels,
        cross_check=cross_check,
        use_t=True if args.use_t else None,
        notes=notes,
    )


def cmd_report(args) -> int:
    from fundeval.etl.sources import akshare as aks

    with aks.collect_notes() as source_notes:
        inputs = _build_report_inputs(args)
    inputs["notes"] = source_notes + inputs["notes"]
    report = evaluate(**inputs)
    out = args.out
    if out is None:
        sys.stdout.write(to_markdown(report))
        return 0
    suffix = Path(out).suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        to_excel(report, out)
    elif suffix in {".md", ".markdown", ".txt"}:
        to_markdown(report, out)
    else:
        raise ValueError(f"--out 只支持 .md 或 .xlsx，收到 {out!r}")
    print(f"报告已写入 {out}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fundeval", description="基金与投资组合绩效评估")
    sub = parser.add_subparsers(dest="command", required=True)
    rp = sub.add_parser("report", help="生成评价报告（Markdown 或 Excel）")
    src = rp.add_mutually_exclusive_group(required=True)
    src.add_argument("--fund", help="基金代码，经 akshare 取单位净值与分红（需 fundeval[data]）")
    src.add_argument("--input", help="本地 CSV / Excel 收益表，列名为 date、portfolio 及可选 benchmark、risk_free")
    rp.add_argument("--columns", help="列名映射，如 \"月份=date,组合收益=portfolio,基准收益=benchmark\"")
    rp.add_argument("--date-col", default=schema.DATE, help="映射后的日期列名（默认 date）")
    rp.add_argument("--percent", action="store_true", help="输入文件中的数值为百分数")
    rp.add_argument("--input-freq", help="输入文件的频率（D/W/M/Q），与 --freq 不同时先按期内复利合成")
    rp.add_argument("--benchmark", help="基准：\"000300:0.8,H11001:0.2\"、单个指数代码，或合同基准文字")
    rp.add_argument(
        "--index-return-type", choices=("total", "price"), default="total",
        help="基准指数收益类型：total（默认，换成全收益指数，如 000300 → H00300）或 price（价格指数，不含成分股分红）",
    )
    rp.add_argument(
        "--index-source", choices=("auto", "em", "csindex"), default="auto",
        help="指数数据源：auto（默认，东方财富失败时改用中证指数官网）、em、csindex",
    )
    rp.add_argument("--start", help="起始日期，如 2021-01-01")
    rp.add_argument("--end", help="截止日期，如 2025-12-31")
    rp.add_argument("--freq", default="M", help="评价频率 D/W/M/Q（默认 M）")
    rp.add_argument(
        "--rf", default="auto",
        help="无风险收益：auto（默认；--fund 时按 shibor3m → cgb2y 依次尝试，本地文件按 0 计）、"
        "shibor3m、shibor1m、shibor_on、cgb2y、cgb10y，或常数年化利率如 0.018",
    )
    rp.add_argument("--mar", type=float, help="Sortino 的最低可接受收益（每期，小数）；缺省取无风险收益")
    rp.add_argument("--hac-lags", type=int, help="回归使用 Newey–West 标准误的滞后阶数")
    rp.add_argument("--use-t", action="store_true", help="HAC 回归的 p 值与置信区间也用 t 分布（小样本建议）")
    rp.add_argument("--target-active", type=float, help="监控：年化目标主动收益（小数）")
    rp.add_argument("--target-te", type=float, help="监控：年化目标 TE（小数）")
    rp.add_argument("--window", type=int, help="监控：窗口期数")
    rp.add_argument(
        "--tolerance", type=float, default=CROSS_CHECK_TOLERANCE,
        help=f"净值与日增长率交叉核对容差（默认 {CROSS_CHECK_TOLERANCE}，即 {CROSS_CHECK_TOLERANCE * 1e4:.0f} 个基点）",
    )
    rp.add_argument("--fees", default=DEFAULT_FEE_BASIS, help=f"费用口径说明（默认“{DEFAULT_FEE_BASIS}”）")
    rp.add_argument("--title", help="报告标题")
    rp.add_argument("--out", help="输出文件（.md 或 .xlsx）；缺省时把 Markdown 打印到标准输出")
    rp.add_argument("--cache-dir", help="akshare 原始数据缓存目录（默认 ~/.fundeval/cache）")
    rp.add_argument("--no-cache", action="store_true", help="不读写缓存")
    rp.add_argument("--refresh", action="store_true", help="忽略已有缓存，重新请求并覆盖")
    rp.set_defaults(func=cmd_report)
    return parser


def _is_network_error(exc: BaseException) -> bool:
    from fundeval.etl.sources.akshare import is_network_error

    return is_network_error(exc) or isinstance(exc, urllib.error.URLError)


def _one_line(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        if _is_network_error(exc):
            print(f"错误：网络请求失败（{_one_line(exc)}）。{NETWORK_HINT}", file=sys.stderr)
            return 2
        if isinstance(exc, (ValueError, KeyError, ImportError, RuntimeError, FileNotFoundError)):
            print(f"错误：{exc}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
