"""命令行入口：``fundeval report`` 一键生成评价报告（正文第九、十部分）。

示例::

    fundeval report --fund 110011 --benchmark "000300:0.8,H11001:0.2" \\
        --start 2021-01-01 --end 2025-12-31 --freq M --rf shibor3m --out report.md

    fundeval report --input returns.csv --freq M --out report.xlsx

``--fund`` 通过 akshare 取数（需 ``pip install "fundeval[data]"``）；``--input`` 读取本地
CSV / Excel（走 etl.sources.files），不依赖网络。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from fundeval.etl import schema
from fundeval.etl.benchmark import composite_benchmark, lookup_index_code, parse_benchmark
from fundeval.etl.returns import period_code, to_frequency
from fundeval.etl.sources import files
from fundeval.report import evaluate, to_excel, to_markdown


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


def _benchmark_from_akshare(spec, start, end, freq, cache):
    from fundeval.etl.sources import akshare as aks

    parts = parse_benchmark_spec(spec)
    comps = {code: aks.index_returns(code, start, end, freq=None if freq == "D" else freq, **cache) for code, _ in parts}
    return composite_benchmark(comps, {code: w for code, w in parts})


def _build_report_inputs(args):
    freq = args.freq.upper()
    k = schema.periods_per_year(freq)
    cache = {"cache_dir": args.cache_dir, "use_cache": not args.no_cache, "refresh": args.refresh}
    labels = {"fees": args.fees}
    if args.title:
        labels["title"] = args.title
    cross_check = None
    risk_free = None

    if args.fund:
        from fundeval.etl.sources import akshare as aks

        portfolio, cross_check = aks.fund_returns(
            args.fund, args.start, args.end, freq=None if freq == "D" else freq,
            tolerance=args.tolerance, return_check=True, **cache,
        )
        labels["portfolio"] = f"基金 {args.fund}（单位净值加分红的总收益）"
        benchmark = None
        if args.benchmark:
            benchmark = _benchmark_from_akshare(args.benchmark, args.start, args.end, freq, cache)
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
            benchmark = _benchmark_from_akshare(args.benchmark, args.start, args.end, freq, cache)
        if schema.RISK_FREE in df.columns:
            if args.rf is not None:
                raise ValueError("输入文件已含 risk_free 列，不能再指定 --rf")
            risk_free = df[schema.RISK_FREE]
            labels["risk_free"] = "输入文件的 risk_free 列（每期）"

    if args.fund:
        portfolio = _trim_partial(portfolio, freq, args.start, args.end)
        benchmark = _trim_partial(benchmark, freq, args.start, args.end)
    if benchmark is not None:
        common = portfolio.index.intersection(benchmark.index)
        portfolio, benchmark = portfolio.loc[common], benchmark.loc[common]
        labels["benchmark"] = args.benchmark or "输入文件的 benchmark 列"

    if risk_free is None:
        rf_arg = args.rf if args.rf is not None else ("shibor3m" if args.fund else "0")
        if _is_number(rf_arg):
            y = float(rf_arg)
            risk_free = ((1 + y) ** (1 / k) - 1) if y else 0.0
            labels["risk_free"] = f"常数年化 {y:.2%}，按 (1 + y)^(1/{k}) − 1 换算为每期" if y else "未提供，按 0 计"
        else:
            from fundeval.etl.sources import akshare as aks

            risk_free = aks.risk_free_returns(args.start, args.end, freq, source=rf_arg, index=portfolio.index, **cache)
            labels["risk_free"] = aks.describe_rate_source(rf_arg, freq)
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
    )


def cmd_report(args) -> int:
    inputs = _build_report_inputs(args)
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
    rp.add_argument("--start", help="起始日期，如 2021-01-01")
    rp.add_argument("--end", help="截止日期，如 2025-12-31")
    rp.add_argument("--freq", default="M", help="评价频率 D/W/M/Q（默认 M）")
    rp.add_argument("--rf", help="无风险收益：shibor3m、shibor1m、shibor_on、cgb2y、cgb10y，或常数年化利率如 0.018")
    rp.add_argument("--mar", type=float, help="Sortino 的最低可接受收益（每期，小数）；缺省取无风险收益")
    rp.add_argument("--hac-lags", type=int, help="回归使用 Newey–West 标准误的滞后阶数")
    rp.add_argument("--target-active", type=float, help="监控：年化目标主动收益（小数）")
    rp.add_argument("--target-te", type=float, help="监控：年化目标 TE（小数）")
    rp.add_argument("--window", type=int, help="监控：窗口期数")
    rp.add_argument("--tolerance", type=float, default=0.0005, help="净值与日增长率交叉核对容差（默认 0.0005，即 5 个基点）")
    rp.add_argument("--fees", default="费用后净值", help="费用口径说明（默认“费用后净值”）")
    rp.add_argument("--title", help="报告标题")
    rp.add_argument("--out", help="输出文件（.md 或 .xlsx）；缺省时把 Markdown 打印到标准输出")
    rp.add_argument("--cache-dir", help="akshare 原始数据缓存目录（默认 ~/.fundeval/cache）")
    rp.add_argument("--no-cache", action="store_true", help="不读写缓存")
    rp.add_argument("--refresh", action="store_true", help="忽略已有缓存，重新请求并覆盖")
    rp.set_defaults(func=cmd_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, KeyError, ImportError, RuntimeError, FileNotFoundError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
