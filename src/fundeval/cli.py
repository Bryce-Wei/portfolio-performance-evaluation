"""命令行入口：``fundeval report`` 一键生成评价报告，``fundeval compare`` 多基金横向对比（正文第九、十部分）。

示例::

    fundeval report --fund 110020 --start 2021-01-01 --end 2025-12-31 --out report.md

    fundeval compare --funds 110011,110020,000001 --start 2021-01-01 --end 2025-12-31 \\
        --benchmark-map "中债总指数=cbond:composite" --sort sharpe --out compare.xlsx

    fundeval report --fund 110011 --benchmark "000300:0.8,H11001:0.2" \\
        --start 2021-01-01 --end 2025-12-31 --freq M --rf shibor3m --out report.md

    fundeval report --input returns.csv --freq M --out report.xlsx

只给 ``--fund`` 时基准默认为 ``contract``：取基金概况中的合同业绩比较基准，按 etl.benchmark.resolve_benchmark
解析（``--benchmark-map`` 指定无法自动解析的成分，``--deposit-rate`` / ``--time-deposit-rate`` 覆盖存款利率），
合成复合基准；取数与解析在 report.inputs 中实现，compare 共用。

``--fund`` 通过 akshare 取数（需 ``pip install "fundeval[data]"``）；``--input`` 读取本地
CSV / Excel（走 etl.sources.files），不依赖网络。

基准成分默认换成全收益指数（``--index-return-type total``，如 000300 → H00300），因为基金净值
含分红而价格指数不含；指数默认 ``--index-source auto``（东方财富失败时改用中证指数官网），
无风险利率默认 ``--rf auto``（Shibor 3M 失败时改用国债 2 年，也可写顺序列表如 ``cgb2y,shibor3m``）。
自动切换时非最后一个候选只尝试 1 次。自动切换与旧缓存回退写入报告附注；网络失败时输出一行
中文提示与替代办法，返回码 2。

``--style cn_equity``（或逗号分隔的代码列表，``cash`` 表示无风险收益）在报告“收益来源”一节做
Sharpe 风格分析，``--style-window 36`` 另附滚动权重；风格指数默认用全收益代码。

``--factors cn_index_proxy`` 在“Alpha 质量”一节做多因子分解（attribution.factor）：MKT = H00300 − 无风险收益，
SMB = H00852 − H00300，HML = H00919 − H00918，均取中证指数官网全收益指数；这是指数代理因子，
与 Fama–French 分组构造不同。多因子回归沿用 ``--hac-lags`` 与 ``--use-t``。
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
from collections.abc import Sequence
from pathlib import Path

from fundeval.attribution.factor import FACTOR_PRESETS
from fundeval.etl import schema
from fundeval.etl.benchmark import DEMAND_DEPOSIT_RATE, DEPOSIT_RATE_SOURCE, STYLE_PRESETS, TIME_DEPOSIT_RATE
from fundeval.etl.quality import CROSS_CHECK_TOLERANCE
from fundeval.etl.sources.akshare import DEFAULT_TIMEOUT, DEFAULT_TOTAL_TIMEOUT
from fundeval.report import evaluate, to_excel, to_markdown
from fundeval.report.compare import SORT_KEYS, compare
from fundeval.report.inputs import (  # noqa: F401  （parse_* 与 resolve_benchmark_codes 供外部与测试沿用 cli.xxx）
    ReportOptions,
    build_report_inputs,
    parse_benchmark_map,
    parse_benchmark_spec,
    parse_style_spec,
    resolve_benchmark_codes,
)
from fundeval.report.summary import DEFAULT_FEE_BASIS

#: 网络失败时提示的替代办法
NETWORK_HINT = "可改用 --index-source csindex（中证指数官网）、常数无风险利率 --rf 0.018，稍后重试，或加 --refresh 重新拉取缓存"


def cmd_report(args) -> int:
    from fundeval.etl.sources import akshare as aks

    with aks.collect_notes() as source_notes:
        inputs = build_report_inputs(ReportOptions.from_namespace(args))
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


def cmd_compare(args) -> int:
    codes = [c.strip() for c in args.funds.replace("，", ",").split(",") if c.strip()]
    opts = ReportOptions.from_namespace(args, fund=None, input=None)
    result = compare(codes, args.start, args.end, freq=args.freq, benchmark=args.benchmark or "contract",
                     sort=args.sort, options=opts)
    out = args.out
    if out is None:
        sys.stdout.write(result.to_markdown())
        return 0
    suffix = Path(out).suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        result.to_excel(out)
    elif suffix in {".md", ".markdown", ".txt"}:
        result.to_markdown(out)
    else:
        raise ValueError(f"--out 只支持 .md 或 .xlsx，收到 {out!r}")
    print(f"对比表已写入 {out}（成功 {len(result.reports)} 只，失败 {len(result.failures)} 只）", file=sys.stderr)
    return 0


def _add_benchmark_args(p: argparse.ArgumentParser, *, compare_mode: bool = False) -> None:
    if compare_mode:
        p.add_argument(
            "--benchmark", default="contract",
            help="基准：contract（默认，每只基金各自的合同业绩比较基准），或统一的 \"000300:0.8,H11001:0.2\"、指数代码、合同文字",
        )
    else:
        p.add_argument(
            "--benchmark",
            help="基准：contract（给出 --fund 时的默认值，取基金合同业绩比较基准并解析）、\"000300:0.8,H11001:0.2\"、"
            "单个指数代码，或合同基准文字",
        )
    p.add_argument(
        "--benchmark-map",
        help="合同基准中无法自动解析的成分：\"名称=代码\"，逗号分隔，如 \"中债总指数=cbond:composite\"；现金可写 cash:0.02",
    )
    p.add_argument(
        "--deposit-rate", type=float,
        help=f"合同基准中活期存款利率的年化数值（小数，默认 {DEMAND_DEPOSIT_RATE}，{DEPOSIT_RATE_SOURCE}）",
    )
    p.add_argument(
        "--time-deposit-rate", type=float,
        help=f"合同基准中一年期定期存款利率的年化数值（小数，默认 {TIME_DEPOSIT_RATE}，{DEPOSIT_RATE_SOURCE}）",
    )
    p.add_argument(
        "--index-return-type", choices=("total", "price"), default="total",
        help="基准指数收益类型：total（默认，换成全收益指数，如 000300 → H00300）或 price（价格指数，不含成分股分红）",
    )
    p.add_argument(
        "--index-source", choices=("auto", "em", "csindex"), default="auto",
        help="指数数据源：auto（默认，东方财富失败时改用中证指数官网）、em、csindex",
    )


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--start", help="起始日期，如 2021-01-01")
    p.add_argument("--end", help="截止日期，如 2025-12-31")
    p.add_argument("--freq", default="M", help="评价频率 D/W/M/Q（默认 M）")
    p.add_argument(
        "--rf", default="auto",
        help="无风险收益：auto（默认；--fund 时等价于 shibor3m,cgb2y，本地文件按 0 计）、"
        "shibor3m、shibor1m、shibor_on、cgb2y、cgb10y，逗号分隔的顺序列表如 cgb2y,shibor3m"
        "（依次尝试，非最后一个只试 1 次），或常数年化利率如 0.018",
    )
    p.add_argument(
        "--style",
        help="风格分析：预设名（" + "、".join(STYLE_PRESETS) + "）或逗号分隔的指数代码，cash 表示无风险收益；"
        "指数按 --index-return-type 默认换成全收益代码",
    )
    p.add_argument(
        "--factors", choices=tuple(FACTOR_PRESETS),
        help="多因子分解的因子预设：cn_index_proxy（MKT = H00300 − rf，SMB = H00852 − H00300，HML = H00919 − H00918，"
        "指数代理因子，与学术因子不可直接比较）；回归沿用 --hac-lags 与 --use-t",
    )
    p.add_argument("--style-window", type=int, help="滚动风格分析的窗口期数（不小于风格资产数 + 2）")
    p.add_argument("--mar", type=float, help="Sortino 的最低可接受收益（每期，小数）；缺省取无风险收益")
    p.add_argument("--hac-lags", type=int, help="回归使用 Newey–West 标准误的滞后阶数")
    p.add_argument("--use-t", action="store_true", help="HAC 回归的 p 值与置信区间也用 t 分布（小样本建议）")
    p.add_argument(
        "--tolerance", type=float, default=CROSS_CHECK_TOLERANCE,
        help=f"净值与日增长率交叉核对容差（默认 {CROSS_CHECK_TOLERANCE}，即 {CROSS_CHECK_TOLERANCE * 1e4:.0f} 个基点）",
    )
    p.add_argument("--fees", default=DEFAULT_FEE_BASIS, help=f"费用口径说明（默认“{DEFAULT_FEE_BASIS}”）")
    p.add_argument("--cache-dir", help="akshare 原始数据缓存目录（默认 ~/.fundeval/cache）")
    p.add_argument("--no-cache", action="store_true", help="不读写缓存")
    p.add_argument("--refresh", action="store_true", help="忽略已有缓存，重新请求并覆盖")
    p.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT,
        help=f"单个 HTTP 请求（连接与读取）的超时秒数（默认 {DEFAULT_TIMEOUT:g}）；分页接口每页单独计时，"
        "超时按网络异常重试或切换数据源",
    )
    p.add_argument(
        "--total-timeout", type=float, default=DEFAULT_TOTAL_TIMEOUT,
        help=f"每次尝试（一次接口调用，可含多个分页请求）的兜底总时限秒数（默认 {DEFAULT_TOTAL_TIMEOUT:g}），"
        "每次重试重新计时；超时后后台线程可能仍在运行，其结果被丢弃",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fundeval", description="基金与投资组合绩效评估")
    sub = parser.add_subparsers(dest="command", required=True)
    rp = sub.add_parser("report", help="生成评价报告（Markdown 或 Excel）")
    src = rp.add_mutually_exclusive_group(required=True)
    src.add_argument("--fund", help="基金代码，经 akshare 取单位净值与分红（需 fundeval[data]）；未给 --benchmark 时用合同基准")
    src.add_argument("--input", help="本地 CSV / Excel 收益表，列名为 date、portfolio 及可选 benchmark、risk_free")
    rp.add_argument("--columns", help="列名映射，如 \"月份=date,组合收益=portfolio,基准收益=benchmark\"")
    rp.add_argument("--date-col", default=schema.DATE, help="映射后的日期列名（默认 date）")
    rp.add_argument("--percent", action="store_true", help="输入文件中的数值为百分数")
    rp.add_argument("--input-freq", help="输入文件的频率（D/W/M/Q），与 --freq 不同时先按期内复利合成")
    _add_benchmark_args(rp)
    _add_common_args(rp)
    rp.add_argument("--target-active", type=float, help="监控：年化目标主动收益（小数）")
    rp.add_argument("--target-te", type=float, help="监控：年化目标 TE（小数）")
    rp.add_argument("--window", type=int, help="监控：窗口期数")
    rp.add_argument("--title", help="报告标题")
    rp.add_argument("--out", help="输出文件（.md 或 .xlsx）；缺省时把 Markdown 打印到标准输出")
    rp.set_defaults(func=cmd_report)

    cp = sub.add_parser("compare", help="多基金横向对比（同一区间、同一无风险收益，逐只评价）")
    cp.add_argument("--funds", required=True, help="基金代码，逗号分隔，如 110011,110020,000001")
    _add_benchmark_args(cp, compare_mode=True)
    _add_common_args(cp)
    cp.add_argument(
        "--sort", choices=tuple(SORT_KEYS),
        help="按单一指标排序（默认按输入顺序，不做综合打分）；表下注明不同类型的基金不宜直接比较",
    )
    cp.add_argument("--out", help="输出文件（.md 或 .xlsx）；缺省时把 Markdown 打印到标准输出")
    cp.set_defaults(func=cmd_compare)
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
