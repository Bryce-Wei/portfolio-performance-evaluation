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

合同基准中的非人民币成分（如中证香港300，港元）默认 ``--fx convert``：按国家外汇管理局人民币汇率中间价
先在日度上换算为人民币收益再合成（etl.fx），汇率取数失败时报错并提示 ``--fx none``。``--style auto`` 按基金类型
选择风格预设；``--charts`` 生成 PNG 图表（report.charts，需 ``fundeval[plot]``）。中债成分取数失败时，
错误信息提示可用 ``--benchmark-map`` 指定中证官网可取的债券指数替代（不自动替换）。

``--factors cn_index_proxy`` 在“Alpha 质量”一节做多因子分解（attribution.factor）：MKT = H00300 − 无风险收益，
SMB = H00852 − H00300，HML = H00919 − H00918，均取中证指数官网全收益指数；这是指数代理因子，
与 Fama–French 分组构造不同。``cn_index_proxy4`` 另加 UMD = 沪深300动量 H30260 − 沪深300 000300（均为价格指数，
不做全收益替换）。``ff3_us`` / ``carhart_us`` 取 French 因子库的美国市场因子（etl.sources.french，美元计价，
无风险收益用因子库 RF），适合投资美股的 QDII 等基金。多因子回归沿用 ``--hac-lags`` 与 ``--use-t``。

``--config <文件.toml>``（report 与 compare）先读取评价口径配置（fundeval.config，示例 examples/config.toml），
再用命令行显式给出的参数覆盖；报告口径写明“配置来源：<文件>（命令行覆盖：…）”。配置中没有对应命令行参数的
置信水平、稳健性开关与日度 K 直接传给 evaluate。

命令行中的 Python 警告改为每条一行“警告：<消息>”输出到标准错误，不显示源码路径与代码行，同一条消息只显示一次；
``--quiet`` 关闭警告输出。取数与评价阶段的警告同时写入报告附注（已在附注中的不重复写入），``--quiet`` 不影响附注。
Python API（evaluate、compare 等）的 warnings 行为不变。``--version`` 显示版本号。
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import urllib.error
import warnings
from collections.abc import Sequence
from pathlib import Path

from fundeval._console import console_print, console_safe, console_write
from fundeval._version import __version__
from fundeval.attribution.factor import FACTOR_PRESET_NAMES
from fundeval.etl import schema
from fundeval.etl.fx import FX_CONVERT, FX_MODES
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


#: 写入报告附注的警告类别（第三方库的 FutureWarning、DeprecationWarning 等只在命令行显示）
NOTE_WARNING_CATEGORIES = (RuntimeWarning, UserWarning)


class WarningLog:
    """命令行的警告处理：每条警告一行“警告：<消息>”写到标准错误（``quiet`` 时不写），同一条消息只显示一次；
    ``messages`` 按出现顺序保存去重后的 (类别, 消息)，供写入报告附注。"""

    def __init__(self, quiet: bool = False, stream=None):
        self.quiet = quiet
        self.stream = stream
        self.messages: list[tuple[type[Warning], str]] = []
        self._seen: set[str] = set()

    @staticmethod
    def format(message) -> str:
        """去掉换行与多余空白，得到一行消息。"""
        return " ".join(str(message).split())

    def showwarning(self, message, category, filename, lineno, file=None, line=None) -> None:
        text = self.format(message)
        if not text or text in self._seen:
            return
        self._seen.add(text)
        self.messages.append((category, text))
        if not self.quiet:
            console_print(f"警告：{text}", file=self.stream or sys.stderr)

    @contextlib.contextmanager
    def capture(self):
        """块内的警告交给 showwarning；RuntimeWarning 与 UserWarning 每次都送达（去重由本类按消息完成），
        其他类别沿用 Python 的默认过滤规则。退出时恢复原有的 warnings 设置。"""
        with warnings.catch_warnings():
            for category in NOTE_WARNING_CATEGORIES:
                warnings.simplefilter("always", category)
            warnings.showwarning = self.showwarning
            yield self

    def since(self, start: int) -> list[str]:
        """第 start 条之后、应写入附注的警告消息。"""
        return [text for category, text in self.messages[start:] if issubclass(category, NOTE_WARNING_CATEGORIES)]


def _merge_notes(notes: list[str], messages: list[str]) -> None:
    """把警告消息追加到附注，已在附注中（相同或被包含）的跳过。"""
    for text in messages:
        if not any(text in WarningLog.format(n) for n in notes):
            notes.append(text)


def _check_charts(args) -> None:
    """--charts 须与 --out 一起使用（图片要存到报告旁的目录）；未安装 matplotlib 时提前给出安装提示。"""
    if not getattr(args, "charts", False):
        return
    if args.out is None:
        raise ValueError("--charts 须与 --out 一起使用（图片保存到报告旁的“<报告名>_files/”目录或 Excel 的“图表”sheet）")
    from fundeval.report.charts import require_matplotlib

    require_matplotlib()


def cmd_report(args, log: WarningLog | None = None) -> int:
    from fundeval.etl.sources import akshare as aks

    log = log or WarningLog()
    _check_charts(args)

    start = len(log.messages)
    with aks.collect_notes() as source_notes:
        inputs = build_report_inputs(ReportOptions.from_namespace(args))
    inputs["notes"] = source_notes + inputs["notes"]
    config = getattr(args, "config_obj", None)
    if config is not None:
        inputs.update(config.evaluate_kwargs())
        inputs["labels"]["config_source"] = args.config_label
    report = evaluate(**inputs)
    _merge_notes(report.notes, log.since(start))
    out = args.out
    if out is None:
        console_write(to_markdown(report), sys.stdout)
        return 0
    suffix = Path(out).suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        to_excel(report, out, charts=args.charts)
    elif suffix in {".md", ".markdown", ".txt"}:
        to_markdown(report, out, charts=args.charts)
    else:
        raise ValueError(f"--out 只支持 .md 或 .xlsx，收到 {out!r}")
    console_print(f"报告已写入 {out}", file=sys.stderr)
    return 0


def cmd_compare(args, log: WarningLog | None = None) -> int:
    log = log or WarningLog()
    _check_charts(args)
    codes = [c.strip() for c in args.funds.replace("，", ",").split(",") if c.strip()]
    opts = ReportOptions.from_namespace(args, fund=None, input=None)
    config = getattr(args, "config_obj", None)
    extra = config.evaluate_kwargs() if config is not None else {}
    start = len(log.messages)
    result = compare(codes, args.start, args.end, freq=args.freq, benchmark=args.benchmark or "contract",
                     sort=args.sort, options=opts, **extra)
    if config is not None:
        result.scope["配置来源"] = args.config_label
    _merge_notes(result.notes, log.since(start))
    out = args.out
    if out is None:
        console_write(result.to_markdown(), sys.stdout)
        return 0
    suffix = Path(out).suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        result.to_excel(out, charts=args.charts)
    elif suffix in {".md", ".markdown", ".txt"}:
        result.to_markdown(out, charts=args.charts)
    else:
        raise ValueError(f"--out 只支持 .md 或 .xlsx，收到 {out!r}")
    console_print(f"对比表已写入 {out}（成功 {len(result.reports)} 只，失败 {len(result.failures)} 只）", file=sys.stderr)
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
    p.add_argument(
        "--fx", choices=FX_MODES, default=FX_CONVERT,
        help="非人民币基准成分：convert（默认，按国家外汇管理局人民币汇率中间价换算为人民币收益，汇率取数失败时报错）"
        "或 none（不换算，口径注明“未做汇率换算”）",
    )


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--config",
        help="评价口径配置文件（TOML，见 examples/config.toml）：先读配置，再用命令行显式给出的参数覆盖；"
        "未知键报错",
    )
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
        help="风格分析：auto（按基金类型选择：偏股、股票 → cn_equity；偏债、债券、FOF、混合 → cn_balanced；"
        "无法判断时 cn_balanced）、预设名（" + "、".join(STYLE_PRESETS) + "）或逗号分隔的指数代码，cash 表示无风险收益；"
        "指数按 --index-return-type 默认换成全收益代码",
    )
    p.add_argument(
        "--factors", choices=FACTOR_PRESET_NAMES,
        help="多因子分解的因子预设：cn_index_proxy（MKT = H00300 − rf，SMB = H00852 − H00300，HML = H00919 − H00918，"
        "指数代理因子，与学术因子不可直接比较）；cn_index_proxy4（再加 UMD = 沪深300动量 H30260 − 沪深300 000300，"
        "两者均为价格指数）；ff3_us、carhart_us（French 因子库的美国市场因子 MKT、SMB、HML 及 UMD，美元计价，"
        "无风险收益用因子库 RF，适合投资美股的 QDII，只支持 --freq M）；回归沿用 --hac-lags 与 --use-t",
    )
    p.add_argument("--style-window", type=int, help="滚动风格分析的窗口期数（不小于风格资产数 + 2）")
    p.add_argument("--mar", type=float, help="Sortino 的最低可接受收益（每期，小数）；缺省取无风险收益")
    p.add_argument("--hac-lags", type=int, help="回归使用 Newey–West 标准误的滞后阶数")
    p.add_argument("--use-t", action="store_true", help="HAC 回归的 p 值与置信区间也用 t 分布（小样本建议）")
    p.add_argument(
        "--tolerance", type=float, default=CROSS_CHECK_TOLERANCE,
        help=f"净值与日增长率交叉核对容差（默认 {CROSS_CHECK_TOLERANCE}，即 {CROSS_CHECK_TOLERANCE * 1e4:.0f} 个基点）",
    )
    p.add_argument(
        "--charts", action="store_true",
        help="生成 PNG 图表（需 pip install \"fundeval[plot]\"）：Markdown 存到“<报告名>_files/”并用相对路径嵌入，"
        "Excel 另加“图表”sheet；须与 --out 一起使用",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="不在标准错误输出警告（警告仍写入报告附注）；错误信息照常输出",
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


class _Parser(argparse.ArgumentParser):
    """帮助、版本与参数错误的输出也按 console_safe 处理编码（帮助文字含“−”等 GBK 没有的字符）。"""

    def _print_message(self, message, file=None):
        if message:
            file = file if file is not None else sys.stderr
            console_write(message, file)


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="fundeval", description="基金与投资组合绩效评估")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
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


def _explicit_dests(argv: Sequence[str]) -> set[str]:
    """命令行中显式给出的参数（argparse 的 dest）：把全部默认值换成 SUPPRESS 后重新解析，留下的就是显式给出的。"""
    parser = build_parser()

    def suppress(p: argparse.ArgumentParser) -> None:
        for action in p._actions:
            if isinstance(action, argparse._SubParsersAction):
                for sub in action.choices.values():
                    suppress(sub)
            elif action.dest not in ("help", "version"):
                action.default = argparse.SUPPRESS
        p._defaults.clear()

    suppress(parser)
    return set(vars(parser.parse_args(list(argv))))


def _option_names(command: str) -> dict[str, str]:
    """子命令的 {dest: 选项写法}，如 {"freq": "--freq", "no_cache": "--no-cache"}。"""
    parser = build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction)).choices[command]
    return {a.dest: max(a.option_strings, key=len) for a in sub._actions if a.option_strings}


def apply_config(args, argv: Sequence[str]) -> None:
    """``--config`` 给出时读取配置文件，命令行未显式给出的参数取配置中的值（显式给出的保留）。

    在 args 上记录 ``config_obj``（EvaluationConfig，或 None）与 ``config_label``
    （“<文件>（命令行覆盖：--freq、--rf）”，写入报告口径“配置来源”）。
    """
    args.config_obj = None
    path = getattr(args, "config", None)
    if not path:
        return
    from fundeval.config import load_config

    config = load_config(path)
    explicit = _explicit_dests(argv)
    names = _option_names(args.command)
    overridden = []
    for dest, value in config.cli_values().items():
        if not hasattr(args, dest):  # 子命令没有的参数（如 compare 的监控目标）
            continue
        if dest in explicit:
            overridden.append(names.get(dest, dest))
            continue
        setattr(args, dest, value)
    args.config_obj = config
    args.config_label = f"{path}（命令行覆盖：{'、'.join(overridden) if overridden else '无'}）"


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    log = WarningLog(quiet=getattr(args, "quiet", False))
    try:
        with log.capture():
            apply_config(args, argv)
            return args.func(args, log)
    except Exception as exc:
        if _is_network_error(exc):
            console_print(f"错误：网络请求失败（{_one_line(exc)}）。{NETWORK_HINT}", file=sys.stderr)
            return 2
        if isinstance(exc, (ValueError, KeyError, ImportError, RuntimeError, FileNotFoundError)):
            console_print(f"错误：{exc}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
