"""评价口径配置：正文第一部分第 1 节“先确定可比较的评价口径”。

正文要求评价前固定样本起止日、计价币种、收益频率、业绩基准、无风险收益和费用口径。EvaluationConfig 把这些口径
（以及 MAR、置信水平、回归、风格、因子、监控目标、稳健性检验、缓存与超时）集中在一个不可变对象里，
默认值与命令行 ``fundeval report`` / ``fundeval compare`` 的默认值一致。

用法::

    from fundeval.config import EvaluationConfig, load_config

    cfg = load_config("examples/config.toml")          # TOML 文件，未知键报错
    report = evaluate(returns, benchmark, config=cfg)   # 关键字参数与 config 同时给出时，关键字参数优先
    table = compare(["110011", "110020"], config=cfg)

    fundeval report --fund 110020 --config examples/config.toml --freq Q   # 命令行显式给出的参数覆盖配置文件

TOML 按以下各节组织（每节的键见 CONFIG_SECTIONS，示例见 examples/config.toml）：

- ``[sample]``：start、end、freq、periods_per_year（K，缺省按 freq 推断）
- ``[benchmark]``：spec（contract、"000300:0.8,H11001:0.2"、单个代码或合同文字，也可写成 {代码 = 权重} 的表）、
  map（{名称 = 代码}，即 --benchmark-map）、index_return_type、index_source、deposit_rate、time_deposit_rate
- ``[risk_free]``：sources（"auto"、"shibor3m" 或顺序列表 ["cgb2y", "shibor3m"]）或 rate（常数年化利率，二选一）
- ``[currency]``：fx（convert / none）
- ``[fees]``：basis（费用口径说明）
- ``[risk]``：mar（Sortino 的最低可接受收益，每期）、confidence（VaR / ES 置信水平）
- ``[regression]``：hac_lags、use_t、robustness（剔除异常期的稳健性检验开关）
- ``[style]``：preset（auto、预设名或代码列表）、window
- ``[factors]``：preset（cn_index_proxy、cn_index_proxy4、ff3_us、carhart_us）
- ``[monitor]``：target_active、target_te、window
- ``[data]``：cache_dir、use_cache、refresh、timeout、total_timeout、tolerance（净值与日增长率核对容差）

读取 TOML 在 Python ≥ 3.11 用标准库 tomllib，3.10 用 tomli（pyproject 中的条件依赖）。
"""

from __future__ import annotations

import datetime as _dt
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import pandas as pd

from fundeval.attribution.factor import FACTOR_PRESET_NAMES
from fundeval.etl import schema
from fundeval.etl.fx import FX_CONVERT, FX_MODES
from fundeval.etl.quality import CROSS_CHECK_TOLERANCE
from fundeval.etl.sources.akshare import DEFAULT_TIMEOUT, DEFAULT_TOTAL_TIMEOUT
from fundeval.report.summary import DEFAULT_FEE_BASIS

#: 支持的频率代码
FREQS = ("D", "W", "M", "Q", "A")
INDEX_RETURN_TYPES = ("total", "price")
INDEX_SOURCES = ("auto", "em", "csindex")

#: TOML 的节与键 → EvaluationConfig 字段名
CONFIG_SECTIONS: dict[str, dict[str, str]] = {
    "sample": {"start": "start", "end": "end", "freq": "freq", "periods_per_year": "periods_per_year"},
    "benchmark": {
        "spec": "benchmark", "map": "benchmark_map", "index_return_type": "index_return_type",
        "index_source": "index_source", "deposit_rate": "deposit_rate", "time_deposit_rate": "time_deposit_rate",
    },
    "risk_free": {"sources": "rf", "rate": "rf"},
    "currency": {"fx": "fx"},
    "fees": {"basis": "fees"},
    "risk": {"mar": "mar", "confidence": "confidence"},
    "regression": {"hac_lags": "hac_lags", "use_t": "use_t", "robustness": "robustness"},
    "style": {"preset": "style", "window": "style_window"},
    "factors": {"preset": "factors"},
    "monitor": {"target_active": "target_active", "target_te": "target_te", "window": "window"},
    "data": {
        "cache_dir": "cache_dir", "use_cache": "use_cache", "refresh": "refresh", "timeout": "timeout",
        "total_timeout": "total_timeout", "tolerance": "tolerance",
    },
}

#: EvaluationConfig 字段 → 命令行参数（argparse 的 dest）；use_cache 与 --no-cache 相反
CLI_DESTS: dict[str, str] = {
    "start": "start", "end": "end", "freq": "freq",
    "benchmark": "benchmark", "benchmark_map": "benchmark_map", "index_return_type": "index_return_type",
    "index_source": "index_source", "deposit_rate": "deposit_rate", "time_deposit_rate": "time_deposit_rate",
    "rf": "rf", "fx": "fx", "fees": "fees", "mar": "mar", "hac_lags": "hac_lags", "use_t": "use_t",
    "style": "style", "style_window": "style_window", "factors": "factors",
    "target_active": "target_active", "target_te": "target_te", "window": "window",
    "cache_dir": "cache_dir", "use_cache": "no_cache", "refresh": "refresh", "timeout": "timeout",
    "total_timeout": "total_timeout", "tolerance": "tolerance",
}

#: 只在评价阶段使用、没有对应命令行参数的字段（传给 evaluate）
EVALUATE_ONLY = ("periods_per_year", "confidence", "robustness")


def _date_text(value, name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, (_dt.date, _dt.datetime, pd.Timestamp)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    try:
        return pd.Timestamp(str(value)).strftime("%Y-%m-%d")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} 无法解析为日期：{value!r}") from exc


def _number(value, name: str, *, allow_none: bool = True) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 须为数值，收到 {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} 须为有限数值，收到 {value!r}")
    return value


def _integer(value, name: str, minimum: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 须为整数，收到 {value!r}")
    if value < minimum:
        raise ValueError(f"{name} 须不小于 {minimum}，收到 {value}")
    return int(value)


def _flag(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} 须为 true 或 false，收到 {value!r}")
    return value


def _choice(value, name: str, choices) -> str:
    if value not in choices:
        raise ValueError(f"{name} 须为 {'、'.join(choices)} 之一，收到 {value!r}")
    return value


def _rf_text(value) -> str:
    """无风险收益：来源名、逗号分隔或列表形式的来源顺序，或常数年化利率 → 命令行 --rf 的写法。"""
    if isinstance(value, (list, tuple)):
        items = [str(v).strip() for v in value]
        if not items or any(not v for v in items):
            raise ValueError(f"rf 的来源列表不能为空或含空项，收到 {value!r}")
        return ",".join(items)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        rate = _number(value, "rf")
        if abs(rate) >= 1:
            raise ValueError(f"rf 为年化小数（1.8% 写作 0.018），收到 {value}")
        return repr(rate)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"rf 须为来源名、来源列表或常数年化利率，收到 {value!r}")


def _benchmark_text(value) -> str | None:
    """基准：字符串原样；{代码: 权重} 的表 → "代码:权重,..."。"""
    if value is None:
        return None
    if isinstance(value, Mapping):
        if not value:
            raise ValueError("benchmark 的成分表不能为空")
        parts = []
        for code, weight in value.items():
            parts.append(f"{str(code).strip()}:{_number(weight, f'benchmark 成分 {code} 的权重', allow_none=False):g}")
        return ",".join(parts)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"benchmark 须为文字或 {{代码 = 权重}} 的表，收到 {value!r}")


@dataclass(frozen=True)
class EvaluationConfig:
    """评价口径（正文第一部分第 1 节）。默认值与命令行默认值一致；构造时校验并规范化取值。

    区间与频率：``start``、``end``（"YYYY-MM-DD"）、``freq``（D/W/M/Q/A）、``periods_per_year``（K，缺省按 freq：
    252 / 52 / 12 / 4 / 1；W、M、Q、A 须与 freq 一致，日度可按市场约定另给，如 244）

    基准：``benchmark``（contract、"代码:权重,..."、单个代码或合同文字；None 时 --fund 默认 contract）、
    ``benchmark_map``、``index_return_type``、``index_source``、``deposit_rate``、``time_deposit_rate``

    无风险收益 ``rf``："auto"、来源名、顺序列表（"cgb2y,shibor3m" 或 ["cgb2y", "shibor3m"]）或常数年化利率；
    币种与汇率 ``fx``；费用口径 ``fees``；``mar``；置信水平 ``confidence``；回归 ``hac_lags``、``use_t``；
    稳健性开关 ``robustness``；风格 ``style``、``style_window``；因子预设 ``factors``；
    监控目标 ``target_active``、``target_te``、``window``；缓存与超时 ``cache_dir``、``use_cache``、``refresh``、
    ``timeout``、``total_timeout``；净值核对容差 ``tolerance``。

    ``source`` 记录配置来源（load_config 的文件路径），不参与比较。
    """

    start: Any = None
    end: Any = None
    freq: str = "M"
    periods_per_year: int | None = None
    benchmark: Any = None
    benchmark_map: Mapping[str, str] | None = None
    index_return_type: str = "total"
    index_source: str = "auto"
    deposit_rate: float | None = None
    time_deposit_rate: float | None = None
    rf: Any = "auto"
    fx: str = FX_CONVERT
    fees: str = DEFAULT_FEE_BASIS
    mar: float | None = None
    confidence: float = 0.95
    hac_lags: int | None = None
    use_t: bool = False
    robustness: bool = True
    style: str | None = None
    style_window: int | None = None
    factors: str | None = None
    target_active: float | None = None
    target_te: float | None = None
    window: int | None = None
    cache_dir: str | None = None
    use_cache: bool = True
    refresh: bool = False
    timeout: float = DEFAULT_TIMEOUT
    total_timeout: float = DEFAULT_TOTAL_TIMEOUT
    tolerance: float = CROSS_CHECK_TOLERANCE
    source: str | None = field(default=None, compare=False)

    def __post_init__(self):
        set_ = lambda name, value: object.__setattr__(self, name, value)  # noqa: E731  （frozen 下规范化取值）
        set_("start", _date_text(self.start, "start"))
        set_("end", _date_text(self.end, "end"))
        if self.start and self.end and self.start > self.end:
            raise ValueError(f"start（{self.start}）不能晚于 end（{self.end}）")
        freq = str(self.freq).strip().upper()[:1]
        set_("freq", "A" if freq == "Y" else freq)
        _choice(self.freq, "freq", FREQS)
        k = _integer(self.periods_per_year, "periods_per_year（K）", 1)
        standard = schema.PERIODS_PER_YEAR[self.freq]
        if k is not None and self.freq != "D" and k != standard:
            raise ValueError(f"freq = {self.freq} 时 K 须为 {standard}，收到 {k}（只有日度可按市场约定另给 K）")
        set_("periods_per_year", k)
        set_("benchmark", _benchmark_text(self.benchmark))
        if self.benchmark_map is not None:
            if not isinstance(self.benchmark_map, Mapping):
                raise ValueError(f"benchmark_map 须为 {{名称 = 代码}} 的表，收到 {self.benchmark_map!r}")
            set_("benchmark_map", {str(k_).strip(): str(v).strip() for k_, v in self.benchmark_map.items()})
        _choice(self.index_return_type, "index_return_type", INDEX_RETURN_TYPES)
        _choice(self.index_source, "index_source", INDEX_SOURCES)
        for name in ("deposit_rate", "time_deposit_rate"):
            value = _number(getattr(self, name), name)
            if value is not None and not 0 <= value < 0.2:
                raise ValueError(f"{name} 为年化小数（0.35% 写作 0.0035），收到 {value}")
            set_(name, value)
        set_("rf", _rf_text(self.rf))
        _choice(self.fx, "fx", FX_MODES)
        if not isinstance(self.fees, str) or not self.fees.strip():
            raise ValueError(f"fees（费用口径）须为非空文字，收到 {self.fees!r}")
        set_("mar", _number(self.mar, "mar"))
        confidence = _number(self.confidence, "confidence", allow_none=False)
        if not 0 < confidence < 1:
            raise ValueError(f"confidence（置信水平）须在 (0, 1) 内，收到 {confidence}")
        set_("confidence", confidence)
        set_("hac_lags", _integer(self.hac_lags, "hac_lags", 0))
        for name in ("use_t", "robustness", "use_cache", "refresh"):
            _flag(getattr(self, name), name)
        if self.style is not None and (not isinstance(self.style, str) or not self.style.strip()):
            raise ValueError(f"style 须为 auto、预设名或逗号分隔的代码，收到 {self.style!r}")
        set_("style_window", _integer(self.style_window, "style_window", 3))
        if self.style_window is not None and self.style is None:
            raise ValueError("style_window 须与 style 一起给出")
        if self.factors is not None:
            _choice(self.factors, "factors", FACTOR_PRESET_NAMES)
        set_("target_active", _number(self.target_active, "target_active"))
        target_te = _number(self.target_te, "target_te")
        if target_te is not None and target_te <= 0:
            raise ValueError(f"target_te（年化目标 TE）须为正数，收到 {target_te}")
        set_("target_te", target_te)
        set_("window", _integer(self.window, "window", 2))
        if self.cache_dir is not None:
            set_("cache_dir", str(self.cache_dir))
        for name in ("timeout", "total_timeout"):
            value = _number(getattr(self, name), name, allow_none=False)
            if value <= 0:
                raise ValueError(f"{name} 须为正数（秒），收到 {value}")
            set_(name, value)
        tolerance = _number(self.tolerance, "tolerance", allow_none=False)
        if tolerance < 0:
            raise ValueError(f"tolerance 须为非负数，收到 {tolerance}")
        set_("tolerance", tolerance)

    @property
    def k(self) -> int:
        """一年期数 K：显式给出的 periods_per_year，否则按 freq（月度 12）。"""
        return self.periods_per_year if self.periods_per_year is not None else schema.PERIODS_PER_YEAR[self.freq]

    @property
    def monitor_targets(self) -> dict[str, float] | None:
        """evaluate 的 monitor_targets；三项都给出时才返回，只给部分时报错。"""
        values = (self.target_active, self.target_te, self.window)
        if all(v is None for v in values):
            return None
        if any(v is None for v in values):
            raise ValueError("监控目标须同时给出 target_active、target_te 与 window")
        return {"target_active_return": self.target_active, "target_te": self.target_te, "window": self.window}

    def cli_values(self) -> dict[str, Any]:
        """{argparse dest: 值}，供命令行在未显式给出参数时取用（use_cache 换成 no_cache）。"""
        out = {}
        for name, dest in CLI_DESTS.items():
            value = getattr(self, name)
            out[dest] = (not value) if name == "use_cache" else value
        return out

    def report_options(self, **overrides):
        """转换为 report.inputs.ReportOptions（取数选项）；``overrides`` 为要覆盖的 ReportOptions 字段。"""
        from fundeval.report.inputs import ReportOptions

        names = {f.name for f in fields(ReportOptions)}
        values = {dest: value for dest, value in self.cli_values().items() if dest in names}
        return ReportOptions(**{**values, **overrides})

    def evaluate_kwargs(self) -> dict[str, Any]:
        """没有对应取数选项、只在 evaluate 中使用的口径：confidence、robustness，以及显式给出的 K。"""
        out: dict[str, Any] = {"confidence": self.confidence, "robustness": self.robustness}
        if self.periods_per_year is not None:
            out["periods_per_year"] = self.periods_per_year
        return out

    def describe(self) -> str:
        """配置来源的文字：文件路径，或“Python 对象”。"""
        return self.source or "EvaluationConfig（Python 对象）"


def config_from_dict(data: Mapping[str, Any], source: str | None = None) -> EvaluationConfig:
    """把按节组织的字典（TOML 的解析结果）转换为 EvaluationConfig。

    未知的节或键报错并列出可选项，不静默忽略；``[risk_free]`` 的 sources 与 rate 只能给一个。
    """
    if not isinstance(data, Mapping):
        raise ValueError("配置须为按节组织的表")
    values: dict[str, Any] = {}
    origin: dict[str, str] = {}
    for section, body in data.items():
        if section not in CONFIG_SECTIONS:
            raise ValueError(f"未知的配置节 [{section}]，可选：{'、'.join(f'[{s}]' for s in CONFIG_SECTIONS)}")
        if not isinstance(body, Mapping):
            raise ValueError(f"[{section}] 须为表（节），收到 {body!r}；可选键：{'、'.join(CONFIG_SECTIONS[section])}")
        keys = CONFIG_SECTIONS[section]
        for key, value in body.items():
            if key not in keys:
                raise ValueError(f"[{section}] 中的未知键 {key!r}，可选：{'、'.join(keys)}")
            name = keys[key]
            if name in origin:
                raise ValueError(f"[{section}] 的 {key} 与 {origin[name]} 不能同时给出")
            origin[name] = f"[{section}] {key}"
            values[name] = value
    try:
        return EvaluationConfig(**values, source=source)
    except ValueError as exc:
        where = f"（{source}）" if source else ""
        raise ValueError(f"配置有误{where}：{exc}") from exc


def _toml():
    try:
        import tomllib  # Python ≥ 3.11
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        try:
            import tomli as tomllib
        except ModuleNotFoundError as exc:
            raise ImportError("Python 3.10 读取 TOML 需要 tomli：pip install tomli") from exc
    return tomllib


def load_config(path) -> EvaluationConfig:
    """读取 TOML 配置文件，返回 EvaluationConfig（``source`` 为文件路径）。

    未知的节或键报错并列出可选项；数值校验见 EvaluationConfig（K 为正、置信水平在 (0, 1) 内等）。
    """
    tomllib = _toml()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    with path.open("rb") as fh:
        try:
            data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"配置文件 {path} 不是有效的 TOML：{exc}") from exc
    return config_from_dict(data, source=str(path))


__all__ = ["CONFIG_SECTIONS", "EvaluationConfig", "config_from_dict", "load_config"]
