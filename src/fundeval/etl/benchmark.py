"""复合基准：正文第一部分“基准选择”。

基金合同中的业绩比较基准常写作“沪深300指数收益率×80%+中债综合指数收益率×20%”。
``parse_benchmark`` 只解析能确定无歧义的写法，``composite_benchmark`` 按权重合成收益。
合同基准通常隐含每日（每期）再平衡；成分指数是否含分红（价格指数或全收益指数）
直接影响基准收益，须在报告口径中写明。基金净值含分红，与价格指数比较会把成分股
股息计入超额收益，因此 ``INDEX_RECORDS`` 记录每个指数的全收益版本，``total_return_code``
给出替换代码，``benchmark_return_type`` 给出报告口径中的“基准收益类型”。
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fundeval.etl import schema

#: 权重加总的容差
WEIGHT_TOL = 1e-8

#: 收益类型：price 价格指数（不含成分股分红）；total 全收益 / 财富指数（分红或利息再投资）；
#: unknown 无法确认。
PRICE, TOTAL, UNKNOWN = "price", "total", "unknown"

#: 基准收益类型的中文标签，供报告口径使用
RETURN_TYPE_LABELS = {TOTAL: "全收益", PRICE: "价格指数", "mixed": "含价格指数成分", UNKNOWN: "未知"}

#: 现金类成分（存款利率）的收益类型
CASH_RETURN = "cash"

#: 价格指数不含分红时的限定语，写入报告口径、附注与结论
PRICE_INDEX_CAVEAT = "价格指数不含成分股分红，超额收益与 Alpha 会高估约为股息率的幅度"


@dataclass(frozen=True)
class IndexRecord:
    """一条指数记录。

    - ``name``：常用名称；``aliases`` 为其他写法
    - ``price_code``：价格指数代码（中债财富指数等没有价格指数版本的为 None）
    - ``total_code``：对应的全收益指数代码，没有或未经核实的为 None
    - ``source``：主代码（price_code，缺省时为 total_code）的默认数据源：
      ``em`` 东方财富、``csindex`` 中证指数官网、``cbond`` 中债；全收益代码一律取自中证官网
    - ``return_type``：主代码的收益类型（price / total / unknown）
    """

    name: str
    price_code: str | None
    total_code: str | None
    source: str
    return_type: str
    aliases: tuple[str, ...] = ()

    @property
    def code(self) -> str:
        """主代码：price_code，缺省时为 total_code。"""
        return self.price_code or self.total_code


#: 指数表。全收益代码（H00300 等）已在中证官网 stock_zh_index_hist_csindex 实测可取（2026-09-27）；
#: 其中沪深300成长 H00918、沪深300价值 H00919、中证红利 H00922 另核对了 2024 年全收益与价格指数的
#: 差异与分红相符（成长 6.91% 对 4.24%，价值 30.84% 对 24.85%，红利 18.76% 对 12.31%）。
#: 中证2000（932000）的全收益代码未知，不填；H20932 不是中证2000全收益（2024 年 +52.37%，
#: 价格指数为 −2.14%），不得使用。中债“财富”指数为全收益口径。中证债券指数（H11001、H11009）在本表中未核实其价格 / 财富口径，
#: 标为 unknown；上证综指、深证成指、创业板指的全收益版本未经核实，不填，不做猜测。
INDEX_RECORDS: tuple[IndexRecord, ...] = (
    IndexRecord("沪深300", "000300", "H00300", "em", PRICE),
    IndexRecord("中证500", "000905", "H00905", "em", PRICE),
    IndexRecord("中证800", "000906", "H00906", "em", PRICE),
    IndexRecord("中证1000", "000852", "H00852", "em", PRICE),
    IndexRecord("上证50", "000016", "H00016", "em", PRICE),
    IndexRecord("沪深300成长", "000918", "H00918", "em", PRICE),
    IndexRecord("沪深300价值", "000919", "H00919", "em", PRICE),
    IndexRecord("中证红利", "000922", "H00922", "em", PRICE),
    IndexRecord("中证2000", "932000", None, "csindex", PRICE),
    IndexRecord("上证综合", "000001", None, "em", PRICE, ("上证综指",)),
    IndexRecord("深证成份", "399001", None, "em", PRICE, ("深证成指",)),
    IndexRecord("创业板", "399006", None, "em", PRICE, ("创业板指",)),
    IndexRecord("中证全债", "H11001", None, "csindex", UNKNOWN),
    IndexRecord("中证综合债", "H11009", None, "csindex", UNKNOWN),
    IndexRecord("中债综合", None, "cbond:composite", "cbond", TOTAL),
    IndexRecord("中债新综合", None, "cbond:new_composite", "cbond", TOTAL),
    # 中债综合全价指数：含应计利息、不含利息再投资，按价格指数处理；不自动换成财富指数
    IndexRecord("中债综合全价", "cbond:composite_full", None, "cbond", PRICE),
)

#: 名称到主代码的对照，供 lookup_index_code 使用。
#: 代码前缀 ``cbond:`` 表示中债指数（中国债券信息网），取财富（全收益）指数。
#: 名称只按下表精确匹配，表外名称报错，由调用方手动指定代码。
INDEX_CODES: dict[str, str] = {
    name: rec.code for rec in INDEX_RECORDS for name in (rec.name, *rec.aliases)
}


def index_record(code: str) -> IndexRecord | None:
    """按价格指数或全收益指数代码查指数记录；表外代码返回 None。"""
    for rec in INDEX_RECORDS:
        if code in (rec.price_code, rec.total_code):
            return rec
    return None


def index_return_type(code: str) -> str:
    """指数代码的收益类型：price、total 或 unknown（表外代码为 unknown）。"""
    rec = index_record(code)
    if rec is None:
        return UNKNOWN
    if code == rec.total_code:
        return TOTAL
    return rec.return_type


def total_return_code(code: str) -> str | None:
    """有对应全收益指数时返回其代码（本身已是全收益指数时返回自身），否则返回 None。"""
    rec = index_record(code)
    if rec is None:
        return None
    if code == rec.total_code or (code == rec.price_code and rec.return_type == TOTAL):
        return code
    return rec.total_code


def benchmark_return_type(codes: Sequence[str]) -> str:
    """复合基准的收益类型标签：全收益、价格指数、含价格指数成分或未知。

    全部成分为全收益时为“全收益”；全部为价格指数时为“价格指数”；部分成分为价格指数时为
    “含价格指数成分”；其余情形（有未知成分、没有价格指数成分）为“未知”。
    """
    types = [index_return_type(c) for c in codes]
    if not types:
        return RETURN_TYPE_LABELS[UNKNOWN]
    if all(t == TOTAL for t in types):
        return RETURN_TYPE_LABELS[TOTAL]
    if all(t == PRICE for t in types):
        return RETURN_TYPE_LABELS[PRICE]
    if PRICE in types:
        return RETURN_TYPE_LABELS["mixed"]
    return RETURN_TYPE_LABELS[UNKNOWN]


def needs_price_caveat(label: str | None) -> bool:
    """基准收益类型为价格指数、含价格指数成分或未知时，报告须写明价格指数的限定语。"""
    return label in (RETURN_TYPE_LABELS[PRICE], RETURN_TYPE_LABELS["mixed"], RETURN_TYPE_LABELS[UNKNOWN])


#: 风格分析中代表无风险收益（现金）的列名与代码
CASH = "cash"

#: 风格指数预设（正文第五部分第 1 节）：{列名: 指数代码}，指数一律取全收益代码，``cash`` 为无风险收益。
#: 风格基准宜覆盖完整且尽量低冗余；中证红利与沪深300价值相关性高，不放进默认预设，只在指数表中提供。
STYLE_PRESETS: dict[str, dict[str, str]] = {
    "cn_equity": {
        "沪深300成长": "H00918",
        "沪深300价值": "H00919",
        "中证500": "H00905",
        "中证1000": "H00852",
        "现金": CASH,
    },
}
STYLE_PRESETS["cn_balanced"] = {**STYLE_PRESETS["cn_equity"], "中证全债": "H11001"}


def style_preset(name: str) -> dict[str, str]:
    """风格指数预设，返回 {列名: 指数代码} 的副本；``cash`` 表示以无风险收益作现金资产。

    - ``cn_equity``：沪深300成长 H00918、沪深300价值 H00919、中证500 H00905、中证1000 H00852、现金
    - ``cn_balanced``：cn_equity 加中证全债 H11001（收益类型未经核实，报告中注明）
    """
    if name not in STYLE_PRESETS:
        raise KeyError(f"未知的风格预设 {name!r}，可选：{sorted(STYLE_PRESETS)}")
    return dict(STYLE_PRESETS[name])


_TERM_PATTERNS = (
    # 名称 × 权重%：沪深300指数收益率*80%
    re.compile(r"^(?P<name>.+?)\s*[×xX*]\s*(?P<weight>\d+(?:\.\d+)?)\s*[%％]$"),
    # 权重% × 名称：80%×沪深300指数收益率
    re.compile(r"^(?P<weight>\d+(?:\.\d+)?)\s*[%％]\s*[×xX*]\s*(?P<name>.+)$"),
)

_UNPARSEABLE_HINT = "请手动指定成分与权重，例如 composite_benchmark({'000300': r1, 'H11001': r2}, {'000300': 0.8, 'H11001': 0.2})"


def _clean_name(name: str) -> str:
    name = name.strip()
    for suffix in ("收益率", "收益", "涨跌幅"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.strip()


def parse_benchmark(text: str) -> list[tuple[str, float]]:
    """把合同中的业绩比较基准文字解析为 [(指数名, 权重), ...]，权重为小数。

    只接受“名称×权重%”或“权重%×名称”以加号连接的写法，乘号可写作 ×、*、x；
    名称去掉末尾的“收益率”。例如::

        parse_benchmark("沪深300指数收益率*80%+中债综合指数收益率*20%")
        # [("沪深300指数", 0.8), ("中债综合指数", 0.2)]

    名称可带括号与连字符（如“银行活期存款利率(税后)”“中债-综合全价(总值)指数收益率”）。
    含减法或嵌套运算、缺少权重、
    权重加总不为 100% 等情形一律报错，提示手动指定，不做猜测。
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("业绩比较基准文字为空")
    compact = re.sub(r"\s+", "", text).replace("＋", "+")
    terms = compact.split("+")
    out: list[tuple[str, float]] = []
    for term in terms:
        match = next((m for p in _TERM_PATTERNS if (m := p.match(term))), None)
        if match is None:
            raise ValueError(f"无法识别基准成分 {term!r}（原文：{text!r}）。{_UNPARSEABLE_HINT}")
        name = _clean_name(match["name"])
        # 名称内的连字符（“中债-综合全价(总值)”）保留；运算符、百分号或以减号开头的写法视为无法识别
        if not name or re.search(r"[+−×*%％]", name) or name[0] in "-－":
            raise ValueError(f"无法识别基准成分 {term!r}（原文：{text!r}）。{_UNPARSEABLE_HINT}")
        out.append((name, float(match["weight"]) / 100))
    names = [n for n, _ in out]
    if len(set(names)) != len(names):
        raise ValueError(f"基准成分重复：{names}。{_UNPARSEABLE_HINT}")
    total = sum(w for _, w in out)
    if abs(total - 1) > WEIGHT_TOL:
        raise ValueError(f"基准权重加总为 {total:.4%}，不等于 100%（原文：{text!r}）。{_UNPARSEABLE_HINT}")
    return out


def lookup_index_code(name: str) -> str:
    """按 INDEX_CODES 把指数名称（可带“指数”二字）换算为数据源代码；表外名称报错。"""
    key = _clean_name(name)
    for candidate in (key, key.removesuffix("指数"), key.removesuffix("全收益").removesuffix("指数")):
        if candidate in INDEX_CODES:
            return INDEX_CODES[candidate]
    raise KeyError(f"未收录指数 {name!r} 的代码，请手动指定（已收录：{sorted(INDEX_CODES)}）")


# ---------------------------------------------------------------------------
# 合同业绩比较基准的成分解析
# ---------------------------------------------------------------------------

#: 人民币活期存款、一年期定期存款基准利率（年化，小数）：中国人民银行存款基准利率，2015-10-24 起未调整
DEMAND_DEPOSIT_RATE = 0.0035
TIME_DEPOSIT_RATE = 0.015
DEPOSIT_RATE_SOURCE = "中国人民银行存款基准利率，2015-10-24 起未调整"

#: 人民币币种的写法（index_csindex_all 的“指数币种”）
CNY = "人民币"

#: 非人民币成分的限定语，写入口径与附注
FX_CAVEAT = "未做汇率换算，基准收益含汇率差异"

#: 名称有歧义、不自动映射的成分：{规范化名称: 说明}
AMBIGUOUS_NAMES = {
    "中债总": "“中债总指数”可能指中债-总财富指数、总全价指数或总净价指数，名称有歧义，不自动映射",
}

#: 中债指数的名称写法（规范化后）→ 代码
CBOND_NAMES = {
    "中债综合财富(总值)": "cbond:composite",
    "中债综合财富": "cbond:composite",
    "中债综合全价(总值)": "cbond:composite_full",
    "中债综合全价": "cbond:composite_full",
    "中债新综合财富(总值)": "cbond:new_composite",
    "中债新综合财富": "cbond:new_composite",
}

#: 解析依据的中文说明
RESOLUTION_METHODS = {
    "override": "调用方指定",
    "cash": "存款利率（常数）",
    "records": "指数表 INDEX_RECORDS",
    "cbond": "中债指数名称",
    "catalog": "中证指数目录精确匹配",
}


class BenchmarkResolutionError(ValueError):
    """合同基准中有成分无法解析；``unresolved`` 为 [(成分名称, 原因), ...]。"""

    def __init__(self, message: str, unresolved: list[tuple[str, str]]):
        super().__init__(message)
        self.unresolved = unresolved


@dataclass(frozen=True)
class ResolvedComponent:
    """一个已解析的基准成分。

    - ``name``：合同中的名称（已去掉“收益率”）；``weight``：权重（小数，单独解析成分时为 None）
    - ``code``：数据源代码；现金类为 ``cash:demand``（活期）或 ``cash:time``（一年期定期）
    - ``return_type``：total、price、unknown 或 cash
    - ``source``：数据源（em、csindex、cbond 或 constant）；``method``：解析依据（见 RESOLUTION_METHODS）
    - ``currency``：指数币种；``rate``：现金类的年化利率；``note``：说明
    """

    name: str
    code: str
    return_type: str
    source: str
    currency: str
    method: str
    note: str = ""
    weight: float | None = None
    rate: float | None = None

    @property
    def is_cash(self) -> bool:
        return self.return_type == CASH_RETURN


def normalize_component_name(name: str) -> str:
    """成分名称的规范化：去掉空白与连字符，全角括号换成半角，去掉末尾的“收益率”与“指数”。

    例如“中债-综合全价(总值)指数收益率”→“中债综合全价(总值)”，“中证800成长指数”→“中证800成长”。
    """
    text = re.sub(r"\s+", "", str(name)).replace("（", "(").replace("）", ")")
    text = re.sub(r"[-－—‐]", "", text)
    text = _clean_name(text)
    return text.removesuffix("指数")


_CASH_DEMAND = {"活期存款利率", "活期存款基准利率"}
_CASH_TIME = {"一年期定期存款利率", "1年期定期存款利率", "一年定期存款利率", "一年期定期存款基准利率"}


def _cash_kind(norm: str) -> str | None:
    text = norm.replace("(税后)", "")
    for prefix in ("人民币", "银行", "同期", "商业银行"):
        text = text.removeprefix(prefix)
    if text in _CASH_DEMAND:
        return "demand"
    if text in _CASH_TIME:
        return "time"
    return None


def _code_component(name: str, code: str, method: str, return_type: str, note: str = "") -> ResolvedComponent:
    """按代码生成成分：return_type 为 "total" 时，指数表中有已核实的全收益版本就换成全收益代码。"""
    used = code
    if return_type == TOTAL:
        total = total_return_code(code)
        if total is not None:
            used = total
    rec = index_record(used)
    if used.startswith("cbond:"):
        source = "cbond"
    elif not used.isdigit():
        source = "csindex"
    else:
        source = rec.source if rec is not None and used == rec.price_code else "em"
    text = note
    if used != code:
        text = f"{code} 换成全收益指数 {used}" + (f"；{note}" if note else "")
    return ResolvedComponent(name, used, index_return_type(used), source, CNY, method, text)


def _catalog_frame(catalog) -> pd.DataFrame | None:
    return catalog() if callable(catalog) else catalog


def _catalog_candidates(catalog: pd.DataFrame, norm: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (精确匹配的行, 名称包含该成分的候选行)。"""
    short = catalog["指数简称"].map(normalize_component_name)
    full = catalog["指数全称"].map(normalize_component_name)
    exact = catalog[(short == norm) | (full == norm)]
    partial = catalog[short.str.contains(norm, regex=False) | full.str.contains(norm, regex=False)]
    return exact, partial


def _describe_rows(rows: pd.DataFrame, limit: int = 8) -> str:
    items = [f"{r['指数代码']} {r['指数简称']}（{r['指数全称']}）" for _, r in rows.head(limit).iterrows()]
    more = f" 等 {len(rows)} 条" if len(rows) > limit else ""
    return "、".join(items) + more


def resolve_component(
    name: str,
    catalog=None,
    overrides: Mapping[str, str] | None = None,
    *,
    return_type: str = TOTAL,
    deposit_rate: float = DEMAND_DEPOSIT_RATE,
    time_deposit_rate: float = TIME_DEPOSIT_RATE,
) -> ResolvedComponent:
    """把合同基准中的一个成分名称解析为代码、收益类型、数据源、币种与说明（正文第一部分“基准选择”）。

    按以下顺序，先命中者为准：

    (a) ``overrides``：调用方给出的 {名称: 代码}（名称按 normalize_component_name 比较；代码写
        ``cash:0.02`` 表示常数年化 2% 的现金），原样采用，不换全收益
    (b) 现金类：“活期存款利率(税后)”“银行活期存款利率”→ 常数年化 ``deposit_rate``（默认 0.35%）；
        “一年期定期存款利率(税后)”→ ``time_deposit_rate``（默认 1.50%），收益类型 cash
    (c) INDEX_RECORDS（如沪深300 → 000300；``return_type="total"`` 时换成已核实的全收益 H00300）
    (d) 中债指数：“中债-综合财富(总值)”→ cbond:composite（全收益），“中债-综合全价(总值)”→
        cbond:composite_full（价格指数）；“中债总指数”名称有歧义，报错
    (e) ``catalog``（index_catalog() 的结果，或返回它的无参函数，需要时才调用）：去掉“收益率”“指数”
        后缀后，与“指数简称”“指数全称”精确比较，恰好匹配一条才采用；收益类型记为 unknown
        （除非指数表中有全收益映射），币种取“指数币种”

    币种不是人民币时发出 RuntimeWarning，说明中写明“未做汇率换算，基准收益含汇率差异”。
    无法解析时抛出 BenchmarkResolutionError，写明原因与候选，不做猜测。
    """
    if return_type not in (TOTAL, PRICE):
        raise ValueError(f"return_type 须为 'total' 或 'price'，收到 {return_type!r}")
    display = _clean_name(name)
    norm = normalize_component_name(name)

    def fail(reason: str):
        raise BenchmarkResolutionError(f"基准成分“{display}”无法解析：{reason}", [(display, reason)])

    # (a) 调用方指定
    for key, code in (overrides or {}).items():
        if normalize_component_name(key) != norm:
            continue
        code = str(code).strip()
        if code.startswith("cash:"):
            try:
                rate = float(code.removeprefix("cash:"))
            except ValueError:
                fail(f"指定的现金代码 {code!r} 无法解析，应写作 cash:年化利率，如 cash:0.02")
            return ResolvedComponent(display, code, CASH_RETURN, "constant", CNY, "override",
                                     f"调用方指定常数年化 {rate:.2%}", rate=rate)
        comp = _code_component(display, code, "override", PRICE, "调用方指定")
        if index_record(code) is None and not code.startswith("cbond:") and catalog is not None:
            comp = _with_catalog_currency(comp, _catalog_frame(catalog))
        return _check_currency(comp)
    # (b) 现金类
    kind = _cash_kind(norm)
    if kind is not None:
        rate, label = (deposit_rate, "活期存款") if kind == "demand" else (time_deposit_rate, "一年期定期存款")
        note = f"常数年化 {rate:.2%}（{label}；" + (
            f"{DEPOSIT_RATE_SOURCE}）" if rate == (DEMAND_DEPOSIT_RATE if kind == "demand" else TIME_DEPOSIT_RATE)
            else "调用方指定）"
        )
        return ResolvedComponent(display, f"cash:{kind}", CASH_RETURN, "constant", CNY, "cash", note, rate=rate)
    # (c) 指数表
    for candidate in (norm, norm.removesuffix("全收益")):
        if candidate in INDEX_CODES:
            return _code_component(display, INDEX_CODES[candidate], "records", return_type)
    # (d) 中债指数
    if norm in AMBIGUOUS_NAMES:
        fail(AMBIGUOUS_NAMES[norm])
    if norm in CBOND_NAMES:
        code = CBOND_NAMES[norm]
        note = ""
        if code == "cbond:composite_full":
            note = "全价指数含应计利息、不含利息再投资，按价格指数处理；如需全收益口径可指定 cbond:composite（财富指数）"
        return _code_component(display, code, "cbond", return_type, note)
    if norm.startswith("中债"):
        fail("未收录的中债指数写法（已收录：" + "、".join(sorted(CBOND_NAMES)) + "）")
    # (e) 中证指数目录
    frame = _catalog_frame(catalog)
    if frame is None:
        fail("不在指数表中，且未提供中证指数目录（index_catalog）")
    exact, partial = _catalog_candidates(frame, norm)
    codes = exact["指数代码"].unique()
    if len(codes) == 0:
        hint = f"；名称相近的候选：{_describe_rows(partial)}" if len(partial) else ""
        fail(f"中证指数目录中没有简称或全称为“{norm}”的指数{hint}")
    if len(codes) > 1:
        fail(f"中证指数目录中匹配到多条：{_describe_rows(exact)}")
    row = exact.iloc[0]
    code = str(row["指数代码"])
    comp = _code_component(display, code, "catalog", return_type)
    if index_record(code) is None:
        comp = ResolvedComponent(
            display, code, UNKNOWN, "csindex", CNY, "catalog",
            f"{row['指数简称']}（{row['指数全称']}），收益类型未经核实，按价格指数的限定语处理",
        )
    return _check_currency(_with_catalog_currency(comp, frame))


def _with_catalog_currency(comp: ResolvedComponent, catalog: pd.DataFrame | None) -> ResolvedComponent:
    if catalog is None:
        return comp
    rows = catalog[catalog["指数代码"] == comp.code]
    if rows.empty:
        return comp
    currency = str(rows.iloc[0]["指数币种"]) or comp.currency
    return ResolvedComponent(**{**comp.__dict__, "currency": currency})


def _check_currency(comp: ResolvedComponent) -> ResolvedComponent:
    if comp.currency in (CNY, "", None):
        return comp
    msg = f"基准成分 {comp.name}（{comp.code}）以{comp.currency}计价，{FX_CAVEAT}"
    warnings.warn(msg, RuntimeWarning, stacklevel=3)
    note = f"{comp.note}；{FX_CAVEAT}" if comp.note else FX_CAVEAT
    return ResolvedComponent(**{**comp.__dict__, "note": note})


def benchmark_map_hint(names: Sequence[str]) -> str:
    """--benchmark-map 的写法提示，如 ``--benchmark-map "中债总指数=cbond:composite"``。"""
    example = ",".join(f"{n}=代码" for n in names) if names else "名称=代码"
    return (
        f'请用 --benchmark-map "{example}" 指定代码（Python 用 overrides={{名称: 代码}}），'
        '例如 --benchmark-map "中债总指数=cbond:composite"；现金可写 cash:年化利率'
    )


def resolve_benchmark(
    text: str,
    catalog=None,
    overrides: Mapping[str, str] | None = None,
    *,
    return_type: str = TOTAL,
    deposit_rate: float = DEMAND_DEPOSIT_RATE,
    time_deposit_rate: float = TIME_DEPOSIT_RATE,
) -> list[ResolvedComponent]:
    """解析合同业绩比较基准原文：parse_benchmark 拆出成分与权重，逐个 resolve_component。

    任一成分无法解析时，收集全部未解析的成分后一并报错（BenchmarkResolutionError），
    列出原因并提示 --benchmark-map 的写法，不做猜测。
    """
    parts = parse_benchmark(text)
    out: list[ResolvedComponent] = []
    unresolved: list[tuple[str, str]] = []
    for name, weight in parts:
        try:
            comp = resolve_component(
                name, catalog, overrides, return_type=return_type,
                deposit_rate=deposit_rate, time_deposit_rate=time_deposit_rate,
            )
        except BenchmarkResolutionError as exc:
            unresolved += exc.unresolved
            continue
        out.append(ResolvedComponent(**{**comp.__dict__, "weight": weight}))
    if unresolved:
        lines = "；".join(f"“{n}”：{reason}" for n, reason in unresolved)
        raise BenchmarkResolutionError(
            f"合同业绩比较基准“{text}”中有 {len(unresolved)} 个成分无法解析：{lines}。"
            + benchmark_map_hint([n for n, _ in unresolved]),
            unresolved,
        )
    return out


def resolution_table(components: Sequence[ResolvedComponent]) -> pd.DataFrame:
    """解析结果表：成分、权重、代码、收益类型、币种、数据源、解析依据、说明。"""
    labels = {**RETURN_TYPE_LABELS, CASH_RETURN: "现金（常数利率）"}
    sources = {"em": "东方财富", "csindex": "中证指数官网", "cbond": "中债", "constant": "常数"}
    return pd.DataFrame(
        [
            {
                "成分": c.name,
                "权重": c.weight,
                "代码": c.code,
                "收益类型": labels.get(c.return_type, c.return_type),
                "币种": c.currency,
                "数据源": sources.get(c.source, c.source),
                "解析依据": RESOLUTION_METHODS.get(c.method, c.method),
                "说明": c.note,
            }
            for c in components
        ]
    )


def components_return_type(components: Sequence[ResolvedComponent]) -> str:
    """复合基准的收益类型标签（同 benchmark_return_type）：现金类成分按全收益计（利息即收益）。"""
    types = [TOTAL if c.is_cash else c.return_type for c in components]
    if not types:
        return RETURN_TYPE_LABELS[UNKNOWN]
    if all(t == TOTAL for t in types):
        return RETURN_TYPE_LABELS[TOTAL]
    if all(t == PRICE for t in types):
        return RETURN_TYPE_LABELS[PRICE]
    if PRICE in types:
        return RETURN_TYPE_LABELS["mixed"]
    return RETURN_TYPE_LABELS[UNKNOWN]


def cash_returns(rate: float, index, periods_per_year: int) -> pd.Series:
    """常数年化利率 y 换算为每期收益 (1 + y)^(1/K) − 1，与无风险收益同一复利口径。"""
    value = (1 + float(rate)) ** (1 / int(periods_per_year)) - 1
    return pd.Series(value, index=pd.DatetimeIndex(index), dtype=float)


def _normalize_weights(weights, columns: Sequence[str]) -> pd.Series:
    if isinstance(weights, Mapping):
        w = pd.Series({str(k): float(v) for k, v in weights.items()})
        extra = set(w.index) - set(columns)
        missing = set(columns) - set(w.index)
        if extra or missing:
            raise ValueError(f"权重与成分不一致：缺少 {sorted(missing)}，多出 {sorted(extra)}")
        w = w.reindex(columns)
    else:
        arr = np.asarray(list(weights), dtype=float)
        if len(arr) != len(columns):
            raise ValueError(f"权重个数 {len(arr)} 与成分个数 {len(columns)} 不一致")
        w = pd.Series(arr, index=columns)
    if not np.isfinite(w.to_numpy()).all():
        raise ValueError("权重存在缺失或无穷值")
    if abs(w.sum() - 1) > WEIGHT_TOL:
        raise ValueError(f"权重须加总为 1，当前为 {w.sum():.10f}")
    return w


def composite_benchmark(components, weights, rebalance: str = "period") -> pd.Series:
    """按权重合成复合基准收益。

    ``components`` 为 DataFrame（每列一个成分的单期收益）或 {名称: Series} 字典，
    各成分按日期取交集对齐；``weights`` 为 {名称: 权重} 或与列顺序一致的序列，须加总为 1。

    - ``rebalance="period"``（默认）：每期期初再平衡回目标权重，r_b,t = Σ w_i r_i,t
    - ``rebalance="none"``：期初按权重买入后不再平衡，
      W_t = Σ w_i ∏_{s≤t}(1 + r_i,s)，r_b,t = W_t / W_{t-1} − 1

    任一成分当期缺失时该期结果为 NaN，不按零收益填补；不再平衡时缺失之后的各期
    无法确定持仓，也为 NaN。
    """
    if isinstance(components, Mapping):
        if not components:
            raise ValueError("至少需要一个成分")
        df = pd.concat({str(k): v.astype(float) for k, v in components.items()}, axis=1, join="inner")
    elif isinstance(components, pd.DataFrame):
        df = components.astype(float).copy()
        df.columns = [str(c) for c in df.columns]
    else:
        raise TypeError("components 须为 DataFrame 或 {名称: Series}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise schema.SchemaError("成分收益的索引必须是 DatetimeIndex")
    df = df.sort_index()
    w = _normalize_weights(weights, list(df.columns))
    any_missing = df.isna().any(axis=1)
    if rebalance == "period":
        out = df.mul(w, axis=1).sum(axis=1).mask(any_missing)
    elif rebalance == "none":
        broken = any_missing.cumsum() > 0
        growth = (1 + df.fillna(0.0)).cumprod()
        wealth = growth.mul(w, axis=1).sum(axis=1)
        prev = wealth.shift(1).fillna(1.0)
        out = (wealth / prev - 1).mask(broken)
    else:
        raise ValueError(f"rebalance 须为 'period' 或 'none'，收到 {rebalance!r}")
    return out.rename(schema.BENCHMARK)
