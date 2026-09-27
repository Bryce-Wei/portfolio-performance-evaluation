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

    名称可带括号说明（如“银行活期存款利率(税后)”）。含减号或嵌套运算、缺少权重、
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
        if not name or re.search(r"[+\-−×*%％]", name):
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
