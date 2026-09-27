"""复合基准：正文第一部分“基准选择”。

基金合同中的业绩比较基准常写作“沪深300指数收益率×80%+中债综合指数收益率×20%”。
``parse_benchmark`` 只解析能确定无歧义的写法，``composite_benchmark`` 按权重合成收益。
合同基准通常隐含每日（每期）再平衡；成分指数是否含分红（价格指数或全收益指数）
直接影响基准收益，须在报告口径中写明。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from fundeval.etl import schema

#: 权重加总的容差
WEIGHT_TOL = 1e-8

#: 常见指数名称到代码的对照，供 lookup_index_code 使用。
#: 代码前缀 ``cbond:`` 表示中债指数（中国债券信息网），取财富（全收益）指数。
#: 名称只按下表精确匹配，表外名称报错，由调用方手动指定代码。
INDEX_CODES: dict[str, str] = {
    "沪深300": "000300",
    "中证500": "000905",
    "中证800": "000906",
    "中证1000": "000852",
    "上证50": "000016",
    "上证综合": "000001",
    "上证综指": "000001",
    "深证成份": "399001",
    "深证成指": "399001",
    "创业板": "399006",
    "创业板指": "399006",
    "中证全债": "H11001",
    "中证综合债": "H11009",
    "中债综合": "cbond:composite",
    "中债新综合": "cbond:new_composite",
}

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
