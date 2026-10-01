"""标准数据结构与校验（正文第一部分第 2 节“建立最小数据集”）。

记号约定沿用正文：p 组合，b 基准，m 市场，f 无风险资产；K 为一年期数。
收益表以日期为索引，列名使用下列常量；收益一律为小数。

- 基础层：收益表、净值、外部现金流（validate_returns、validate_nav、validate_cashflows）
- 归因层：持仓表（date、asset、sector、weight，可选 market_value），validate_holdings；
  sector_weights 汇总为每期行业权重，可直接作为 attribution.brinson.brinson_multi_period 的权重输入
- 执行层：成交表（date、asset、side、amount、price、fee），validate_trades；
  turnover_from_trades 按成交金额调用 costs.turnover 计算换手率
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

DATE = "date"
PORTFOLIO = "portfolio"
BENCHMARK = "benchmark"
MARKET = "market"
RISK_FREE = "risk_free"
NAV = "nav"
PRICE = "price"
DIVIDEND = "dividend"
VALUE = "value"
CASHFLOW = "cashflow"

RETURN_COLUMNS = (PORTFOLIO, BENCHMARK, MARKET, RISK_FREE)

# 持仓表与成交表的标准列名
ASSET = "asset"
SECTOR = "sector"
WEIGHT = "weight"
MARKET_VALUE = "market_value"
SIDE = "side"
AMOUNT = "amount"
FEE = "fee"

HOLDING_COLUMNS = (DATE, ASSET, SECTOR, WEIGHT)
HOLDING_OPTIONAL_COLUMNS = (MARKET_VALUE,)
TRADE_COLUMNS = (DATE, ASSET, SIDE, AMOUNT, PRICE, FEE)

#: 成交方向
BUY = "buy"
SELL = "sell"
#: 成交方向的其他写法 → buy / sell
SIDE_ALIASES = {"buy": BUY, "b": BUY, "买": BUY, "买入": BUY, "sell": SELL, "s": SELL, "卖": SELL, "卖出": SELL}

#: 现金在持仓表中单列为一个行业的默认名称
CASH_SECTOR = "现金"
#: 视为现金的资产名（这些资产的 sector 须为现金行业）
CASH_ASSETS = ("现金", "cash", "CASH", "Cash")

#: 持仓权重每期加总为 1 的默认容差（披露的权重常保留两位百分数，加总可能差几个基点）
HOLDING_WEIGHT_TOL = 1e-4

#: 一年期数 K。日度按 252 个交易日约定，如另有约定应显式传入。
PERIODS_PER_YEAR = {"D": 252, "W": 52, "M": 12, "Q": 4, "A": 1}


class SchemaError(ValueError):
    """数据不符合标准结构。"""


def periods_per_year(freq: str | int) -> int:
    """把频率代码（D/W/M/Q/A）或整数转换为 K。"""
    if isinstance(freq, (int, np.integer)):
        if freq <= 0:
            raise SchemaError(f"K 必须为正整数，收到 {freq}")
        return int(freq)
    key = str(freq).upper()[:1]
    if key == "Y":
        key = "A"
    if key not in PERIODS_PER_YEAR:
        raise SchemaError(f"无法识别的收益频率：{freq!r}")
    return PERIODS_PER_YEAR[key]


def _check_index(obj: pd.Series | pd.DataFrame, what: str) -> None:
    if not isinstance(obj.index, pd.DatetimeIndex):
        raise SchemaError(f"{what} 的索引必须是 DatetimeIndex")
    if obj.index.has_duplicates:
        dup = obj.index[obj.index.duplicated()].unique()
        raise SchemaError(f"{what} 存在重复日期：{list(dup[:5])}")
    if not obj.index.is_monotonic_increasing:
        raise SchemaError(f"{what} 的日期未按升序排列")


def validate_returns(
    df: pd.DataFrame,
    required: Iterable[str] = (PORTFOLIO,),
    allow_missing: bool = False,
) -> pd.DataFrame:
    """校验收益表：日期索引唯一升序、所需列存在、数值为有限小数且大于 -100%。"""
    _check_index(df, "收益表")
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise SchemaError(f"收益表缺少列：{missing_cols}")
    cols = [c for c in df.columns if c in RETURN_COLUMNS]
    values = df[cols].apply(pd.to_numeric, errors="raise")
    if not allow_missing and values[list(required)].isna().any().any():
        raise SchemaError("收益表必需列存在缺失值；缺失不应机械填零，请先复核")
    arr = values.to_numpy(dtype=float)
    finite = arr[~np.isnan(arr)]
    if np.isinf(finite).any():
        raise SchemaError("收益表存在无穷值")
    if (finite <= -1).any():
        raise SchemaError("存在小于等于 -100% 的收益，请确认是否误用了百分数")
    return df


def validate_nav(nav: pd.Series) -> pd.Series:
    """校验净值序列：日期索引唯一升序、严格为正、无缺失。"""
    _check_index(nav, "净值序列")
    if nav.isna().any():
        raise SchemaError("净值序列存在缺失值")
    if (nav <= 0).any():
        raise SchemaError("净值必须为正")
    return nav


def validate_cashflows(flows: pd.Series) -> pd.Series:
    """校验外部现金流序列：日期索引升序、数值有限。正值表示流入账户。"""
    if not isinstance(flows.index, pd.DatetimeIndex):
        raise SchemaError("现金流序列的索引必须是 DatetimeIndex")
    if not flows.index.is_monotonic_increasing:
        raise SchemaError("现金流序列的日期未按升序排列")
    if not np.isfinite(flows.to_numpy(dtype=float)).all():
        raise SchemaError("现金流序列存在缺失或无穷值")
    return flows


def _table(df: pd.DataFrame, columns: Iterable[str], what: str) -> pd.DataFrame:
    """检查必需列、把日期列转为 Timestamp，返回副本。"""
    if not isinstance(df, pd.DataFrame):
        raise SchemaError(f"{what}须为 DataFrame")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SchemaError(f"{what}缺少列：{missing}（标准列为 {list(columns)}）")
    out = df.copy()
    try:
        out[DATE] = pd.to_datetime(out[DATE])
    except (ValueError, TypeError) as exc:
        raise SchemaError(f"{what}的 date 列无法解析为日期：{exc}") from exc
    return out


def _require_complete(df: pd.DataFrame, columns: Iterable[str], what: str) -> None:
    for col in columns:
        na = df[col].isna()
        if na.any():
            raise SchemaError(f"{what}的 {col} 列有 {int(na.sum())} 个缺失值；缺失不应机械填零，请先复核")


def _numeric(df: pd.DataFrame, col: str, what: str) -> pd.Series:
    try:
        values = pd.to_numeric(df[col], errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise SchemaError(f"{what}的 {col} 列含非数值：{exc}") from exc
    if np.isinf(values.dropna().to_numpy()).any():
        raise SchemaError(f"{what}的 {col} 列存在无穷值")
    return values


def validate_holdings(
    holdings: pd.DataFrame, tol: float = HOLDING_WEIGHT_TOL, cash_sector: str = CASH_SECTOR
) -> pd.DataFrame:
    """校验持仓表（正文第一部分第 2 节归因层：期初持仓权重、行业分类），返回按 (date, asset) 排序的副本。

    标准列：``date``、``asset``（证券代码或名称）、``sector``（行业）、``weight``（占组合净值的比例，小数），
    可选 ``market_value``（市值，可缺失）。每行为某期期初的一个持仓。

    - 必需列不得缺失；同一期同一资产不得重复
    - 权重每期加总为 1，容差 ``tol``（默认 1e-4）；现金须单列为一个行业（``cash_sector``，默认“现金”）计入权重，
      资产名为“现金”或 cash 的行其行业必须是该现金行业
    - ``market_value`` 若给出须非负；缺失保留为 NaN，不填零
    """
    what = "持仓表"
    if tol < 0:
        raise ValueError("tol 须为非负数")
    df = _table(holdings, HOLDING_COLUMNS, what)
    _require_complete(df, HOLDING_COLUMNS, what)
    df[WEIGHT] = _numeric(df, WEIGHT, what)
    df[ASSET] = df[ASSET].astype(str)
    df[SECTOR] = df[SECTOR].astype(str)
    dup = df.duplicated([DATE, ASSET])
    if dup.any():
        first = df.loc[dup].iloc[0]
        raise SchemaError(f"{what}在 {first[DATE]:%Y-%m-%d} 的资产 {first[ASSET]} 重复")
    cash_rows = df[ASSET].isin(CASH_ASSETS) & (df[SECTOR] != cash_sector)
    if cash_rows.any():
        bad = df.loc[cash_rows].iloc[0]
        raise SchemaError(
            f"{what}在 {bad[DATE]:%Y-%m-%d} 的现金行业为 {bad[SECTOR]!r}，现金须单列为行业“{cash_sector}”"
        )
    totals = df.groupby(DATE)[WEIGHT].sum()
    off = totals[(totals - 1).abs() > tol]
    if len(off):
        date, total = off.index[0], off.iloc[0]
        raise SchemaError(
            f"{what}在 {date:%Y-%m-%d} 的权重合计为 {total:.6f}，须为 1（容差 {tol:g}，共 {len(off)} 期不满足）；"
            f"现金须单列为行业“{cash_sector}”计入权重"
        )
    if MARKET_VALUE in df.columns:
        mv = _numeric(df, MARKET_VALUE, what)
        if (mv.dropna() < 0).any():
            raise SchemaError(f"{what}的 market_value 不能为负")
        df[MARKET_VALUE] = mv
    return df.sort_values([DATE, ASSET]).reset_index(drop=True)


def validate_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """校验成交表（正文第一部分第 2 节执行层：成交金额、成交价格、费用），返回按日期排序的副本。

    标准列：``date``、``asset``、``side``（buy / sell，也接受 买入 / 卖出、b / s 等写法，统一为 buy / sell）、
    ``amount``（成交金额，正数）、``price``（成交价格，正数）、``fee``（费用，非负）。全部列不得缺失。
    """
    what = "成交表"
    df = _table(trades, TRADE_COLUMNS, what)
    _require_complete(df, TRADE_COLUMNS, what)
    df[ASSET] = df[ASSET].astype(str)
    side = df[SIDE].astype(str).str.strip()
    mapped = side.map(lambda v: SIDE_ALIASES.get(v, SIDE_ALIASES.get(v.lower())))
    if mapped.isna().any():
        raise SchemaError(f"{what}的 side 只能为 buy 或 sell，收到 {sorted(set(side[mapped.isna()]))}")
    df[SIDE] = mapped
    for col in (AMOUNT, PRICE, FEE):
        df[col] = _numeric(df, col, what)
    if (df[AMOUNT] <= 0).any():
        raise SchemaError(f"{what}的 amount（成交金额）须为正数，方向由 side 表示")
    if (df[PRICE] <= 0).any():
        raise SchemaError(f"{what}的 price 须为正数")
    if (df[FEE] < 0).any():
        raise SchemaError(f"{what}的 fee 不能为负")
    return df.sort_values(DATE, kind="stable").reset_index(drop=True)


def sector_weights(holdings: pd.DataFrame, tol: float = HOLDING_WEIGHT_TOL) -> pd.DataFrame:
    """把持仓按行业汇总为每期权重：行为期（date），列为行业，值为权重之和。

    先按 validate_holdings 校验。某期没有持仓的行业权重为 0：这是“未持有”，不是缺失值。
    结果可直接作为 attribution.brinson.brinson_multi_period 的 ``wp`` 或 ``wb``（组合与基准的行业须一致，
    可用 ``reindex(columns=..., fill_value=0)`` 对齐到同一组行业）。
    """
    df = validate_holdings(holdings, tol=tol)
    out = df.pivot_table(index=DATE, columns=SECTOR, values=WEIGHT, aggfunc="sum", fill_value=0.0)
    out.columns.name = SECTOR
    return out.astype(float)


def turnover_from_trades(trades: pd.DataFrame, average_nav) -> float:
    """由成交表计算换手率：买入、卖出成交金额分别加总后调用 costs.turnover（正文第六部分）。

    TO = Σ(|B_t| + |S_t|) / (2 × 平均资产净值)。``average_nav`` 为期间平均资产净值（标量或序列）。
    成交表覆盖一年时即年化换手率；费用不计入成交金额。
    """
    from fundeval import costs

    df = validate_trades(trades)
    buys = df.loc[df[SIDE] == BUY, AMOUNT]
    sells = df.loc[df[SIDE] == SELL, AMOUNT]
    return costs.turnover(buys.to_numpy() if len(buys) else 0.0, sells.to_numpy() if len(sells) else 0.0, average_nav)
