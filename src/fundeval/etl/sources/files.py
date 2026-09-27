"""从本地 CSV 或 Excel 读取为标准结构。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from fundeval.etl import schema


def read_table(path: str | Path, sheet_name: str | int = 0, **kwargs) -> pd.DataFrame:
    """按扩展名读取 CSV（.csv/.txt）或 Excel（.xlsx/.xls）。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, **kwargs)
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name, **kwargs)
    raise ValueError(f"不支持的文件类型：{suffix}")


def _parse_number(col: pd.Series, percent: bool) -> pd.Series:
    """接受数值或 "2.00%" 形式文本；percent=True 时数值按百分数除以 100。"""
    if not pd.api.types.is_numeric_dtype(col):
        text = col.astype(str).str.strip().str.replace("−", "-", regex=False)
        has_pct = text.str.endswith("%")
        num = pd.to_numeric(text.str.rstrip("%").str.replace(",", "", regex=False), errors="coerce")
        num = num.where(~has_pct, num / 100)
        if percent:
            num = num.where(has_pct, num / 100)
        num[col.isna()] = float("nan")
        return num
    num = pd.to_numeric(col, errors="coerce")
    return num / 100 if percent else num


def _indexed(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if date_col not in df.columns:
        raise schema.SchemaError(f"缺少日期列：{date_col!r}")
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    return out.set_index(date_col).sort_index().rename_axis(schema.DATE)


def load_returns(
    path: str | Path,
    columns: Mapping[str, str] | None = None,
    date_col: str = schema.DATE,
    percent: bool = False,
    **read_kwargs,
) -> pd.DataFrame:
    """读取收益表。

    ``columns`` 把文件中的列名映射到标准列名，例如
    ``{"组合收益": "portfolio", "基准收益": "benchmark", "无风险收益": "risk_free"}``。
    未给出时，文件须已使用标准列名。``percent=True`` 表示数值以百分数存储。
    """
    raw = read_table(path, **read_kwargs)
    if columns:
        raw = raw.rename(columns=dict(columns))
    df = _indexed(raw, date_col)
    keep = [c for c in schema.RETURN_COLUMNS if c in df.columns]
    if not keep:
        raise schema.SchemaError(f"未找到收益列，需要其中之一：{schema.RETURN_COLUMNS}")
    out = pd.DataFrame({c: _parse_number(df[c], percent) for c in keep}, index=df.index)
    return schema.validate_returns(out, required=[keep[0]], allow_missing=True)


def load_series(
    path: str | Path,
    value_col: str,
    date_col: str = schema.DATE,
    name: str | None = None,
    percent: bool = False,
    **read_kwargs,
) -> pd.Series:
    """读取单列时间序列（净值、账户资产、现金流等）。"""
    df = _indexed(read_table(path, **read_kwargs), date_col)
    if value_col not in df.columns:
        raise schema.SchemaError(f"缺少列：{value_col!r}")
    return _parse_number(df[value_col], percent).rename(name or value_col)


def load_nav(path: str | Path, nav_col: str = schema.NAV, date_col: str = schema.DATE, **read_kwargs) -> pd.Series:
    """读取复权单位净值序列并校验。"""
    nav = load_series(path, nav_col, date_col=date_col, name=schema.NAV, **read_kwargs)
    return schema.validate_nav(nav)


def load_cashflows(
    path: str | Path,
    amount_col: str = schema.CASHFLOW,
    date_col: str = schema.DATE,
    **read_kwargs,
) -> pd.Series:
    """读取外部现金流。同一日期多笔现金流会合并。正值表示流入账户。"""
    flows = load_series(path, amount_col, date_col=date_col, name=schema.CASHFLOW, **read_kwargs)
    flows = flows.groupby(level=0).sum()
    return schema.validate_cashflows(flows)
