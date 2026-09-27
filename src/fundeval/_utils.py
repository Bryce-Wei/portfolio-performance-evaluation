"""内部工具：输入统一为 Series、安全除法。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def as_series(values, name: str | None = None) -> pd.Series:
    """把 Series、列表或数组统一为 float 型 Series，不改动原对象。"""
    if isinstance(values, pd.Series):
        s = values.astype(float)
    elif isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError("需要单列数据，收到的 DataFrame 有多列")
        s = values.iloc[:, 0].astype(float)
    else:
        s = pd.Series(np.asarray(values, dtype=float))
    if name is not None:
        s = s.rename(name)
    return s


def broadcast_like(values, like: pd.Series) -> pd.Series:
    """标量按 like 的索引广播；序列须与 like 同长或可按索引对齐。"""
    if np.isscalar(values):
        return pd.Series(float(values), index=like.index)
    s = as_series(values)
    if isinstance(values, pd.Series):
        aligned = s.reindex(like.index)
        if aligned.isna().any() and not like.isna().all():
            missing = like.index[aligned.isna()]
            raise ValueError(f"序列未覆盖全部观测期，缺少 {len(missing)} 期：{list(missing[:5])}")
        return aligned
    if len(s) != len(like):
        raise ValueError(f"长度不一致：{len(s)} 与 {len(like)}")
    return pd.Series(s.to_numpy(), index=like.index)


def paired(a, b) -> tuple[pd.Series, pd.Series]:
    """把两组收益对齐到同一索引；两者都是 Series 时按索引取交集，否则按位置配对。"""
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        idx = a.index.intersection(b.index, sort=False)
        if len(idx) != len(a) or len(idx) != len(b):
            raise ValueError("两组收益的观测期不一致，请先用 fundeval.etl.clean.align 对齐")
        return a.astype(float), b.reindex(a.index).astype(float)
    sa, sb = as_series(a), as_series(b)
    if len(sa) != len(sb):
        raise ValueError(f"长度不一致：{len(sa)} 与 {len(sb)}")
    return sa, pd.Series(sb.to_numpy(), index=sa.index)


#: 小于该值的分母视为零（浮点误差下常数序列的标准差约为 1e-18）
ZERO_TOL = 1e-12


def safe_div(num: float, den: float) -> float:
    """分母为零或非有限时返回 NaN，对应正文“比率分母为零时报告不适用”。"""
    if not np.isfinite(den) or not np.isfinite(num) or abs(den) < ZERO_TOL:
        return float("nan")
    return float(num / den)
