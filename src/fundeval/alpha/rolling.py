"""稳定性与样本外检验：正文第四部分第 3 节。

- 稳定性：滚动估计 Alpha、Beta、IR，观察优势是否集中于少数月份或特定市场
- 样本外：按时间顺序留出验证期，禁止随机打乱
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from statsmodels.regression.rolling import RollingOLS

from fundeval._utils import ZERO_TOL, paired
from fundeval.alpha.regression import ALPHA, excess_frame


def _check_window(window: int, minimum: int) -> int:
    if isinstance(window, bool) or int(window) != window or window < minimum:
        raise ValueError(f"窗口须为不小于 {minimum} 的整数，收到 {window}")
    return int(window)


def rolling_regression(returns, factors, window: int, risk_free=0.0) -> pd.DataFrame:
    """滚动因子回归：每个窗口估计 r_p - r_f = α + Σ β_k F_k + ε。

    返回按期（窗口末期）排列的 DataFrame，列为 ``alpha``、各因子系数、
    ``alpha_t``（普通 OLS 标准误下的 t 值）与 ``r_squared``。前 window - 1 期
    以及窗口内含缺失值的期为 NaN。输入口径与 factor_regression 相同。
    Alpha 为每期值，与输入同频。
    """
    y, f = excess_frame(returns, factors, risk_free)
    w = _check_window(window, f.shape[1] + 2)
    exog = f.copy()
    exog.insert(0, ALPHA, 1.0)
    res = RollingOLS(y, exog, window=w, missing="skip").fit()
    out = res.params.copy()
    out["alpha_t"] = res.tvalues[ALPHA]
    out["r_squared"] = res.rsquared
    return out


def rolling_information_ratio(portfolio, benchmark, window: int, periods_per_year: int) -> pd.Series:
    """滚动年化信息比率 IR_t = mean(a) / s(a) × √K，a = r_p - r_b，窗口内样本标准差。

    ``periods_per_year`` 必须显式给出。窗口内 TE 为零时返回 NaN，前 window - 1 期为 NaN。
    """
    w = _check_window(window, 2)
    p, b = paired(portfolio, benchmark)
    a = p - b
    mean = a.rolling(w).mean()
    std = a.rolling(w).std(ddof=1)
    ir = (mean / std.where(std.abs() >= ZERO_TOL)) * np.sqrt(periods_per_year)
    return ir.rename("rolling_ir")


def split_in_out_of_sample(data, split):
    """按时间顺序切分样本内与样本外，返回 (in_sample, out_of_sample)。

    - ``split`` 为日期（字符串、Timestamp 或 date）：索引不晚于该日期的行进入样本内，
      其余进入样本外
    - ``split`` 为 (0, 1) 内的比例：前 ⌊ratio × n⌋ 行进入样本内

    数据须按时间升序排列，不做排序或随机打乱；索引非升序时报错，提醒先检查数据。
    任一部分为空时报错。
    """
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        data = pd.Series(np.asarray(data, dtype=float))
    idx = data.index
    if not idx.is_monotonic_increasing:
        raise ValueError("数据索引须按时间升序排列；样本内外须按时间顺序切分，不能打乱")
    if isinstance(split, (bool, np.bool_)):
        raise TypeError("split 须为日期或 (0, 1) 内的比例")
    if isinstance(split, (int, float, np.integer, np.floating)):
        if not 0 < split < 1:
            raise ValueError(f"比例须在 (0, 1) 之间，收到 {split}")
        m = split * len(data)
        n_in = round(m) if abs(m - round(m)) < 1e-9 else math.floor(m)  # 0.29 × 100 不应取成 28
        inside, outside = data.iloc[:n_in], data.iloc[n_in:]
    else:
        if not isinstance(idx, pd.DatetimeIndex):
            raise TypeError("按日期切分需要 DatetimeIndex")
        cut = pd.Timestamp(split)
        inside, outside = data[idx <= cut], data[idx > cut]
    if len(inside) == 0 or len(outside) == 0:
        raise ValueError(f"切分后样本内 {len(inside)} 期、样本外 {len(outside)} 期，两部分都不能为空")
    return inside, outside
