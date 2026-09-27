"""持续风险监控与预警：正文第七部分。

借鉴 Green Zone 分区思路，同时比较两项指标：

- 风险倍数 = 实现年化 TE / 目标年化 TE（主动风险相对预算的偏离）
- z 值 = (Σa - μ·h/K) / (TE_target·√(h/K))（按期限标准化的收益偏离）

注意：本模块的全部阈值（0.8、1.2、1.5、2、3 及连续 Red 期数）是正文中的演示值，
须根据策略、历史分布和风险容忍度校准，并非行业标准。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from fundeval._utils import ZERO_TOL, as_series, safe_div

GREEN = "green"
YELLOW = "yellow"
RED = "red"

_LEVEL = {GREEN: 0, YELLOW: 1, RED: 2}


def _usable_denominator(den: float) -> bool:
    return bool(np.isfinite(den) and abs(den) >= ZERO_TOL)


def realized_tracking_error(
    active, window: int | Sequence[int] = 20, periods_per_year: int = 252
) -> pd.Series | pd.DataFrame:
    """滚动窗口实现跟踪误差，年化：TE_t = s(a_{t-w+1..t}) × √K。

    ``active`` 为日度主动收益 a_t；标准差使用样本口径（n-1），与 risk.tracking_error 一致。
    ``window`` 为单个窗口时返回 Series；给出多个窗口（如 ``(20, 60)``）时返回
    以 ``te_20``、``te_60`` 为列的 DataFrame。前 w-1 期数据不足，结果为 NaN。
    短窗口噪声较大；事后实现 TE 与模型预测 TE 应分别保留，不能互相替代。
    """
    a = as_series(active)
    windows = [window] if np.isscalar(window) else list(window)
    for w in windows:
        if int(w) != w or w < 2:
            raise ValueError(f"窗口须为不小于 2 的整数，收到 {w}")
    scale = np.sqrt(periods_per_year)
    out = {f"te_{int(w)}": a.rolling(int(w)).std(ddof=1) * scale for w in windows}
    if np.isscalar(window):
        return out[f"te_{int(window)}"].rename("realized_te")
    return pd.DataFrame(out)


def risk_multiple(realized_te, target_te: float):
    """风险倍数 = 实现年化 TE / 目标年化 TE。目标 TE 为零时返回 NaN。

    ``realized_te`` 可为标量，或 realized_tracking_error 返回的 Series / DataFrame。
    """
    if np.isscalar(realized_te):
        return safe_div(float(realized_te), float(target_te))
    if not _usable_denominator(float(target_te)):
        return realized_te * np.nan
    return realized_te / float(target_te)


def expected_active_return(mu_ann: float, h: int, periods_per_year: int = 252) -> float:
    """h 期预算主动收益 μ_ann × h / K（收益按时间线性缩放）。"""
    return mu_ann * h / periods_per_year


def target_risk(te_target_ann: float, h: int, periods_per_year: int = 252) -> float:
    """h 期目标风险 TE_target × √(h / K)（风险按时间平方根缩放）。"""
    return te_target_ann * np.sqrt(h / periods_per_year)


def z_score(active_sum, h: int, mu_ann: float, te_target_ann: float, periods_per_year: int = 252):
    """按期限标准化的收益偏离 z = (Σa - μ·h/K) / (TE_target·√(h/K))。

    ``active_sum`` 为 h 期算术主动收益之和，可为标量或 Series（如 ``a.rolling(h).sum()``）。
    z 值基于独立同分布近似，只用于诊断，不能直接解释为严格概率。改用复利超额收益
    或收益明显自相关时，应重新估计同期限均值与方差。目标 TE 为零时返回 NaN。
    """
    budget = expected_active_return(mu_ann, h, periods_per_year)
    risk = target_risk(te_target_ann, h, periods_per_year)
    if np.isscalar(active_sum):
        return safe_div(float(active_sum) - budget, risk)
    if not _usable_denominator(risk):
        return active_sum * np.nan
    return (active_sum - budget) / risk


def _classify_one(rm: float, z: float, green_band, z_green, red_multiple, z_red) -> str:
    rm_ok, z_ok = np.isfinite(rm), np.isfinite(z)
    if (rm_ok and rm > red_multiple) or (z_ok and abs(z) >= z_red):
        return RED
    if rm_ok and z_ok and green_band[0] <= rm <= green_band[1] and abs(z) < z_green:
        return GREEN
    return YELLOW


def classify(
    risk_multiple,
    z,
    green_band: tuple[float, float] = (0.8, 1.2),
    z_green: float = 2.0,
    red_multiple: float = 1.5,
    z_red: float = 3.0,
):
    """按两项指标分区，返回 "green" / "yellow" / "red"。

    - Green：风险倍数在 green_band 内（含端点），且 |z| < z_green
    - Red：风险倍数 > red_multiple，或 |z| ≥ z_red
    - Yellow：其余情况，包括风险倍数低于下限（风险预算未使用）

    两项指标触发不同状态时取较高等级；收益偏离按 |z| 双侧检查，异常高收益同样预警。
    某项指标为 NaN（数据不足或目标 TE 为零）时，它不能触发 Red，但也不能判为 Green，
    结果至少为 Yellow，提示人工复核。

    默认阈值是正文的演示值，须按策略、历史分布和风险容忍度校准，并非行业标准。
    输入为标量时返回字符串；为 Series 时按索引对齐后返回状态 Series。
    """
    args = (green_band, z_green, red_multiple, z_red)
    if np.isscalar(risk_multiple) and np.isscalar(z):
        return _classify_one(float(risk_multiple), float(z), *args)
    rm_s = risk_multiple if isinstance(risk_multiple, pd.Series) else None
    z_s = z if isinstance(z, pd.Series) else None
    like = rm_s if rm_s is not None else z_s
    if like is None:
        rm_s, z_s = as_series(risk_multiple), as_series(z)
        if len(rm_s) != len(z_s):
            raise ValueError(f"长度不一致：{len(rm_s)} 与 {len(z_s)}")
        z_s.index = rm_s.index
    else:
        rm_s = rm_s if rm_s is not None else pd.Series(float(risk_multiple), index=like.index)
        z_s = z_s if z_s is not None else pd.Series(float(z), index=like.index)
        if not rm_s.index.equals(z_s.index):
            raise ValueError("风险倍数与 z 值的观测期不一致")
    status = [_classify_one(float(r), float(v), *args) for r, v in zip(rm_s, z_s)]
    return pd.Series(status, index=rm_s.index, name="status")


def worst_status(statuses: Iterable[str]) -> str:
    """多个状态取最高等级（red > yellow > green）。"""
    return max(statuses, key=lambda s: _LEVEL[s])


def consecutive_red(statuses, n: int = 3) -> pd.Series:
    """标记连续 n 个监控期均为 Red 的位置（第 n 个及之后仍持续为 Red 的期均为 True）。

    n 是需要事前规定的演示值，须按监控频率和策略校准。
    """
    if n < 1:
        raise ValueError("n 须为正整数")
    s = statuses if isinstance(statuses, pd.Series) else pd.Series(list(statuses))
    is_red = (s == RED).astype(int)
    run = is_red.groupby((is_red == 0).cumsum()).cumsum()  # 当前连续 Red 的长度
    return (run >= n).rename("consecutive_red")
