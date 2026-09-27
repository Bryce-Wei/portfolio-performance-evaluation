"""交易成本与策略容量：正文第六部分。

- 换手率 TO = Σ(|B_t| + |S_t|) / (2 × 平均资产净值)
- 线性成本率 ≈ 2 × TO × c，c 为单位成交额成本（小数，20 个基点 = 0.0020）；买卖各计一次，故乘 2
- 近似净 Alpha：α_net ≈ α_gross − c_trade − c_fee
- 容量：逐资产所需参与率 = 交易金额 / (日均成交额 × 可用天数)，超过上限即容量受限
- 规模敏感性：规模扩大时成交额随之放大，冲击成本通常非线性上升

冲击成本没有通用参数：平方根模型 cost = Y·σ·√(Q/V) 只是常见形式，系数 Y 与日波动 σ 应优先用
成交记录校准，本模块不提供默认值。

公募基金披露的净值通常已扣除管理费、托管费与交易成本，用费用后净值估计的 Alpha 已是扣费后口径，
不能再扣一次。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from fundeval._utils import safe_div

#: 单位成交额成本超过该值（500 个基点）视为以百分数或基点输入
MAX_UNIT_COST = 0.05

ImpactModel = Callable[[np.ndarray], np.ndarray]


def _values(x, name: str) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(x.to_numpy() if isinstance(x, (pd.Series, pd.DataFrame)) else x, dtype=float))
    if np.isnan(arr).any():
        raise ValueError(f"{name} 有缺失值，不能按零处理，请先补齐或剔除对应期")
    return arr.ravel()


def _nonneg(value, name: str) -> float:
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"{name} 缺失或非有限值")
    if v < 0:
        raise ValueError(f"{name} 不能为负，收到 {v}")
    return v


def _check_unit_cost(c: float) -> float:
    c = _nonneg(c, "unit_cost")
    if c > MAX_UNIT_COST:
        raise ValueError(f"unit_cost = {c} 疑似以百分数或基点输入；请用小数，如 20 个基点写 0.0020")
    return c


def turnover(buys, sells, average_nav) -> float:
    """换手率 TO = Σ(|B_t| + |S_t|) / (2 × 平均资产净值)（正文第六部分）。

    ``buys``、``sells`` 为各期买入、卖出金额（标量或序列，取绝对值），``average_nav`` 为期间平均资产净值
    （标量；给出序列时取其均值）。金额缺失时报错，不填零；平均资产净值为零时返回 NaN，为负时报错。
    区间为一年时即年化换手率；否则须自行换算。
    """
    b = np.abs(_values(buys, "buys")).sum()
    s = np.abs(_values(sells, "sells")).sum()
    nav = float(np.mean(_values(average_nav, "average_nav")))
    if nav < 0:
        raise ValueError(f"平均资产净值不能为负，收到 {nav}")
    return safe_div(b + s, 2 * nav)


def linear_cost_rate(turnover: float, unit_cost: float) -> float:
    """线性交易成本率 ≈ 2 × TO × c。

    ``turnover`` 为换手率（小数，100% = 1.0），``unit_cost`` 为单位成交额成本（小数，20 个基点 = 0.0020，
    含佣金、印花税、买卖价差等线性部分）。换手率按单边定义，买卖两边都产生成本，故乘 2。
    例：TO = 100%、c = 20 bp → 成本约 40 bp。``unit_cost`` 大于 0.05 时视为单位错误而报错。
    """
    return 2 * _nonneg(turnover, "turnover") * _check_unit_cost(unit_cost)


def net_alpha(gross_alpha: float, trade_cost: float, other_fees: float = 0.0) -> float:
    """近似净 Alpha：α_net ≈ α_gross − c_trade − c_fee。

    三者须为同一期间、同一资产基数（如均为年化、占平均资产净值的比例）。这只是近似：成本发生的
    时点与收益路径相关，严格的净 Alpha 应对扣费后净收益序列重新回归。
    例：费用前 Alpha 4.00%、交易成本 1.20%、其他费用 0.60% → 近似净 Alpha 2.20%。
    """
    vals = [float(v) for v in (gross_alpha, trade_cost, other_fees)]
    if not all(np.isfinite(v) for v in vals):
        raise ValueError("gross_alpha、trade_cost、other_fees 须为有限数值")
    return vals[0] - vals[1] - vals[2]


def _asset_series(x, name: str, index=None) -> pd.Series:
    if isinstance(x, pd.Series):
        s = x.astype(float)
    elif isinstance(x, Mapping):
        s = pd.Series(x, dtype=float)
    elif np.isscalar(x):
        if index is None:
            s = pd.Series([float(x)])
        else:
            s = pd.Series(float(x), index=index)
    else:
        s = pd.Series(np.asarray(x, dtype=float))
    return s.rename(name)


def capacity_check(trade_amount, adv, max_participation: float, days: float = 1) -> pd.DataFrame:
    """逐资产的容量检查：所需参与率 = 交易金额 / (日均成交额 × 可用天数)。

    ``trade_amount`` 为各资产需完成的交易金额（Series，按资产索引，取绝对值），``adv`` 为日均成交额
    （同索引的 Series，或对全部资产相同的标量），``max_participation`` 为参与率上限（小数，如 0.10），
    ``days`` 为可用于完成交易的天数。

    返回表：交易金额、日均成交额、可用天数、所需参与率、上限、是否超限。日均成交额缺失时报错；
    为零时参与率为 NaN，“是否超限”记为“无法判断”。
    """
    mp = _nonneg(max_participation, "max_participation")
    if not 0 < mp <= 1:
        raise ValueError(f"max_participation 须在 (0, 1] 内（小数，如 0.10），收到 {mp}")
    d = _nonneg(days, "days")
    if d == 0:
        raise ValueError("days 须为正数")
    q = _asset_series(trade_amount, "交易金额").abs()
    v = _asset_series(adv, "日均成交额", index=q.index)
    missing = q.index.difference(v.index)
    if len(missing):
        raise ValueError(f"日均成交额缺少资产：{list(missing)}")
    v = v.reindex(q.index)
    if q.isna().any() or v.isna().any():
        raise ValueError("交易金额或日均成交额有缺失值，不能按零处理")
    if (v < 0).any():
        raise ValueError("日均成交额不能为负")
    part = pd.Series([safe_div(a, b * d) for a, b in zip(q, v)], index=q.index)
    flag = np.where(part.isna(), "无法判断", np.where(part > mp, "是", "否"))
    return pd.DataFrame(
        {
            "交易金额": q,
            "日均成交额": v,
            "可用天数": d,
            "所需参与率": part,
            "上限": mp,
            "是否超限": flag,
        }
    )


def square_root_impact(coefficient: float, daily_volatility) -> ImpactModel:
    """平方根冲击成本模型：cost(q) = Y × σ × √q，q 为参与率（交易金额 / 日均成交额）。

    ``coefficient`` 为系数 Y，``daily_volatility`` 为日收益波动 σ（标量或按资产的数组）。冲击成本通常
    随交易规模非线性上升，参数因市场、资产与交易方式差异很大，没有通用默认值，应优先用成交记录
    校准。返回可传给 cost_sensitivity 的函数：输入参与率数组，输出每单位成交额的冲击成本（小数）。
    """
    y = _nonneg(coefficient, "coefficient")
    sigma = _values(daily_volatility, "daily_volatility")
    if (sigma < 0).any():
        raise ValueError("daily_volatility 不能为负")

    def model(participation: np.ndarray) -> np.ndarray:
        q = np.asarray(participation, dtype=float)
        return y * sigma * np.sqrt(np.clip(q, 0, None))

    return model


def cost_sensitivity(
    aum_grid,
    turnover: float,
    adv,
    *,
    weights=None,
    unit_cost: float = 0.0,
    impact_model: ImpactModel | None = None,
    impact_params: Mapping[str, float] | None = None,
    trading_days: float = 250,
    max_participation: float | None = None,
) -> pd.DataFrame:
    """规模变化下的成本敏感性（正文第六部分“策略容量”）。

    对 ``aum_grid`` 中的每个规模 A：年成交额 = 2 × TO × A；资产 i 的日均交易额
    = 2 × TO × A × w_i / ``trading_days``，参与率 q_i = 日均交易额 / ADV_i；
    冲击成本率（占资产净值，年化）= 2 × TO × Σ w_i × impact(q_i)，线性成本率 = 2 × TO × c。

    参数
    ----
    aum_grid : 规模列表（与 adv 同一货币单位）
    turnover : 年化换手率 TO（小数）
    adv : 日均成交额，标量（整体）或按资产的 Series（须与 weights 同索引）
    weights : 各资产在成交中的权重（和为 1）；adv 为标量时省略
    unit_cost : 单位成交额的线性成本（小数，20 bp = 0.0020）
    impact_model : 冲击成本函数，输入参与率数组、输出每单位成交额的成本；省略时用平方根模型，
        其参数须由 ``impact_params``（coefficient、daily_volatility）给出，没有通用默认值
    trading_days : 一年中用于交易的天数，决定日均交易额（默认 250）
    max_participation : 给出时标出最大参与率超过上限的规模

    冲击成本通常非线性，应优先用成交记录校准；结果只用于比较规模变化的方向与量级。
    """
    to = _nonneg(turnover, "turnover")
    c = _check_unit_cost(unit_cost)
    td = _nonneg(trading_days, "trading_days")
    if td == 0:
        raise ValueError("trading_days 须为正数")
    if impact_model is None:
        if not impact_params:
            raise ValueError(
                "冲击成本模型没有通用默认参数：请传入 impact_model，或用 impact_params 给出平方根模型的 "
                "coefficient 与 daily_volatility（应以成交记录校准）"
            )
        impact_model = square_root_impact(**impact_params)
    if np.isscalar(adv):
        if weights is not None:
            raise ValueError("adv 为标量时不能给出 weights")
        v = pd.Series([_nonneg(adv, "adv")], index=["整体"])
        w = pd.Series([1.0], index=v.index)
    else:
        if weights is None:
            raise ValueError("adv 按资产给出时须同时给出 weights")
        v = _asset_series(adv, "adv")
        w = _asset_series(weights, "weights")
        if set(v.index) != set(w.index):
            raise ValueError(f"adv 与 weights 的资产不一致：{list(v.index)} 与 {list(w.index)}")
        w = w.reindex(v.index)
        if v.isna().any() or w.isna().any():
            raise ValueError("adv 或 weights 有缺失值，不能按零处理")
        if (w < 0).any() or abs(float(w.sum()) - 1) > 1e-6:
            raise ValueError(f"weights 须非负且和为 1，当前和为 {float(w.sum()):.6f}")
    if (v <= 0).any():
        raise ValueError("日均成交额须为正数")
    rows = []
    for aum in _values(aum_grid, "aum_grid"):
        if aum < 0:
            raise ValueError(f"规模不能为负，收到 {aum}")
        daily = 2 * to * aum * w.to_numpy() / td
        q = daily / v.to_numpy()
        impact = np.asarray(impact_model(q), dtype=float)
        impact_rate = 2 * to * float(np.sum(w.to_numpy() * impact))
        linear = 2 * to * c
        row = {
            "规模": float(aum),
            "年成交额": 2 * to * aum,
            "最大参与率": float(q.max()),
            "线性成本率": linear,
            "冲击成本率": impact_rate,
            "总成本率": linear + impact_rate,
        }
        if max_participation is not None:
            row["超过参与率上限"] = "是" if q.max() > max_participation else "否"
        rows.append(row)
    return pd.DataFrame(rows)
