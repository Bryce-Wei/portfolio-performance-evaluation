"""Campisi 固定收益归因：正文第五部分第 4 节。

把债券组合的期间收益拆成四部分：

    R = R_income + R_rate + R_spread + R_specific

- R_income：期间票息与应计收入（调用方给出，小数）
- R_rate ≈ −D·Δy + ½·C·Δy²（平行移动近似）；关键期限久期版本 R_rate ≈ −Σ KRD_j·Δy_j（凸性项可选）
- R_spread ≈ −D_spread·Δs
- R_specific = R − (R_income + R_rate + R_spread)，未给出总收益时无法计算

Δy、Δs 一律为小数（0.0010 表示上升 10 个基点）。输入疑似百分数或基点（|Δy| > 0.2）时报错。

使用限定：

- 平行移动近似不适合所有债券：久期、凸性只是收益率小幅变动下的局部近似；
- 曲线扭曲（陡峭化、平坦化、蝶式）应使用关键期限久期，按期限分别给出 KRD_j 与 Δy_j；
- 含权债券（可赎回、可回售、MBS 等）须使用有效久期与有效凸性，修正久期会误判利率敏感性；
- 剩余项 R_specific 包含个券选择、流动性、估值差异、违约与近似误差等，不能全部视为选券能力。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: |Δy|、|Δs| 超过该值视为单位错误（小数口径下 0.2 即 2000 个基点）
MAX_DECIMAL_CHANGE = 0.2
#: 收入与总收益的绝对值超过该值视为单位错误（单期收益 50% 以上的债券组合不现实）
MAX_DECIMAL_RETURN = 0.5
#: 逐项相减后与主动收益对账的容差
RECONCILE_TOL = 1e-12

COMPONENTS = ("收入", "利率", "利差", "剩余")

CAVEAT = (
    "平行移动近似不适合所有债券；曲线扭曲应使用关键期限久期；含权债券需使用有效久期；"
    "剩余项包含个券选择、流动性、估值差异、违约与近似误差，不能全部视为选券能力"
)


def _finite(value, name: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 须为数值，收到 {value!r}") from exc
    if not np.isfinite(v):
        raise ValueError(f"{name} 缺失或非有限值")
    return v


def _check_change(value: float, name: str) -> None:
    if abs(value) > MAX_DECIMAL_CHANGE:
        raise ValueError(
            f"{name} = {value} 疑似以百分数或基点输入；请用小数，如上升 10 个基点写 0.0010"
        )


def _check_return(value: float, name: str) -> None:
    if abs(value) > MAX_DECIMAL_RETURN:
        raise ValueError(f"{name} = {value} 疑似以百分数输入；请用小数，如 1.20% 写 0.012")


def _tenor_series(value, name: str) -> pd.Series:
    s = pd.Series(value, dtype=float)
    if s.isna().any():
        raise ValueError(f"{name} 有缺失的期限：{list(s.index[s.isna()])}")
    if s.index.duplicated().any():
        raise ValueError(f"{name} 的期限重复")
    return s


@dataclass(frozen=True)
class CampisiResult:
    """单个组合的 Campisi 归因（单期，小数）。

    - ``income``、``rate``、``spread``：收入、利率、利差效应
    - ``specific``：剩余项；未给出 ``total_return`` 时为 None
    - ``total_return``：调用方给出的期间总收益
    - ``rate_by_tenor``：关键期限久期版本下各期限的利率效应（含凸性项），平行移动时为 None
    """

    income: float
    rate: float
    spread: float
    specific: float | None
    total_return: float | None
    rate_by_tenor: pd.Series | None = None

    @property
    def explained(self) -> float:
        """收入、利率、利差三项之和。"""
        return self.income + self.rate + self.spread

    @property
    def components(self) -> pd.Series:
        """收入、利率、利差、剩余四项（剩余项未知时为 NaN）。"""
        return pd.Series(
            [self.income, self.rate, self.spread, np.nan if self.specific is None else self.specific],
            index=list(COMPONENTS),
            name="贡献",
        )

    def table(self) -> pd.DataFrame:
        """分解表：项目、贡献（小数），最后一行为总收益。"""
        comps = self.components
        rows = list(zip(comps.index, comps.to_numpy()))
        rows.append(("总收益", np.nan if self.total_return is None else self.total_return))
        return pd.DataFrame(rows, columns=["项目", "贡献"])


def campisi(
    income,
    duration,
    delta_yield,
    convexity=0.0,
    spread_duration=0.0,
    delta_spread=0.0,
    total_return=None,
) -> CampisiResult:
    """单期 Campisi 归因（正文第五部分第 4 节）。

    参数（均为小数）
    ----------------
    income : 期间票息与应计收入，如 0.006 表示 0.60%
    duration : 平行移动时为（有效）久期 D；关键期限版本为按期限索引的 Series（KRD_j）
    delta_yield : 平行移动时为无风险收益率变动 Δy；关键期限版本为与 ``duration`` 同索引的 Series（Δy_j）
    convexity : 平行移动时为凸性 C，R_rate ≈ −D·Δy + ½·C·Δy²；关键期限版本可为同索引的 Series，
        凸性项为 ½·Σ C_j·Δy_j²（缺省 0，不计凸性）
    spread_duration : 利差久期 D_spread
    delta_spread : 利差变动 Δs，R_spread ≈ −D_spread·Δs
    total_return : 期间总收益；给出时 R_specific = total_return − (R_income + R_rate + R_spread)，否则为 None

    Δy、Δs 用小数（0.0010 表示上升 10 个基点），绝对值大于 0.2 时视为百分数或基点输入而报错；
    收入与总收益的绝对值大于 0.5 时同样报错。关键期限久期与 Δy 的期限须一一对应。

    平行移动近似不适合所有债券；曲线扭曲应使用关键期限久期；含权债券需使用有效久期；剩余项包含
    个券选择、流动性、估值差异、违约与近似误差，不能全部视为选券能力。
    """
    inc = _finite(income, "income")
    _check_return(inc, "income")
    krd = isinstance(duration, (pd.Series, Mapping)) or isinstance(delta_yield, (pd.Series, Mapping))
    rate_by_tenor = None
    if krd:
        if not (isinstance(duration, (pd.Series, Mapping)) and isinstance(delta_yield, (pd.Series, Mapping))):
            raise ValueError("关键期限久期版本中 duration 与 delta_yield 须同为按期限索引的 Series")
        d = _tenor_series(duration, "duration")
        dy = _tenor_series(delta_yield, "delta_yield")
        if set(d.index) != set(dy.index):
            raise ValueError(
                f"关键期限不一致：久期有 {list(d.index)}，收益率变动有 {list(dy.index)}"
            )
        dy = dy.reindex(d.index)
        for tenor, v in dy.items():
            _check_change(float(v), f"delta_yield[{tenor}]")
        if isinstance(convexity, (pd.Series, Mapping)):
            c = _tenor_series(convexity, "convexity")
            if set(c.index) != set(d.index):
                raise ValueError(f"凸性的期限 {list(c.index)} 与久期 {list(d.index)} 不一致")
            c = c.reindex(d.index)
        else:
            c0 = _finite(convexity, "convexity")
            if c0 != 0.0:
                raise ValueError("关键期限久期版本的凸性须按期限给出（Series），或省略")
            c = pd.Series(0.0, index=d.index)
        rate_by_tenor = (-d * dy + 0.5 * c * dy**2).rename("利率效应")
        rate = float(rate_by_tenor.sum())
    else:
        d = _finite(duration, "duration")
        dy = _finite(delta_yield, "delta_yield")
        _check_change(dy, "delta_yield")
        c = _finite(convexity, "convexity")
        rate = -d * dy + 0.5 * c * dy**2
    sd = _finite(spread_duration, "spread_duration")
    ds = _finite(delta_spread, "delta_spread")
    _check_change(ds, "delta_spread")
    spread = -sd * ds
    specific = total = None
    if total_return is not None:
        total = _finite(total_return, "total_return")
        _check_return(total, "total_return")
        specific = total - (inc + rate + spread)
    return CampisiResult(
        income=inc, rate=float(rate), spread=float(spread), specific=specific, total_return=total,
        rate_by_tenor=rate_by_tenor,
    )


@dataclass(frozen=True)
class ActiveCampisi:
    """基金相对基准的主动 Campisi 归因：逐项相减。

    - ``fund``、``benchmark``：两者各自的归因
    - ``active``：收入、利率、利差、剩余四项的主动贡献（基金 − 基准）
    - ``active_return``：主动收益 = 基金总收益 − 基准总收益（算术差）
    """

    fund: CampisiResult
    benchmark: CampisiResult
    active: pd.Series
    active_return: float

    def table(self) -> pd.DataFrame:
        """对比表：项目、基金、基准、主动（小数），最后一行为总收益。"""
        rows = [
            (name, float(self.fund.components[name]), float(self.benchmark.components[name]), float(self.active[name]))
            for name in COMPONENTS
        ]
        rows.append(("总收益", self.fund.total_return, self.benchmark.total_return, self.active_return))
        return pd.DataFrame(rows, columns=["项目", "基金", "基准", "主动"])


def _as_result(x, name: str) -> CampisiResult:
    if isinstance(x, CampisiResult):
        return x
    if isinstance(x, Mapping):
        return campisi(**x)
    raise TypeError(f"{name} 须为 CampisiResult 或 campisi 的参数字典")


def campisi_active(fund, benchmark) -> ActiveCampisi:
    """对基金与基准用同一方法分别做 Campisi 归因，再逐项相减得到主动贡献，并与主动收益对账。

    ``fund``、``benchmark`` 为 campisi 的结果或其参数字典，两者都须给出 ``total_return``（否则剩余项
    未知，无法对账而报错）。两者须同为平行移动或同为关键期限版本，避免方法不一致。
    主动收益为算术差 R_fund − R_benchmark；四项主动贡献之和与之不一致时报错。
    """
    f = _as_result(fund, "fund")
    b = _as_result(benchmark, "benchmark")
    for res, name in ((f, "基金"), (b, "基准")):
        if res.total_return is None:
            raise ValueError(f"{name}未给出 total_return，剩余项未知，无法与主动收益对账")
    if (f.rate_by_tenor is None) != (b.rate_by_tenor is None):
        raise ValueError("基金与基准须使用同一方法：同为平行移动近似，或同为关键期限久期")
    active = (f.components - b.components).rename("主动贡献")
    active_return = f.total_return - b.total_return
    if abs(float(active.sum()) - active_return) > RECONCILE_TOL:
        raise ValueError(
            f"主动贡献之和 {float(active.sum()):.10f} 与主动收益 {active_return:.10f} 不一致，无法对账"
        )
    return ActiveCampisi(fund=f, benchmark=b, active=active, active_return=active_return)
