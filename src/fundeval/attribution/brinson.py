"""Brinson 归因区分配置与选择：正文第五部分第 2 节。

单期 BHB（Brinson–Hood–Beebower）三项分解：

    A_i = (w_p,i - w_b,i)·r_b,i
    S_i = w_b,i·(r_p,i - r_b,i)
    I_i = (w_p,i - w_b,i)(r_p,i - r_b,i)
    r_p - r_b = Σ_i (A_i + S_i + I_i)

BF（Brinson–Fachler）版本的配置项为 (w_p,i - w_b,i)(r_b,i - R_b)，选择与交互不变；
完整权重下总配置贡献与 BHB 一致，但行业分配不同，报告须标明版本。

权重须使用与收益区间一致的期初权重，两组权重各自加总为 1，现金也应作为一个类别纳入。
跨期归因不能直接把各期贡献相加，见 brinson_multi_period 的 Cariño 链接。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fundeval._utils import ZERO_TOL

METHODS = ("BHB", "BF")
EFFECTS = ["allocation", "selection", "interaction"]
COLUMNS = EFFECTS + ["total"]
TOTAL = "total"

#: 对账残差超过该值时发出警告（权重在容差内偏离 1 时，BF 会产生 R_b × 权重差 的残差）
RECONCILE_TOL = 1e-10


@dataclass(frozen=True)
class BrinsonResult:
    """单期 Brinson 结果。

    ``effects``：各行业的 allocation、selection、interaction 与 total；
    ``totals``：四项合计；``portfolio_return`` 与 ``benchmark_return`` 为按权重加权的
    R_p = Σ w_p,i r_p,i、R_b = Σ w_b,i r_b,i；``residual`` = Σ(A+S+I) - (R_p - R_b)，
    ``reconciled`` 表示残差是否在 1e-10 以内。
    """

    effects: pd.DataFrame
    totals: pd.Series
    portfolio_return: float
    benchmark_return: float
    active_return: float
    residual: float
    reconciled: bool
    method: str

    def table(self) -> pd.DataFrame:
        """各行业贡献加一行 ``total``，对应正文的归因表。"""
        return pd.concat([self.effects, self.totals.to_frame(TOTAL).T])


def _check_method(method: str) -> str:
    m = str(method).upper()
    if m not in METHODS:
        raise ValueError(f"method 须为 {METHODS} 之一，收到 {method!r}")
    return m


def _to_series(values, name: str, index=None) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.astype(float).rename(name)
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} 须为一维数据")
    return pd.Series(arr, index=index if index is not None else range(len(arr)), name=name)


def _single_inputs(wp, wb, rp, rb) -> pd.DataFrame:
    """把四组输入对齐为以行业为索引的 DataFrame。Series 按行业名对齐，数组按位置配对。"""
    first = next((x for x in (wp, wb, rp, rb) if isinstance(x, pd.Series)), None)
    base = first.index if first is not None else None
    parts = {}
    for name, v in zip(("wp", "wb", "rp", "rb"), (wp, wb, rp, rb)):
        s = _to_series(v, name, base)
        if base is None:
            base = s.index
        if len(s) != len(base) or set(s.index) != set(base):
            raise ValueError(f"{name} 的行业与其他输入不一致")
        if s.index.duplicated().any():
            raise ValueError(f"{name} 的行业名重复")
        parts[name] = s.reindex(base)
    df = pd.DataFrame(parts)
    if df.isna().any().any():
        raise ValueError("权重或收益含缺失值；请先补全数据，不能按零处理")
    return df


def _check_weights(w: pd.Series, label: str, tol: float) -> None:
    total = float(w.sum())
    if abs(total - 1.0) > tol:
        raise ValueError(
            f"{label}权重合计为 {total:.6g}，不等于 1（容差 {tol:g}）；"
            "现金、衍生品保证金等也需作为单独类别纳入分类，使权重完整"
        )


def _effects(df: pd.DataFrame, method: str) -> tuple[pd.DataFrame, float, float]:
    wp, wb, rp, rb = df["wp"], df["wb"], df["rp"], df["rb"]
    r_p = float((wp * rp).sum())
    r_b = float((wb * rb).sum())
    active_w = wp - wb
    if method == "BHB":
        allocation = active_w * rb
    else:
        allocation = active_w * (rb - r_b)
    out = pd.DataFrame(
        {
            "allocation": allocation,
            "selection": wb * (rp - rb),
            "interaction": active_w * (rp - rb),
        }
    )
    out["total"] = out[EFFECTS].sum(axis=1)
    out.index.name = "sector"
    return out, r_p, r_b


def brinson_single(wp, wb, rp, rb, method: str = "BHB", tol: float = 1e-6) -> BrinsonResult:
    """单期 Brinson 归因，按行业输出配置 A_i、选择 S_i、交互 I_i 及合计。

    ``wp``、``wb`` 为组合与基准期初权重，``rp``、``rb`` 为组合与基准的行业收益，
    均为小数；可为以行业为索引的 Series（按行业名对齐），或等长列表 / 数组。
    ``method`` 为 ``"BHB"`` 或 ``"BF"``。

    两组权重须各自加总为 1（偏差不超过 ``tol``），否则报错并提示现金也需纳入分类。
    结果中的 ``residual`` 与 ``reconciled`` 给出 Σ(A+S+I) 与 R_p - R_b 的对账检查；
    残差超过 1e-10 时发出警告。
    """
    method = _check_method(method)
    df = _single_inputs(wp, wb, rp, rb)
    _check_weights(df["wp"], "组合", tol)
    _check_weights(df["wb"], "基准", tol)
    effects, r_p, r_b = _effects(df, method)
    totals = effects.sum().rename(TOTAL)
    active = r_p - r_b
    residual = float(totals["total"] - active)
    reconciled = abs(residual) <= RECONCILE_TOL
    if not reconciled:
        warnings.warn(f"Brinson 对账残差 {residual:.3g}：Σ(A+S+I) 与 R_p - R_b 不一致", RuntimeWarning, stacklevel=2)
    return BrinsonResult(effects, totals, r_p, r_b, active, residual, reconciled, method)


def _carino_coefficient(r_p, r_b):
    """Cariño 系数 k = [ln(1+R_p) - ln(1+R_b)] / (R_p - R_b)。

    R_p = R_b 时取解析极限 1 / (1 + R_p)，而不是 NaN，使对账恒等式仍然成立。
    """
    r_p = np.asarray(r_p, dtype=float)
    r_b = np.asarray(r_b, dtype=float)
    if np.any(r_p <= -1) or np.any(r_b <= -1):
        raise ValueError("收益不能低于或等于 -100%，无法取对数链接")
    diff = r_p - r_b
    equal = np.abs(diff) < ZERO_TOL
    safe = np.where(equal, 1.0, diff)
    k = np.where(equal, 1.0 / (1.0 + r_p), (np.log1p(r_p) - np.log1p(r_b)) / safe)
    return k


@dataclass(frozen=True)
class MultiPeriodBrinsonResult:
    """多期 Brinson 结果（Cariño 对数链接）。

    - ``periods``：按 (period, sector) 排列的链接后贡献，列为 allocation、selection、
      interaction、total；各行之和等于累计收益差额
    - ``by_period``：每期的链接后四项合计
    - ``by_sector``：每个行业跨期的链接后贡献之和
    - ``totals``：全部链接后贡献的合计
    - ``unlinked``：各期单期归因的原始贡献（未调整，不能直接跨期相加）
    - ``coefficients``：各期系数 k_t；``k`` 为累计系数
    - ``residual``：Σ 链接后贡献 - [(∏(1+R_p,t) - 1) - (∏(1+R_b,t) - 1)]
    """

    periods: pd.DataFrame
    by_period: pd.DataFrame
    by_sector: pd.DataFrame
    totals: pd.Series
    unlinked: pd.DataFrame
    portfolio_returns: pd.Series
    benchmark_returns: pd.Series
    coefficients: pd.Series
    k: float
    cumulative_portfolio: float
    cumulative_benchmark: float
    cumulative_active: float
    residual: float
    reconciled: bool
    method: str


def _frame(values, name: str) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        return values.astype(float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} 须为二维数据（行为期、列为行业）")
    return pd.DataFrame(arr)


def brinson_multi_period(wp, wb, rp, rb, method: str = "BHB", tol: float = 1e-6) -> MultiPeriodBrinsonResult:
    """多期 Brinson 归因，使用 Cariño 对数链接。

    四个输入均为 DataFrame（行为期、列为行业，四者的期与行业须一致）或同形状二维数组。
    每期先做单期归因（权重校验与 brinson_single 相同），再把第 t 期各项贡献乘以
    k_t / k：

        k_t = [ln(1+R_p,t) - ln(1+R_b,t)] / (R_p,t - R_b,t)
        k   = [ln(1+R_p) - ln(1+R_b)] / (R_p - R_b)，R 为 ∏(1+R_t) - 1

    调整后全部贡献之和恰等于累计收益差额 (∏(1+R_p,t) - 1) - (∏(1+R_b,t) - 1)。
    某期（或累计）组合与基准收益相等时，系数取解析极限 1/(1+R)，而不是 NaN，
    对账恒等式仍成立。

    不能直接把各期单期贡献相加：算术和不等于复利后的累计差额（见 ``unlinked``）。
    """
    method = _check_method(method)
    frames = {n: _frame(v, n) for n, v in zip(("wp", "wb", "rp", "rb"), (wp, wb, rp, rb))}
    ref = frames["wp"]
    for n, f in frames.items():
        if f.shape != ref.shape or set(f.index) != set(ref.index) or set(f.columns) != set(ref.columns):
            raise ValueError(f"{n} 的期或行业与组合权重不一致")
        frames[n] = f.reindex(index=ref.index, columns=ref.columns)

    per_period, r_p, r_b = {}, {}, {}
    for t in ref.index:
        try:
            res = brinson_single(*(frames[n].loc[t] for n in ("wp", "wb", "rp", "rb")), method=method, tol=tol)
        except ValueError as exc:
            raise ValueError(f"第 {t} 期：{exc}") from exc
        per_period[t] = res.effects
        r_p[t], r_b[t] = res.portfolio_return, res.benchmark_return

    r_p = pd.Series(r_p, name="portfolio")
    r_b = pd.Series(r_b, name="benchmark")
    unlinked = pd.concat(per_period, names=["period", "sector"])
    cum_p = float(np.prod(1 + r_p.to_numpy()) - 1)
    cum_b = float(np.prod(1 + r_b.to_numpy()) - 1)
    k_t = pd.Series(_carino_coefficient(r_p, r_b), index=r_p.index, name="k_t")
    k = float(_carino_coefficient(cum_p, cum_b))
    scale = (k_t / k).reindex(unlinked.index.get_level_values("period")).to_numpy()
    linked = unlinked.mul(scale, axis=0)

    totals = linked.sum().rename(TOTAL)
    active = cum_p - cum_b
    residual = float(totals["total"] - active)
    return MultiPeriodBrinsonResult(
        periods=linked,
        by_period=linked.groupby(level="period", sort=False).sum(),
        by_sector=linked.groupby(level="sector", sort=False).sum(),
        totals=totals,
        unlinked=unlinked,
        portfolio_returns=r_p,
        benchmark_returns=r_b,
        coefficients=k_t,
        k=k,
        cumulative_portfolio=cum_p,
        cumulative_benchmark=cum_b,
        cumulative_active=active,
        residual=residual,
        reconciled=abs(residual) <= RECONCILE_TOL,
        method=method,
    )
