"""下行风险与尾部风险：正文第八部分。

- Sortino：以下行偏差代替波动率，只惩罚低于 MAR 的收益
- Calmar：年化几何收益与最大回撤之比
- 历史模拟 VaR 与 ES：损失 L = -r，以正数报告

口径：下行偏差分母为全样本期数 n，高于 MAR 的偏差按零计入；Sortino 的分子为
同频算术均值，乘以 √K 年化。比率分母为零时返回 NaN，不报告无穷大。
"""

from __future__ import annotations

import math

import numpy as np

from fundeval._utils import ZERO_TOL, as_series, broadcast_like, safe_div
from fundeval.returns import annualized_return
from fundeval.risk import max_drawdown


def _below_mar(returns, mar):
    r = as_series(returns)
    return r - broadcast_like(mar, r)


def downside_deviation(returns, mar=0.0) -> float:
    """每期下行偏差 DD = sqrt( Σ[min(r_t - MAR_t, 0)]² / n )。

    MAR 为同频最低可接受收益，须事前指定，可为常数或同期序列。
    分母是全样本期数 n，而不是低于 MAR 的期数。
    """
    d = _below_mar(returns, mar)
    if len(d) == 0:
        return float("nan")
    shortfall = np.minimum(d.to_numpy(), 0.0)
    return float(np.sqrt(np.sum(shortfall**2) / len(d)))


def sortino_ratio(returns, mar=0.0, periods_per_year: int = 12) -> float:
    """年化 Sortino = mean(r - MAR) / DD × √K。样本内没有低于 MAR 的收益时返回 NaN。"""
    d = _below_mar(returns, mar)
    return safe_div(float(d.mean()), downside_deviation(returns, mar)) * np.sqrt(periods_per_year)


def calmar_ratio(returns, periods_per_year: int = 12) -> float:
    """Calmar = 年化几何收益 / 最大回撤（同一观察区间，回撤取正值）。

    样本内回撤为零时返回 NaN，不报告无穷大排名。
    """
    r = as_series(returns)
    return safe_div(annualized_return(r, periods_per_year), max_drawdown(r))


def _sorted_losses(returns) -> np.ndarray:
    r = as_series(returns).dropna()
    if len(r) == 0:
        raise ValueError("收益序列为空")
    return np.sort(-r.to_numpy())  # 损失从小到大


def _check_confidence(confidence: float) -> None:
    if not 0 < confidence < 1:
        raise ValueError("置信水平须在 (0, 1) 之间")


def _tail_size(confidence: float, n: int) -> float:
    # (1 - 0.95) × 100 在浮点下为 5.000000000000004，先按 1e-9 取整避免多算一个秩
    m = (1 - confidence) * n
    return round(m) if abs(m - round(m)) < 1e-9 else m


def historical_var(returns, confidence: float = 0.95) -> float:
    """历史模拟 VaR：损失 L = -r 的 c 分位数，按最近秩法取第 ⌈c·n⌉ 小的损失，以正数报告。

    只对历史净值收益排序时，结果应标为历史组合经验风险，不等于当前持仓的完整风险。
    """
    _check_confidence(confidence)
    losses = _sorted_losses(returns)
    n = len(losses)
    rank = math.ceil(n - _tail_size(confidence, n))  # ⌈c·n⌉，与尾部比例使用同一取整
    return float(losses[max(rank, 1) - 1])


def historical_es(returns, confidence: float = 0.95) -> float:
    """历史模拟 ES：最差 (1 - c) 比例结果的平均损失，以正数报告。

    尾部比例 m = (1 - c)·n 不是整数时，最差的 ⌊m⌋ 个损失各计权重 1，下一个损失
    计权重 m - ⌊m⌋，再除以 m；分母始终是尾部比例 m，不随并列或取整更换。
    m < 1 时 ES 等于最大损失。
    """
    _check_confidence(confidence)
    losses = _sorted_losses(returns)[::-1]  # 损失从大到小
    n = len(losses)
    m = _tail_size(confidence, n)
    full = int(math.floor(m + ZERO_TOL))
    frac = m - full
    total = float(np.sum(losses[:full]))
    if frac > ZERO_TOL:
        total += frac * float(losses[full])
    return total / m
