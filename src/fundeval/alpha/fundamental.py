"""用 IC 和有效广度理解优势来源：正文第四部分第 1 节。

主动管理基本定律 IR ≈ TC × IC × √BR：IC 为预测与随后实现收益的相关性，
BR 为同一评价期内有效独立投资决策的数量，TC 为在约束下把预测转化为头寸的效率。
"""

from __future__ import annotations

import numpy as np


def expected_ir(ic, breadth, transfer_coefficient=1.0):
    """理想化预期信息比率 IR ≈ TC × IC × √BR。

    正文例：IC = 0.05、年度有效广度 BR = 100 时 IR ≈ 0.50；TC = 0.70 时 IR ≈ 0.35。
    IR 的频率与 BR 的统计期一致（BR 按年计则为年化 IR）。

    该关系依赖模型假设，用于解释优势来源（预测质量、有效广度、转化效率），
    不能代替实测 IR。高换手不自动等于高 BR：重复交易高度相关的股票或因子，
    可能增加成本却没有增加有效信息。

    参数可为标量或数组；BR 为负时报错。
    """
    br = np.asarray(breadth, dtype=float)
    if np.any(br < 0):
        raise ValueError("有效广度 BR 不能为负")
    out = np.asarray(transfer_coefficient, dtype=float) * np.asarray(ic, dtype=float) * np.sqrt(br)
    return float(out) if out.ndim == 0 else out
