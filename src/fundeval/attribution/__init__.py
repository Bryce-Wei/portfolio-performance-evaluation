"""风格分析与收益归因：正文第五部分。

- ``brinson``：第 2 节，单期 BHB / BF 归因与多期 Cariño 链接
- ``timing``：第 5 节，Treynor–Mazuy 与 Henriksson–Merton 择时回归

风格分析（第 1 节）、多因子分解（第 3 节）与 Campisi 固收归因（第 4 节）待实现。
"""

from fundeval.attribution import brinson, timing
from fundeval.attribution.brinson import BrinsonResult, MultiPeriodBrinsonResult, brinson_multi_period, brinson_single
from fundeval.attribution.timing import TimingResult, henriksson_merton, treynor_mazuy

__all__ = [
    "brinson",
    "timing",
    "BrinsonResult",
    "MultiPeriodBrinsonResult",
    "brinson_single",
    "brinson_multi_period",
    "TimingResult",
    "treynor_mazuy",
    "henriksson_merton",
]
