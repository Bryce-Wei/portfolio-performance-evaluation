"""数据源。

- ``files``：本地 CSV / Excel
- ``akshare``：国内公募基金、指数与利率。akshare 为可选依赖，本包不在导入时加载它，
  使用时 ``from fundeval.etl.sources import akshare``

- ``french``：Kenneth R. French 因子库（美国市场 Fama–French 三因子与动量因子，美元计价），
  ``from fundeval.etl.sources import french``；下载用标准库，不依赖 akshare
"""

from fundeval.etl.sources import files

__all__ = ["files"]
