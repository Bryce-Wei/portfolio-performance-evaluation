"""数据源。

- ``files``：本地 CSV / Excel
- ``akshare``：国内公募基金、指数与利率。akshare 为可选依赖，本包不在导入时加载它，
  使用时 ``from fundeval.etl.sources import akshare``

French 因子库在后续步骤加入。
"""

from fundeval.etl.sources import files

__all__ = ["files"]
