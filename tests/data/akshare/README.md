# akshare 测试样本

供 `tests/test_akshare.py` 等用 monkeypatch 替换接口的小样本。列名、日期格式与百分数单位与接口一致，但数值大多为构造：

- **来自真实返回（akshare 1.18.97，本地核实，2026-09-27）**：`分红送配详情` 的列名（年份、权益登记日、除息日、每10份分红、分红发放日）与金额写法（“每10份派现金9.0000元”，按每 10 份计）；110011 在 2021-02-25 除息、每10份派现金9.0000元；单位净值 2021-02-24 为 9.4782、2021-02-25 为 8.4677，2021-02-25 的日增长率为 −1.17%。
- **构造**：其余所有数值（包括 2021 年分红的权益登记日与发放日、2023-11 起的净值、指数与利率），以及其他接口的列名。这些列名按 akshare 1.18.97 源码填写，未在本仓库用真实返回核对。

| 文件 | 接口与参数 |
| --- | --- |
| `fund_open_fund_info_em_110011_单位净值走势.csv` | `fund_open_fund_info_em(symbol="110011", indicator="单位净值走势")`：净值日期、单位净值、日增长率（%）。开头两行为 2021-02-24/25 的真实数据（2021-02-24 的日增长率留空）；其后为构造数据，2024-01-15 为除息日，2024-02-20 的日增长率人为多记 0.30 个百分点，用于交叉核对 |
| `fund_open_fund_info_em_110011_累计净值走势.csv` | `indicator="累计净值走势"`：净值日期、累计净值（单位净值 + 累计分红） |
| `fund_open_fund_info_em_110011_分红送配详情.csv` | `indicator="分红送配详情"`：年份、权益登记日、除息日、每10份分红、分红发放日（真实格式）。2021 年一行的除息日与金额为真实数据，2024 年一行为构造 |
| `index_zh_a_hist_000300.csv` | `index_zh_a_hist(symbol="000300", period="daily", ...)` |
| `stock_zh_index_hist_csindex_H11001.csv` | `stock_zh_index_hist_csindex(symbol="H11001", ...)` |
| `stock_zh_index_hist_csindex_H00300.csv` | `stock_zh_index_hist_csindex(symbol="H00300", ...)`：沪深 300 全收益，在价格指数基础上每个交易日多约 0.8 个基点（模拟分红再投资） |
| `stock_zh_index_hist_csindex_H00918.csv`、`_H00919.csv`、`_H00905.csv`、`_H00852.csv` | `stock_zh_index_hist_csindex(symbol=...)`：沪深300成长 / 价值、中证500、中证1000 全收益，供风格分析（`--style cn_equity`）的测试。列名与 H00300 相同，收盘价由 H00300 的日收益乘以不同系数、加固定种子的随机扰动构造，不是真实行情 |
| `stock_zh_index_hist_csindex_000300.csv` | `stock_zh_index_hist_csindex(symbol="000300", ...)`：收盘价与 `index_zh_a_hist_000300.csv` 相同，供东方财富失败时改用中证官网的测试 |
| `bond_composite_index_cbond_财富_总值.csv` | `bond_composite_index_cbond(indicator="财富", period="总值")` |
| `rate_interbank_Shibor人民币_3月.csv` | `rate_interbank(market="上海银行同业拆借市场", symbol="Shibor人民币", indicator="3月")` |
| `bond_zh_us_rate.csv` | `bond_zh_us_rate(start_date=...)` |

`tests/fake_akshare.py` 的 `FakeAkshare(fail={接口名: 异常})` 可让指定接口抛出网络异常（如 `ConnectionError`、`requests.exceptions.ChunkedEncodingError`），用于测试重试、数据源自动切换与旧缓存回退；`hang={接口名: 秒数}` 模拟请求挂起不返回，用于测试超时。

## 联网测试的判定规则

`pytest -m network` 的冒烟测试对“数据源不可达”与“数据有误”区别对待（见 `tests/test_akshare.py` 的 `skip_if_unreachable`）：

- 连接类或超时类异常（`requests.ConnectionError` 及其子类、`requests.Timeout`、内置 `ConnectionError`、`UpstreamTimeout`）时 `pytest.skip`，并在原因中写明数据源与异常；东方财富指数、Shibor 与中债三项测试采用这一规则。
- 列名、格式或数值错误（如 `AkshareInterfaceError`、`KeyError`、断言失败）仍判失败，不被 skip 掩盖。
