# akshare 测试样本

按 akshare 1.18.97 各接口源码中的返回列名与单位构造的小样本，供 `tests/test_akshare.py` 用 monkeypatch 替换接口。
数值为虚构，不代表真实基金或指数；列名、日期格式与百分数单位与接口一致。

| 文件 | 接口与参数 |
| --- | --- |
| `fund_open_fund_info_em_110011_单位净值走势.csv` | `fund_open_fund_info_em(symbol="110011", indicator="单位净值走势")`：净值日期、单位净值、日增长率（%）。2024-01-15 为除息日；2024-02-20 的日增长率人为多记 0.30 个百分点，用于交叉核对 |
| `fund_open_fund_info_em_110011_累计净值走势.csv` | `indicator="累计净值走势"`：净值日期、累计净值（单位净值 + 累计分红） |
| `fund_open_fund_info_em_110011_分红送配详情.csv` | `indicator="分红送配详情"`：年份、权益登记日、除息日、每份分红、分红发放日 |
| `index_zh_a_hist_000300.csv` | `index_zh_a_hist(symbol="000300", period="daily", ...)` |
| `stock_zh_index_hist_csindex_H11001.csv` | `stock_zh_index_hist_csindex(symbol="H11001", ...)` |
| `stock_zh_index_hist_csindex_H00300.csv` | `stock_zh_index_hist_csindex(symbol="H00300", ...)`：沪深 300 全收益，在价格指数基础上每个交易日多约 0.8 个基点（模拟分红再投资） |
| `stock_zh_index_hist_csindex_000300.csv` | `stock_zh_index_hist_csindex(symbol="000300", ...)`：收盘价与 `index_zh_a_hist_000300.csv` 相同，供东方财富失败时改用中证官网的测试 |
| `bond_composite_index_cbond_财富_总值.csv` | `bond_composite_index_cbond(indicator="财富", period="总值")` |
| `rate_interbank_Shibor人民币_3月.csv` | `rate_interbank(market="上海银行同业拆借市场", symbol="Shibor人民币", indicator="3月")` |
| `bond_zh_us_rate.csv` | `bond_zh_us_rate(start_date=...)` |

`tests/fake_akshare.py` 的 `FakeAkshare(fail={接口名: 异常})` 可让指定接口抛出网络异常（如 `ConnectionError`、`requests.exceptions.ChunkedEncodingError`），用于测试重试、数据源自动切换与旧缓存回退。
