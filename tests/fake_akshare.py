"""测试用的 akshare 替身：按接口签名返回 tests/data/akshare/ 中的样本，并记录调用。"""

import types
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data" / "akshare"


class FakeAkshare(types.SimpleNamespace):
    """按接口签名返回样本，并记录调用次数。"""

    def __init__(self):
        super().__init__(__version__="fake", calls=[])

    def _log(self, name, **kwargs):
        self.calls.append((name, kwargs))

    def fund_open_fund_info_em(self, symbol="710001", indicator="单位净值走势", period="成立来"):
        self._log("fund_open_fund_info_em", symbol=symbol, indicator=indicator)
        return pd.read_csv(DATA / f"fund_open_fund_info_em_{symbol}_{indicator}.csv")

    def index_zh_a_hist(self, symbol="000859", period="daily", start_date="19700101", end_date="22220101"):
        self._log("index_zh_a_hist", symbol=symbol, start_date=start_date, end_date=end_date)
        return pd.read_csv(DATA / f"index_zh_a_hist_{symbol}.csv", dtype={"日期": str})

    def stock_zh_index_hist_csindex(self, symbol="000928", start_date="20180526", end_date="20240604"):
        self._log("stock_zh_index_hist_csindex", symbol=symbol)
        return pd.read_csv(DATA / f"stock_zh_index_hist_csindex_{symbol}.csv")

    def bond_composite_index_cbond(self, indicator="财富", period="总值"):
        self._log("bond_composite_index_cbond", indicator=indicator, period=period)
        return pd.read_csv(DATA / f"bond_composite_index_cbond_{indicator}_{period}.csv")

    def rate_interbank(self, market="上海银行同业拆借市场", symbol="Shibor人民币", indicator="隔夜"):
        self._log("rate_interbank", market=market, symbol=symbol, indicator=indicator)
        return pd.read_csv(DATA / f"rate_interbank_{symbol}_{indicator}.csv")

    def bond_zh_us_rate(self, start_date="19901219"):
        self._log("bond_zh_us_rate", start_date=start_date)
        return pd.read_csv(DATA / "bond_zh_us_rate.csv")

    def count(self, name):
        return sum(1 for n, _ in self.calls if n == name)
