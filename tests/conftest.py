"""共享夹具：第九部分 12 个月演示数据（日期为演示用月末，正文只给出月序号）。"""

from pathlib import Path

import pytest

from fundeval.etl.sources import files

DATA = Path(__file__).parent / "data"

COLUMNS = {"月份": "date", "组合收益": "portfolio", "基准收益": "benchmark", "无风险收益": "risk_free"}


@pytest.fixture
def worked_example():
    return files.load_returns(DATA / "worked_example.csv", columns=COLUMNS)
