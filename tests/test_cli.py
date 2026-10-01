"""命令行 fundeval report：本地文件不联网出报告；--fund 路径用替换后的 akshare 接口。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fundeval import cli
from fundeval.etl.sources import akshare as aks
from fake_akshare import FakeAkshare

DATA = Path(__file__).parent / "data"
COLUMNS = "月份=date,组合收益=portfolio,基准收益=benchmark,无风险收益=risk_free"


def test_parse_benchmark_spec():
    assert cli.parse_benchmark_spec("000300:0.8,H11001:0.2") == [("000300", 0.8), ("H11001", 0.2)]
    assert cli.parse_benchmark_spec("000905") == [("000905", 1.0)]
    assert cli.parse_benchmark_spec("沪深300指数收益率*80%+中债综合指数收益率*20%") == [
        ("000300", 0.8),
        ("cbond:composite", 0.2),
    ]
    with pytest.raises(ValueError):
        cli.parse_benchmark_spec("000300:,H11001:0.2")


def test_report_from_local_csv_markdown(tmp_path, capsys):
    out = tmp_path / "report.md"
    code = cli.main(["report", "--input", str(DATA / "worked_example.csv"), "--columns", COLUMNS, "--out", str(out)])
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "| 累计收益（组合） | 10.2058% |" in text and "| Sharpe 比率 | 1.1254 |" in text
    assert "输入文件的 risk_free 列" in text
    assert "样本较短" in text and "具备管理能力" not in text


def test_report_from_local_csv_excel_and_stdout(tmp_path, capsys):
    pytest.importorskip("openpyxl")
    out = tmp_path / "report.xlsx"
    assert cli.main(["report", "--input", str(DATA / "worked_example.csv"), "--columns", COLUMNS, "--out", str(out)]) == 0
    assert "指标" in pd.read_excel(out, sheet_name=None)
    assert cli.main(["report", "--input", str(DATA / "worked_example.csv"), "--columns", COLUMNS]) == 0
    assert capsys.readouterr().out.startswith("# worked_example 绩效评价报告")


def test_report_from_daily_csv_resampled(tmp_path):
    idx = pd.bdate_range("2024-01-01", "2024-12-31")
    rng = np.random.default_rng(2)
    bench = rng.normal(0.0003, 0.008, len(idx))
    df = pd.DataFrame({"date": idx, "portfolio": bench + rng.normal(0.0001, 0.002, len(idx)), "benchmark": bench})
    path = tmp_path / "daily.csv"
    df.to_csv(path, index=False)
    out = tmp_path / "r.md"
    code = cli.main(["report", "--input", str(path), "--input-freq", "D", "--freq", "M", "--rf", "0.018", "--out", str(out)])
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert "| 期数 | 12 |" in text and "常数年化 1.80%" in text


def test_report_errors_are_reported(tmp_path, capsys):
    code = cli.main(["report", "--input", str(DATA / "worked_example.csv"), "--columns", COLUMNS, "--rf", "0.02"])
    assert code == 2 and "risk_free" in capsys.readouterr().err
    code = cli.main(["report", "--input", str(DATA / "worked_example.csv"), "--columns", COLUMNS, "--out", str(tmp_path / "x.pdf")])
    assert code == 2


def test_report_from_akshare_with_fake_interfaces(monkeypatch, tmp_path, capsys):
    fake = FakeAkshare()
    monkeypatch.setattr(aks, "_ak", lambda: fake)
    out = tmp_path / "fund.md"
    code = cli.main([
        "report", "--fund", "110011", "--benchmark", "000300:0.8,H11001:0.2",
        "--start", "2024-01-01", "--end", "2024-02-29", "--freq", "W", "--rf", "shibor3m",
        "--cache-dir", str(tmp_path / "cache"), "--out", str(out),
    ])
    assert code == 0
    assert any(line.startswith("警告：") and "日增长率" in line for line in capsys.readouterr().err.splitlines())
    text = out.read_text(encoding="utf-8")
    assert "基金 110011" in text and "000300:0.8,H11001:0.2" in text
    assert "Shibor 3M" in text and "(1 + y)^(1/52) − 1" in text
    # 数据源日增长率与净值推算不一致的日期进入数据质量报告
    assert "2024-02-20" in text and "基点" in text
    assert "样本较短" in text and "具备管理能力" not in text


def test_cli_use_t_flag(tmp_path):
    out = tmp_path / "r.md"
    args = ["report", "--input", str(DATA / "worked_example.csv"), "--columns", COLUMNS, "--hac-lags", "2", "--out", str(out)]
    assert cli.main(args) == 0
    assert "HAC（Newey–West，滞后 2，正态近似）" in out.read_text(encoding="utf-8")
    assert cli.main([*args, "--use-t"]) == 0
    text = out.read_text(encoding="utf-8")
    assert "HAC（Newey–West，滞后 2，t 分布）" in text and "正态近似" not in text
