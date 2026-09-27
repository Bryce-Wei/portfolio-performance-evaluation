"""ETL：读取、校验、清洗与单期收益。"""

import pandas as pd
import pytest

from fundeval.etl import clean, schema
from fundeval.etl import returns as etl_returns
from fundeval.etl.sources import files


def test_load_worked_example_parses_percent_text(worked_example):
    assert list(worked_example.columns) == ["portfolio", "benchmark", "risk_free"]
    assert worked_example.index.name == "date"
    assert len(worked_example) == 12
    assert worked_example.portfolio.iloc[0] == pytest.approx(0.02)
    assert worked_example.risk_free.iloc[0] == pytest.approx(0.0015)


def test_load_returns_numeric_percent_flag(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("date,portfolio\n2025-01-31,2.0\n2025-02-28,-1.0\n")
    df = files.load_returns(p, percent=True)
    assert df.portfolio.tolist() == pytest.approx([0.02, -0.01])


def test_load_returns_from_excel(tmp_path, worked_example):
    pytest.importorskip("openpyxl")
    p = tmp_path / "r.xlsx"
    worked_example.reset_index().to_excel(p, index=False)
    df = files.load_returns(p)
    pd.testing.assert_frame_equal(df, worked_example, check_freq=False)


def test_validate_rejects_unconverted_percent():
    df = pd.DataFrame({"portfolio": [2.0, -150.0]}, index=pd.to_datetime(["2025-01-31", "2025-02-28"]))
    with pytest.raises(schema.SchemaError):
        schema.validate_returns(df)


def test_validate_rejects_duplicate_dates():
    df = pd.DataFrame({"portfolio": [0.01, 0.02]}, index=pd.to_datetime(["2025-01-31", "2025-01-31"]))
    with pytest.raises(schema.SchemaError):
        schema.validate_returns(df)


def test_validate_rejects_missing_required():
    df = pd.DataFrame({"portfolio": [0.01, None]}, index=pd.to_datetime(["2025-01-31", "2025-02-28"]))
    with pytest.raises(schema.SchemaError):
        schema.validate_returns(df)


def test_periods_per_year():
    assert schema.periods_per_year("M") == 12
    assert schema.periods_per_year("daily") == 252
    assert schema.periods_per_year("Y") == 1
    assert schema.periods_per_year(250) == 250
    with pytest.raises(schema.SchemaError):
        schema.periods_per_year("X")


def test_align_returns_keeps_common_dates_without_zero_fill():
    idx = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    p = pd.Series([0.01, 0.02, -0.01], index=idx)
    b = pd.Series([0.005, 0.01], index=idx[[0, 2]])
    df = clean.align_returns(p, b, risk_free=0.0001)
    assert list(df.index) == list(idx[[0, 2]])
    assert df.risk_free.tolist() == [0.0001, 0.0001]


def test_align_returns_requires_risk_free_coverage():
    idx = pd.to_datetime(["2025-01-02", "2025-01-03"])
    p = pd.Series([0.01, 0.02], index=idx)
    with pytest.raises(schema.SchemaError):
        clean.align_returns(p, risk_free=pd.Series([0.0001], index=idx[:1]))


def test_align_outer_keeps_missing_as_nan():
    a = pd.Series([1.0], index=pd.to_datetime(["2025-01-02"]))
    b = pd.Series([2.0], index=pd.to_datetime(["2025-01-03"]))
    df = clean.align(a, b, how="outer", names=["a", "b"])
    rep = clean.missing_report(df)
    assert rep.loc["a", "missing"] == 1 and rep.loc["b", "missing"] == 1


def test_drop_duplicate_dates_keeps_last():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2025-01-03", "2025-01-02", "2025-01-03"]))
    out = clean.drop_duplicate_dates(s)
    assert out.tolist() == [2.0, 3.0]


def test_convert_currency_requires_full_fx_coverage():
    idx = pd.to_datetime(["2025-01-02", "2025-01-03"])
    v = pd.Series([100.0, 101.0], index=idx)
    assert clean.convert_currency(v, pd.Series([7.0, 7.1], index=idx)).tolist() == pytest.approx([700.0, 717.1])
    with pytest.raises(schema.SchemaError):
        clean.convert_currency(v, pd.Series([7.0], index=idx[:1]))


def test_flag_outliers_and_stale():
    r = pd.Series([0.01, -0.01, 0.005, 0.0, 0.5, 0.008, -0.004])
    assert clean.flag_outliers(r).tolist() == [False, False, False, False, True, False, False]
    nav = pd.Series([1.0, 1.1, 1.1, 1.1, 1.2])
    assert clean.flag_stale(nav).tolist() == [False, True, True, True, False]


def test_nav_to_returns():
    nav = pd.Series([1.0, 1.02, 1.0098], index=pd.to_datetime(["2025-01-01", "2025-01-31", "2025-02-28"]))
    assert etl_returns.nav_to_returns(nav).tolist() == pytest.approx([0.02, -0.01])


def test_price_to_returns_with_dividend():
    idx = pd.to_datetime(["2025-01-01", "2025-01-31", "2025-02-28"])
    price = pd.Series([10.0, 10.0, 10.5], index=idx)
    div = pd.Series([0.2], index=idx[1:2])
    assert etl_returns.price_to_returns(price, div).tolist() == pytest.approx([0.02, 0.05])


def test_value_to_returns_rejects_flow_without_valuation():
    idx = pd.to_datetime(["2025-01-01", "2025-01-31"])
    v = pd.Series([100.0, 105.0], index=idx)
    with pytest.raises(schema.SchemaError):
        etl_returns.value_to_returns(v, pd.Series([5.0], index=pd.to_datetime(["2025-01-15"])))


def test_load_cashflows_merges_same_day(tmp_path):
    p = tmp_path / "cf.csv"
    p.write_text("date,cashflow\n2025-01-15,100\n2025-01-15,-30\n2025-02-01,50\n")
    cf = files.load_cashflows(p)
    assert cf.tolist() == [70.0, 50.0]
