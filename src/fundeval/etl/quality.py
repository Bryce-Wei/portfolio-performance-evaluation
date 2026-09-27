"""数据质量报告：正文第一部分“数据准备”的复核清单。

汇总样本起止与期数、缺失、异常收益、疑似停牌或估值滞后，以及净值推算收益与数据源
“日增长率”的交叉核对差异。本模块只标记与报告，不修改或填补数据。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from fundeval.etl import clean, schema

#: 净值推算收益与数据源日增长率的默认容差：5 个基点。
#: 日增长率通常以百分数保留两位小数，四位小数的净值在 0.5 元附近也有约 2 个基点的舍入误差。
CROSS_CHECK_TOLERANCE = 0.0005


def nav_growth_check(
    total_return: pd.Series, reported_growth: pd.Series, tolerance: float = CROSS_CHECK_TOLERANCE
) -> pd.DataFrame:
    """交叉核对：由净值与分红推算的总收益，对照数据源给出的日增长率（均为小数）。

    返回按日期排列的 DataFrame，列为 ``total_return``、``reported_growth``、
    ``difference``（前者减后者）与 ``flagged``（|差异| > tolerance）。
    任一方缺失的日期 ``difference`` 为 NaN、``flagged`` 为 False，另由缺失统计反映。
    差异常见原因：分红未计入或除息日错位、份额拆分、净值更正、数据源舍入。
    """
    if tolerance < 0:
        raise ValueError("tolerance 须为非负数")
    df = pd.concat(
        [total_return.astype(float).rename("total_return"), reported_growth.astype(float).rename("reported_growth")],
        axis=1,
        join="outer",
    ).sort_index()
    df["difference"] = df["total_return"] - df["reported_growth"]
    df["flagged"] = (df["difference"].abs() > tolerance).fillna(False).astype(bool)
    df.attrs["tolerance"] = tolerance
    return df


@dataclass(frozen=True)
class QualityReport:
    """数据质量报告。

    - ``start`` / ``end`` / ``periods``：样本起止与期数
    - ``missing``：clean.missing_report 的逐列缺失统计
    - ``outliers``：稳健 z 值标记的异常收益（列 ``date``、``column``、``value``）
    - ``stale``：疑似停牌或估值滞后的区间（列 ``column``、``start``、``end``、``length``）
    - ``cross_check``：净值推算收益与日增长率差异超过容差的日期；未提供核对数据时为 None
    """

    start: pd.Timestamp | None
    end: pd.Timestamp | None
    periods: int
    missing: pd.DataFrame
    outliers: pd.DataFrame
    stale: pd.DataFrame
    cross_check: pd.DataFrame | None = None
    cross_check_tolerance: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def issue_count(self) -> int:
        n = int(self.missing["missing"].sum()) + len(self.outliers) + len(self.stale)
        if self.cross_check is not None:
            n += len(self.cross_check)
        return n

    def issues(self) -> list[str]:
        """逐条列出需要复核的问题（中文），没有问题时返回空列表。"""
        out: list[str] = []
        for col, row in self.missing.iterrows():
            if row["missing"]:
                out.append(f"{col} 缺失 {int(row['missing'])} 期（首个缺失 {row['first_missing']:%Y-%m-%d}），未填补")
        for _, row in self.outliers.iterrows():
            out.append(f"{row['column']} 在 {row['date']:%Y-%m-%d} 收益 {row['value']:.4%}，稳健 z 值异常，需复核")
        for _, row in self.stale.iterrows():
            out.append(
                f"{row['column']} 在 {row['start']:%Y-%m-%d} 至 {row['end']:%Y-%m-%d} 连续 {int(row['length'])} 期不变，"
                "疑似停牌或估值滞后"
            )
        if self.cross_check is not None:
            for date, row in self.cross_check.iterrows():
                out.append(
                    f"{date:%Y-%m-%d} 净值推算收益 {row['total_return']:.4%} 与日增长率 {row['reported_growth']:.4%} "
                    f"相差 {row['difference'] * 1e4:.1f} 个基点"
                )
        out.extend(self.notes)
        return out

    def summary(self) -> pd.DataFrame:
        """汇总表（项目、结果），供报告与导出使用。"""
        rows = [
            ("样本起", f"{self.start:%Y-%m-%d}" if self.start is not None else "—"),
            ("样本止", f"{self.end:%Y-%m-%d}" if self.end is not None else "—"),
            ("期数", str(self.periods)),
        ]
        for col, row in self.missing.iterrows():
            rows.append((f"缺失期数（{col}）", str(int(row["missing"]))))
        rows.append(("异常收益（稳健 z 值）", str(len(self.outliers))))
        rows.append(("疑似停牌或估值滞后区间", str(len(self.stale))))
        if self.cross_check is None:
            rows.append(("净值与日增长率交叉核对", "未提供核对数据"))
        else:
            tol = self.cross_check_tolerance
            label = f"净值与日增长率差异超过 {tol * 1e4:.0f} 个基点的日期" if tol is not None else "净值与日增长率差异日期"
            rows.append((label, str(len(self.cross_check))))
        return pd.DataFrame(rows, columns=["项目", "结果"])


def _stale_runs(values: pd.Series, min_run: int, zero_only: bool) -> list[dict]:
    v = values.dropna()
    if len(v) == 0:
        return []
    flagged = clean.flag_stale(v, min_run=min_run)
    if zero_only:
        flagged &= v.eq(0)
    runs = []
    run_id = (flagged != flagged.shift()).cumsum()
    for _, block in v[flagged].groupby(run_id[flagged]):
        runs.append({"column": values.name, "start": block.index[0], "end": block.index[-1], "length": len(block)})
    return runs


def data_quality_report(
    returns: pd.Series | pd.DataFrame,
    *,
    nav: pd.Series | None = None,
    cross_check: pd.DataFrame | None = None,
    outlier_threshold: float = 5.0,
    stale_min_run: int = 3,
) -> QualityReport:
    """汇总收益数据的质量问题，返回 QualityReport。

    - 缺失：clean.missing_report，逐列统计，缺失不填补
    - 异常收益：clean.flag_outliers（中位数与 MAD 的稳健 z 值，默认阈值 5）
    - 疑似停牌或估值滞后：给出 ``nav`` 时对净值用 clean.flag_stale（连续 stale_min_run 个
      净值不变）；否则对收益找连续 stale_min_run 期及以上的零收益
    - 交叉核对：``cross_check`` 为 nav_growth_check 的结果，只保留 flagged 的日期

    无风险收益列（``risk_free``）通常为常数或缓慢变化，不做异常与停牌检查。
    """
    df = returns.to_frame(returns.name or schema.PORTFOLIO) if isinstance(returns, pd.Series) else returns
    if not isinstance(df.index, pd.DatetimeIndex):
        raise schema.SchemaError("收益的索引必须是 DatetimeIndex")
    df = df.astype(float)
    checked = [c for c in df.columns if c != schema.RISK_FREE]

    outliers = []
    for col in checked:
        r = df[col].dropna()
        if len(r) >= 3:
            flags = clean.flag_outliers(r, threshold=outlier_threshold)
            for date in r.index[flags.to_numpy()]:
                outliers.append({"date": date, "column": col, "value": float(r[date])})
    outlier_df = pd.DataFrame(outliers, columns=["date", "column", "value"])

    stale: list[dict] = []
    if nav is not None:
        stale += _stale_runs(nav.astype(float).rename(schema.NAV), stale_min_run, zero_only=False)
    else:
        for col in checked:
            stale += _stale_runs(df[col], stale_min_run, zero_only=True)
    stale_df = pd.DataFrame(stale, columns=["column", "start", "end", "length"])

    flagged = None
    tol = None
    if cross_check is not None:
        missing_cols = {"total_return", "reported_growth", "difference", "flagged"} - set(cross_check.columns)
        if missing_cols:
            raise ValueError(f"cross_check 缺少列：{sorted(missing_cols)}，请用 nav_growth_check 生成")
        flagged = cross_check[cross_check["flagged"]].drop(columns="flagged")
        tol = cross_check.attrs.get("tolerance")

    return QualityReport(
        start=df.index[0] if len(df) else None,
        end=df.index[-1] if len(df) else None,
        periods=len(df),
        missing=clean.missing_report(df),
        outliers=outlier_df,
        stale=stale_df,
        cross_check=flagged,
        cross_check_tolerance=tol,
    )
