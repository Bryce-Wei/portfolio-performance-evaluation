"""数据质量报告：正文第一部分“数据准备”的复核清单。

汇总样本起止与期数、样本长度、缺失、异常收益、疑似停牌或估值滞后，以及净值推算收益与数据源
“日增长率”的交叉核对差异。本模块只标记与报告，不修改或填补数据。

样本长度：少于 MIN_SAMPLE_YEARS（3 年，月度 36 期、季度 12 期、周度 156 期、日度 756 期）时，
问题清单写明“样本较短，统计推断与能力判断受限”（正文第四部分：短样本的 Alpha 显著性与能力判断不可靠）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from fundeval.etl import clean, schema

#: 净值推算收益与数据源日增长率的默认容差：5 个基点。
#: 日增长率通常以百分数保留两位小数，四位小数的净值在 0.5 元附近也有约 2 个基点的舍入误差。
CROSS_CHECK_TOLERANCE = 0.0005

#: 样本长度检查的最短年数：少于 MIN_SAMPLE_YEARS × K 期时提示样本较短
MIN_SAMPLE_YEARS = 3

#: 频率名称（问题清单使用）
_FREQ_NAMES = {252: "日度", 52: "周度", 12: "月度", 4: "季度", 1: "年度"}

#: 按相邻日期间隔的中位数（自然日）推断 K：不超过上界即取该 K
_SPACING_TO_K = ((5, 252), (10, 52), (45, 12), (135, 4))


def infer_periods_per_year(index: pd.DatetimeIndex) -> int | None:
    """按相邻日期间隔的中位数推断一年期数 K（日度 252、周度 52、月度 12、季度 4、年度 1）；少于两个日期时返回 None。"""
    idx = pd.DatetimeIndex(index).sort_values()
    if len(idx) < 2:
        return None
    spacing = float(pd.Series(idx).diff().dt.days.dropna().median())
    for upper, k in _SPACING_TO_K:
        if spacing <= upper:
            return k
    return 1


def sample_length_issue(periods: int, periods_per_year: int | None, min_years: float = MIN_SAMPLE_YEARS) -> str | None:
    """样本长度检查：期数少于 min_years × K 时返回问题描述，否则返回 None（K 未知时不检查）。"""
    if periods_per_year is None:
        return None
    k = int(periods_per_year)
    need = int(round(min_years * k))
    if periods >= need:
        return None
    freq = _FREQ_NAMES.get(k, f"K = {k} ")
    return f"样本较短（{periods} 期，{freq}少于 {need} 期，即不足 {min_years:g} 年），统计推断与能力判断受限"


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
    - ``periods_per_year`` / ``min_years``：样本长度检查使用的 K 与最短年数；``min_periods`` = min_years × K
      （K 未知时为 None，不检查），``short_sample`` 为期数少于 min_periods
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
    periods_per_year: int | None = None
    min_years: float = MIN_SAMPLE_YEARS

    @property
    def min_periods(self) -> int | None:
        return None if self.periods_per_year is None else int(round(self.min_years * self.periods_per_year))

    @property
    def short_sample(self) -> bool:
        return self.min_periods is not None and self.periods < self.min_periods

    @property
    def issue_count(self) -> int:
        n = int(self.missing["missing"].sum()) + len(self.outliers) + len(self.stale) + int(self.short_sample)
        if self.cross_check is not None:
            n += len(self.cross_check)
        return n

    def issues(self) -> list[str]:
        """逐条列出需要复核的问题（中文），没有问题时返回空列表。"""
        out: list[str] = []
        if self.short_sample:
            out.append(sample_length_issue(self.periods, self.periods_per_year, self.min_years))
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
        if self.min_periods is not None:
            verdict = "样本较短，统计推断与能力判断受限" if self.short_sample else "满足"
            rows.append((f"样本长度（不少于 {self.min_periods} 期）", verdict))
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
    periods_per_year: int | None = None,
    min_years: float = MIN_SAMPLE_YEARS,
) -> QualityReport:
    """汇总收益数据的质量问题，返回 QualityReport。

    - 缺失：clean.missing_report，逐列统计，缺失不填补
    - 异常收益：clean.flag_outliers（中位数与 MAD 的稳健 z 值，默认阈值 5）
    - 疑似停牌或估值滞后：给出 ``nav`` 时对净值用 clean.flag_stale（连续 stale_min_run 个
      净值不变）；否则对收益找连续 stale_min_run 期及以上的零收益
    - 交叉核对：``cross_check`` 为 nav_growth_check 的结果，只保留 flagged 的日期
    - 样本长度：期数少于 ``min_years`` × K（默认 3 年，月度 36 期）时，问题清单写明“样本较短，统计推断与
      能力判断受限”；``periods_per_year`` 缺省时按日期间隔推断（infer_periods_per_year）

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

    k = int(periods_per_year) if periods_per_year is not None else infer_periods_per_year(df.index)
    if k is not None and k <= 0:
        raise ValueError(f"periods_per_year 须为正整数，收到 {periods_per_year}")
    if min_years <= 0:
        raise ValueError(f"min_years 须为正数，收到 {min_years}")
    return QualityReport(
        periods_per_year=k,
        min_years=min_years,
        start=df.index[0] if len(df) else None,
        end=df.index[-1] if len(df) else None,
        periods=len(df),
        missing=clean.missing_report(df),
        outliers=outlier_df,
        stale=stale_df,
        cross_check=flagged,
        cross_check_tolerance=tol,
    )
