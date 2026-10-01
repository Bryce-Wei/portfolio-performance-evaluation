"""Fama–French 因子库（Kenneth R. French Data Library）：正文第五部分第 3 节“多因子分解”的美国市场因子。

数据来自 Dartmouth Tuck 商学院的公开数据库，每个数据集是一个只含单个 CSV 的 zip：

- ``F-F_Research_Data_Factors``：Mkt-RF、SMB、HML、RF（美国 1 个月国库券）
- ``F-F_Momentum_Factor``：Mom（动量，即 Carhart 四因子中的 UMD）

CSV 的格式（2026-10-01 本地实测，数据截至 202608）依次为：若干行文字说明、空行、表头行
（“,Mkt-RF,SMB,HML,RF” 或 “,Mom”）、月度数据行（“YYYYMM,  数值”，单位为百分数）、一行
“ Annual Factors: January-December ” 之后的年度段（“YYYY,  数值”），最后一行版权声明。
−99.99 或 −999 表示缺失。

**限定**：这是美国市场因子，以美元计价，适合投资美股的 QDII 等基金；人民币计价的基金收益含汇率影响，
因子回归的残差与 Alpha 会混入人民币兑美元的变动。A 股基金应使用 A 股指数代理因子
（attribution.factor 的 cn_index_proxy 系列）。

下载用标准库 urllib，不依赖 akshare；重试、单请求超时、每次尝试的总时限与缓存沿用 etl.sources.akshare
的同一套机制（_cached_call：网络类异常按指数退避共尝试 3 次，重新拉取失败时回退旧缓存并写入附注）。
缓存有效期默认 7 天（FRENCH_MAX_AGE）。连接失败、超时与 HTTP 403 / 429 / 5xx 按网络类异常处理
（ConnectionError / TimeoutError），其余 HTTP 错误与格式错误直接报错。
"""

from __future__ import annotations

import io
import re
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from fundeval.etl import schema
from fundeval.etl.sources import akshare as _fetch

#: 数据库下载地址前缀；数据集 URL 为 <前缀><数据集>_CSV.zip
FRENCH_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

#: 三因子与动量因子的数据集名
FF3_DATASET = "F-F_Research_Data_Factors"
MOMENTUM_DATASET = "F-F_Momentum_Factor"

#: 缓存有效期
FRENCH_MAX_AGE = pd.Timedelta(days=7)

#: 支持的频率：M 月度段、A 年度段（1–12 月）
FREQS = ("M", "A")

#: 缺失值标记（百分数）
MISSING_VALUES = (-99.99, -999.0)

#: 数据源说明
SOURCE_LABEL = "Kenneth R. French Data Library"

#: 计价币种
CURRENCY = "USD"

#: 美国市场因子的限定语（报告口径与结论使用）
US_FACTOR_CAVEAT = (
    "因子为美国市场的 Fama–French 因子（Kenneth R. French Data Library），以美元计价，适合投资美股的 QDII 等基金；"
    "人民币计价的基金收益含汇率影响，A 股基金应使用 cn_index_proxy 系列"
)

#: carhart_factors 的列名映射：{数据库列名: 本包列名}
CARHART_COLUMNS = {"Mkt-RF": "MKT", "SMB": "SMB", "HML": "HML", "Mom": "UMD", "RF": "RF"}

_MONTHLY = re.compile(r"^\d{6}$")
_ANNUAL = re.compile(r"^\d{4}$")

#: 视为网络类（可重试、联网测试 skip）的 HTTP 状态码：拒绝访问（含代理拦截）、限流与服务端错误
_RETRYABLE_HTTP = {403, 408, 429}

#: 缓存表的固定列
_FREQ_COL = "freq"
_DATE_COL = "date"


def dataset_url(dataset: str) -> str:
    """数据集的下载地址。"""
    return f"{FRENCH_BASE_URL}{dataset}_CSV.zip"


def _download(url: str, timeout: float | None) -> bytes:
    """下载 url 的内容；测试中用 monkeypatch 替换。

    连接失败转为 ConnectionError，超时为 TimeoutError（socket.timeout），HTTP 403 / 408 / 429 / 5xx 转为
    ConnectionError（可重试）；其余 HTTP 错误（如 404，数据集名或地址有误）转为 ValueError，不重试。
    """
    request = urllib.request.Request(url, headers={"User-Agent": "fundeval"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code in _RETRYABLE_HTTP or exc.code >= 500:
            raise ConnectionError(f"HTTP {exc.code}：{url}") from exc
        raise ValueError(f"French 因子库返回 HTTP {exc.code}：{url}（数据集名或地址可能有误）") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise TimeoutError(f"{url}：{exc.reason}") from exc
        raise ConnectionError(f"{url}：{exc.reason}") from exc


def read_zip_csv(content: bytes) -> str:
    """从 zip 中取出唯一的 CSV 文本（文件名大小写不限）。zip 中没有或多于一个 CSV 时报错。"""
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise ValueError("French 因子库返回的不是 zip 文件（可能被代理或网页替换）") from exc
    names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
    if len(names) != 1:
        raise ValueError(f"zip 中应只有一个 CSV，实际为 {archive.namelist()}")
    raw = archive.read(names[0])
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _fields(line: str) -> list[str]:
    return [f.strip() for f in line.split(",")]


def parse_french_csv(text: str) -> pd.DataFrame:
    """把 French 因子库的 CSV 文本解析为整齐的表：列 ``freq``（M / A）、``date``（YYYYMM 或 YYYY 文字）
    与各因子列（百分数原值，未换算、未处理缺失标记）。

    以表头行（首个字段为空、其余非空）开始一个数据段，紧随其后的连续数据行属于该段；月度只取第一个
    日期为 6 位数字的段，年度只取第一个日期为 4 位数字的段。说明文字、空行与版权行忽略。
    找不到月度段，或数据行的字段数与表头不符时报错。
    """
    header: list[str] | None = None
    blocks: dict[str, tuple[list[str], list[list[str]]]] = {}
    current: str | None = None
    for line in text.splitlines():
        parts = _fields(line)
        if len(parts) >= 2 and parts[0] == "" and all(parts[1:]):
            header, current = parts[1:], None
            continue
        if header is None or not parts or not parts[0]:
            current = None
            continue
        key = parts[0]
        freq = "M" if _MONTHLY.match(key) else "A" if _ANNUAL.match(key) else None
        if freq is None:  # 说明文字、“Annual Factors”标题或版权行：结束当前段
            current = None
            continue
        if current is None:
            if freq in blocks:  # 同频率的后续段（如其他数据集的第二张表）不取
                header = None
                continue
            current = freq
            blocks[freq] = (header, [])
        elif current != freq:
            raise ValueError(f"French CSV 的同一数据段中日期格式不一致：{key}")
        values = parts[1:]
        if len(values) != len(blocks[freq][0]):
            raise ValueError(f"French CSV 第 {key} 行有 {len(values)} 个数值，表头为 {blocks[freq][0]}")
        blocks[freq][1].append([key, *values])
    if "M" not in blocks:
        raise ValueError("French CSV 中没有找到月度数据段（表头行之后的 YYYYMM 行）")
    frames = []
    for freq, (cols, rows) in blocks.items():
        df = pd.DataFrame(rows, columns=[_DATE_COL, *cols])
        df.insert(0, _FREQ_COL, freq)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    factor_cols = [c for c in out.columns if c not in (_FREQ_COL, _DATE_COL)]
    out[factor_cols] = out[factor_cols].apply(pd.to_numeric, errors="raise")
    return out


def _tidy_to_factors(raw: pd.DataFrame, freq: str, dataset: str) -> pd.DataFrame:
    """整齐表 → 指定频率的因子收益（小数），索引为期末日期；缺失标记换成 NaN。"""
    part = raw[raw[_FREQ_COL] == freq].drop(columns=_FREQ_COL)
    if part.empty:
        raise ValueError(f"{dataset} 中没有{'月度' if freq == 'M' else '年度'}数据段")
    dates = part[_DATE_COL].astype(str).str.strip()
    periods = pd.PeriodIndex(dates.str[:4] + "-" + dates.str[4:6], freq="M") if freq == "M" else pd.PeriodIndex(
        dates, freq="Y"
    )
    values = part.drop(columns=_DATE_COL).astype(float)
    values = values.mask(np.isin(values.to_numpy(), MISSING_VALUES))
    out = values / 100.0
    out.index = pd.DatetimeIndex(periods.end_time.normalize(), name=schema.DATE)
    out.columns = [str(c).strip() for c in out.columns]
    out = out.sort_index()
    if out.index.has_duplicates:
        raise ValueError(f"{dataset} 的{freq}数据段有重复日期")
    return out


def french_factors(
    dataset: str,
    freq: str = "M",
    *,
    cache_dir=None,
    use_cache: bool = True,
    refresh: bool = False,
    cache_max_age=FRENCH_MAX_AGE,
    timeout: float | None = _fetch.DEFAULT_TIMEOUT,
    total_timeout: float | None = _fetch.DEFAULT_TOTAL_TIMEOUT,
) -> pd.DataFrame:
    """下载并解析 French 因子库的一个数据集（如 ``"F-F_Research_Data_Factors"``）。

    - 只取 ``freq`` 对应的数据段：``"M"`` 月度（默认）或 ``"A"`` 年度（1–12 月），另一段不返回
    - 百分数换算为小数；月度索引为当月月末日期（如 202512 → 2025-12-31），年度为当年 12 月 31 日，
      与 etl.returns.to_frequency 的日历期末索引一致
    - −99.99 或 −999 视为缺失（NaN），不填补
    - 缓存有效期 7 天；重试、单请求超时 ``timeout`` 与每次尝试的总时限 ``total_timeout`` 同 akshare 数据源

    **美国市场因子，以美元计价**：适合投资美股的 QDII 等基金；人民币计价的基金收益含汇率影响；
    A 股基金应使用 cn_index_proxy 系列（attribution.factor）。

    结果的 ``attrs`` 记录数据集、频率、币种与数据源说明。
    """
    key = str(freq).upper()[:1]
    if key == "Y":
        key = "A"
    if key not in FREQS:
        raise ValueError(f"French 因子库只提供月度（M）与年度（A）数据，收到 freq={freq!r}")
    url = dataset_url(dataset)

    def fetch() -> pd.DataFrame:
        return parse_french_csv(read_zip_csv(_download(url, timeout)))

    raw = _fetch._cached_call(
        "french", dataset, None, None, fetch, cache_dir=cache_dir, use_cache=use_cache, refresh=refresh,
        cache_max_age=cache_max_age, timeout=timeout, total_timeout=total_timeout, dated=False,
        read_dtype={_DATE_COL: str},
    )
    out = _tidy_to_factors(raw, key, dataset)
    out.attrs = {
        "dataset": dataset,
        "freq": key,
        "currency": CURRENCY,
        "source_label": f"{SOURCE_LABEL}（{dataset}，{'月度' if key == 'M' else '年度'}，美元）",
        "url": url,
    }
    return out


def carhart_factors(freq: str = "M", **kwargs) -> pd.DataFrame:
    """Carhart 四因子：合并 F-F_Research_Data_Factors 与 F-F_Momentum_Factor（按日期取交集），
    返回列 MKT（即 Mkt-RF）、SMB、HML、UMD（即 Mom）、RF，均为小数。

    ``kwargs`` 为 french_factors 的缓存与超时参数。缺失值保留为 NaN，不填补。
    美国市场因子、美元计价，限定语见本模块说明（US_FACTOR_CAVEAT）。
    """
    ff = french_factors(FF3_DATASET, freq, **kwargs)
    mom = french_factors(MOMENTUM_DATASET, freq, **kwargs)
    for name, frame, cols in ((FF3_DATASET, ff, ("Mkt-RF", "SMB", "HML", "RF")), (MOMENTUM_DATASET, mom, ("Mom",))):
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} 缺少列 {missing}（实际列：{list(frame.columns)}）")
    out = ff[["Mkt-RF", "SMB", "HML", "RF"]].join(mom[["Mom"]], how="inner").rename(columns=CARHART_COLUMNS)
    out = out[["MKT", "SMB", "HML", "UMD", "RF"]]
    out.attrs = {
        "dataset": f"{FF3_DATASET} + {MOMENTUM_DATASET}",
        "freq": ff.attrs["freq"],
        "currency": CURRENCY,
        "source_label": f"{SOURCE_LABEL}（{FF3_DATASET}、{MOMENTUM_DATASET}，{'月度' if ff.attrs['freq'] == 'M' else '年度'}，美元）",
    }
    return out


__all__ = [
    "CARHART_COLUMNS",
    "FF3_DATASET",
    "FRENCH_BASE_URL",
    "MOMENTUM_DATASET",
    "US_FACTOR_CAVEAT",
    "carhart_factors",
    "dataset_url",
    "french_factors",
    "parse_french_csv",
    "read_zip_csv",
]
