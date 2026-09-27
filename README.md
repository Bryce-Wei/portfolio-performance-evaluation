# 投资组合绩效评估

围绕收益、风险和可实现的 Alpha，整理投资组合绩效评估方法、风格分析、收益归因、择时检验与持续监控流程。

当前收录文章与计算示例，并在 `src/fundeval/` 中逐步加入对应的 Python 实现。

## 阅读与下载

- [在线阅读全文](docs/投资组合绩效评估.md)：含公式、案例图、归因表和 Excel 计算步骤。
- [下载 Word 修订稿 v1](docs/投资组合绩效评估.docx)：适合离线阅读和继续编辑。

## 文章目录

| 章节 | 内容 |
| --- | --- |
| [一 评价框架与数据准备](docs/投资组合绩效评估.md#framework) | 六个评价维度、基准与数据口径 |
| [二 收益衡量与计算口径](docs/投资组合绩效评估.md#returns) | 累计收益、年化收益、TWR 与 MWR |
| [三 风险调整后的绩效衡量](docs/投资组合绩效评估.md#risk-adjusted) | 波动率、回撤、Sharpe、IR 与 M² |
| [四 Alpha 来源与能力判断](docs/投资组合绩效评估.md#alpha) | IC、有效广度、回归与稳健性检验 |
| [五 风格分析与收益归因](docs/投资组合绩效评估.md#attribution) | Sharpe 风格分析、Brinson、多因子、Campisi 与择时模型 |
| [六 交易成本与策略容量](docs/投资组合绩效评估.md#costs) | 毛收益与净收益、换手率、执行与流动性 |
| [七 持续风险监控与预警](docs/投资组合绩效评估.md#monitoring) | 主动风险预算与 Green Zone 分区 |
| [八 下行风险与尾部风险](docs/投资组合绩效评估.md#tail-risk) | Sortino、Calmar、VaR 与 ES |
| [九 从原始数据到绩效结果](docs/投资组合绩效评估.md#worked-example) | 12 个月演示数据、Excel 公式与计算结果 |
| [十 形成可用于管理决策的结论](docs/投资组合绩效评估.md#conclusion) | 证据、结论与后续管理动作 |

## 代码：fundeval 工具包

`src/fundeval/` 按正文章节组织。当前已实现数据准备（本地文件与 akshare 数据源、频率转换、复合基准、数据质量报告）、收益衡量、风险调整指标、Alpha 回归与稳健性检验、Brinson 归因、择时模型、持续风险监控、下行及尾部风险，以及一键评价报告与命令行。

| 模块 | 对应章节 | 状态 |
| --- | --- | --- |
| `etl/`（schema、sources/files、clean、returns：单期收益与 to_frequency 频率转换） | 一 数据准备 | 已实现 |
| `etl/sources/akshare.py`（基金单位净值加分红的总收益、指数行情、Shibor / 国债利率换算、网络重试与数据源自动切换、带覆盖检查的本地缓存） | 一 数据准备 | 已实现（可选依赖 `data`） |
| `etl/benchmark.py`（指数表与全收益代码、基准收益类型、复合基准合成、合同基准文字解析）、`etl/quality.py`（数据质量报告、净值与日增长率交叉核对） | 一 数据准备 | 已实现 |
| `etl/sources` French 因子数据源 | 一 数据准备 | 待实现 |
| `returns.py`（TWR、年化、MWR/XIRR、超额收益） | 二 收益衡量 | 已实现 |
| `risk.py`（波动、回撤、Sharpe、IR、Treynor、M²） | 三 风险调整 | 已实现 |
| `alpha/`（regression：因子与 CAPM 回归、HAC 标准误、可选 t 分布推断；rolling：滚动 Alpha/Beta、滚动 IR、样本内外切分；fundamental：IR ≈ TC × IC × √BR） | 四 Alpha 来源 | 已实现 |
| `attribution/brinson.py`（单期 BHB / BF、多期 Cariño 链接与对账） | 五 第 2 节 | 已实现 |
| `attribution/timing.py`（Treynor–Mazuy、Henriksson–Merton） | 五 第 5 节 | 已实现 |
| `attribution/` 风格分析（style）、多因子分解（factor）、Campisi | 五 第 1、3、4 节 | 待实现 |
| `costs.py` | 六 交易成本 | 待实现 |
| `monitor.py`（滚动实现 TE（K 须显式给出）、风险倍数、z 值、Green/Yellow/Red 分区、连续 Red） | 七 持续监控 | 已实现（阈值为演示值，需按策略校准） |
| `tail.py`（下行偏差、Sortino、Calmar、历史模拟 VaR 与 ES） | 八 尾部风险 | 已实现 |
| `report/`（evaluate 按六个评价维度汇总、conclusion 三段结论、to_markdown、to_excel）与 `cli.py` | 九、十 | 已实现 |

### 安装

```bash
pip install -e ".[data,excel]"      # data：akshare 数据源；excel：读写 xlsx
pip install -e ".[test]" && pytest  # 测试不访问网络
pytest -m network                   # 本地运行联网冒烟测试（需安装 data）
```

联网冒烟测试按数据源拆成独立用例（基金、东方财富指数、中证指数、中债、Shibor、国债），一个数据源不可达不影响其他项；中债网站 yield.chinabond.com.cn 在部分网络环境下无法建立连接，遇到连接类异常时跳过并注明原因，数据格式或列名错误仍然算失败。

依赖：pandas、numpy、scipy、statsmodels；可选 `data`（akshare）、`excel`（openpyxl）与 `test`（pytest）。akshare 只在使用 `fundeval.etl.sources.akshare` 时导入，未安装时报错并提示安装命令。GitHub Actions（`.github/workflows/tests.yml`）在推送到 main 与 pull request 时于 Python 3.10、3.11、3.12 上运行全部测试。

### 命令行

```bash
# 经 akshare 取基金净值与分红、复合基准与无风险利率，生成月度报告
fundeval report --fund 110011 --benchmark "000300:0.8,H11001:0.2" \
    --start 2021-01-01 --end 2025-12-31 --freq M --out report.md

# 基准也可以写成合同文字，名称按 etl/benchmark.py 的指数表换算为代码，表外名称报错
fundeval report --fund 110011 --benchmark "沪深300指数收益率*80%+中债综合指数收益率*20%" \
    --start 2021-01-01 --end 2025-12-31 --out report.xlsx

# 东方财富或 Shibor 接口不稳定时，直接指定中证指数官网与常数无风险利率
fundeval report --fund 110020 --benchmark 000300 --index-source csindex --rf 0.018 \
    --start 2021-01-01 --end 2025-12-31 --hac-lags 3 --use-t --out report.md

# 本地 CSV / Excel，不依赖网络
fundeval report --input tests/data/worked_example.csv \
    --columns "月份=date,组合收益=portfolio,基准收益=benchmark,无风险收益=risk_free" --out report.md
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--fund` / `--input` | 二选一 | 基金代码（经 akshare 取单位净值与分红）或本地收益表 |
| `--benchmark` | 无 | `"000300:0.8,H11001:0.2"`（代码:权重）、单个代码或合同基准文字 |
| `--index-return-type` | `total` | `total` 把基准成分换成全收益指数（000300 → H00300 等，见下节），没有全收益版本的成分沿用原代码并警告；`price` 使用价格指数，报告写明高估限定 |
| `--index-source` | `auto` | `auto`：H 开头等中证代码走中证指数官网，纯数字代码先试东方财富、网络失败后改用中证官网；也可指定 `em` 或 `csindex` |
| `--rf` | `auto` | `auto`：`--fund` 时按 Shibor 3M → 国债 2 年期依次尝试，全部失败时报错并提示改用常数；本地文件（无 risk_free 列时）按 0 计。也可取 `shibor3m`、`shibor1m`、`shibor_on`、`cgb2y`、`cgb10y` 或常数年化利率（如 `0.018`） |
| `--freq` / `--input-freq` | `M` / 同 `--freq` | 评价频率；`--input-freq D --freq M` 先把日度收益按期内复利合成为月度 |
| `--hac-lags` / `--use-t` | 无 / 关 | Newey–West 标准误的滞后阶数；`--use-t` 让 HAC 的 p 值与置信区间也用 t 分布。月度样本不足 120 期且使用 HAC 时建议加 `--use-t`，否则正态近似会高估显著性 |
| `--mar` | 无风险收益 | Sortino 的最低可接受收益（每期，小数） |
| `--target-active` / `--target-te` / `--window` | 无 | 三者同时给出时报告包含持续监控分区 |
| `--tolerance` | `0.0005` | 净值推算收益与日增长率交叉核对容差（5 个基点） |
| `--fees` / `--title` | `费用后净值` / 自动 | 费用口径说明与报告标题 |
| `--cache-dir` / `--no-cache` / `--refresh` | `~/.fundeval/cache` | 原始数据缓存目录、关闭缓存、强制重新拉取 |
| `--timeout` | `30` | 单个 HTTP 请求（连接与读取）的超时秒数。akshare 内部调用 requests 时多未设超时，请求可能无限挂起；分页接口每页单独计时；超时后按网络异常重试或切换数据源 |
| `--total-timeout` | `300` | 每次尝试（一次接口调用，可含多个分页请求）的兜底总时限秒数，每次重试重新计时；超时后按网络异常重试或切换数据源 |

无风险利率均按复利口径 (1 + y)^(1/K) − 1 换算为每期，并取期初已知的报价。组合与基准的对齐由 `evaluate` 完成，报告口径写明“组合 N 期、基准 M 期、共同 K 期”，被丢弃的期列入附注。

数据源与缓存：超时分两层。`--timeout`（默认 30 秒，Python 中为各取数函数的 `timeout` 参数）限制每个 HTTP 请求的连接与读取，足以防止连接挂起；一次接口调用可能翻很多页，例如 Shibor 3M（`rate_interbank`）约 10 页、本地实测 44–119 秒，每页各自计时，不会因累计耗时而超时。`--total-timeout`（默认 300 秒，参数 `total_timeout`）限制一次接口调用的总时长，只作兜底，防止上游不经 requests 或反复慢而不断；它由后台线程实现，线程无法强行终止，超时后该线程可能仍在运行，其结果会被丢弃。两个参数设为 `None` 表示不启用对应一层。网络类异常（requests 异常、连接断开、超时）自动重试，共 3 次，指数退避；指数与无风险利率的自动切换都会发出警告，实际来源写入报告口径与附注。缓存命中时检查覆盖范围：数据最后日期早于截止日（或今天）之前最后一个工作日 7 天以上即重新拉取；不接受日期参数的接口（基金净值、Shibor、中债）与未给截止日的请求，缓存文件超过 1 天也会重新拉取（Python 中可用 `cache_lag_days`、`cache_max_age` 调整）。重新拉取失败时回退到旧缓存，但一定发出警告，并在报告附注中注明“使用 YYYY-MM-DD 的缓存数据”。网络失败导致无法出报告时，命令输出一行中文错误与替代办法，返回码 2。

### 价格指数与全收益指数

基金净值包含成分股分红，沪深 300（000300）等价格指数不含分红。用价格指数作基准，会把股息率计入基金的超额收益与 Alpha。在本地用 akshare 1.18.97 实测（2026-09-27）易方达沪深300ETF联接A（110020），2021-01 至 2025-12 月度：

| 基准 | CAPM 年化 Alpha | t 值 |
| --- | --- | --- |
| 沪深 300 价格指数（000300） | +2.15% | 3.94 |
| 沪深 300 全收益指数（H00300） | −0.18% | −1.07 |

以价格指数为基准时，结论会写成“存在正截距的统计证据”；换成全收益指数后，Alpha 不再显著。这一差异全部来自成分股分红。因此 CLI 默认 `--index-return-type total`，把 000300、000905、000906、000852、000016 换成 H00300、H00905、H00906、H00852、H00016（经中证指数官网 `stock_zh_index_hist_csindex` 取数）。中债“财富”指数本身是全收益口径；H11001 等中证债券指数的口径在指数表中标为未知。基准含价格指数或未知类型成分时，报告在口径、附注与结论中都会写明“价格指数不含成分股分红，超额收益与 Alpha 会高估约为股息率的幅度”。在 Python 中可用 `fundeval.etl.benchmark.total_return_code("000300")` 查询全收益代码，用 `labels={"benchmark_return_type": ...}` 把收益类型传给 `evaluate`。

### Python 示例

```python
from fundeval import risk, tail
from fundeval.etl.sources import files
from fundeval.report import evaluate, to_markdown, to_excel

df = files.load_returns(
    "tests/data/worked_example.csv",
    columns={"月份": "date", "组合收益": "portfolio", "基准收益": "benchmark", "无风险收益": "risk_free"},
)
risk.summary(df.portfolio, df.benchmark, df.risk_free, periods_per_year=12)
tail.sortino_ratio(df.portfolio, mar=df.risk_free, periods_per_year=12)  # MAR 需事前指定

report = evaluate(
    df, periods_per_year=12,
    monitor_targets={"target_active_return": 0.03, "target_te": 0.02, "window": 6},
    labels={"portfolio": "演示组合", "benchmark": "演示基准"},
)
report.metric("information_ratio")  # 2.2327
print(report.conclusion())          # 观察到的表现 / 可以支持的解释 / 需要进一步验证的判断
to_markdown(report, "report.md"); to_excel(report, "report.xlsx")

from fundeval.alpha import capm_regression
res = capm_regression(df.portfolio, df.benchmark, df.risk_free, hac_lags=2, use_t=True)  # 小样本 HAC 用 t 分布
res.table(); res.annualized_alpha(12)  # 算术年化 α × K

from fundeval.etl.sources import akshare as aks  # 需 pip install "fundeval[data]"
from fundeval.etl.benchmark import benchmark_return_type, total_return_code
with aks.collect_notes() as notes:  # 收集数据源自动切换与旧缓存回退的说明
    fund, check = aks.fund_returns("110020", "2021-01-01", "2025-12-31", freq="M", return_check=True)
    code = total_return_code("000300")  # "H00300"
    bench = aks.index_returns(code, "2021-01-01", "2025-12-31", freq="M")
    rf = aks.risk_free_returns("2021-01-01", "2025-12-31", "M", source="auto", index=fund.index)
report = evaluate(
    fund, bench, rf, cross_check=check, hac_lags=3, use_t=True, notes=notes,
    labels={"benchmark_return_type": benchmark_return_type([code]), "risk_free": rf.attrs["description"]},
)
```

收益一律以小数表示（0.02 即 2%）。比率使用同频算术均值与样本标准差（n−1），乘以 √K 年化；分母为零时返回 NaN，表示不适用；缺失值只标记，不填零。基金总收益以单位净值加除息日分红计算，累计净值没有做分红再投资，不能直接当复权净值使用。`tests/test_worked_example.py` 用第九部分的演示数值作为基准测试，`evaluate` 在同一数据上给出一致的结果。

## 仓库结构

```text
.
├── README.md                     # 简介、目录与阅读入口
├── pyproject.toml                # fundeval 包配置与依赖
├── .github/workflows/tests.yml   # CI：多版本 Python 运行 pytest
├── docs/
│   ├── 投资组合绩效评估.md        # 完整正文
│   ├── 投资组合绩效评估.docx      # Word 修订稿 v1
│   └── assets/
│       └── style-weights.png      # 文章案例图
├── src/fundeval/                 # 工具包源码
└── tests/                        # 单元测试与第九部分基准测试
```

## 数据说明

文章中的新增数值为教学演示，不代表真实基金业绩。原有基金案例保留其数据口径和解释边界；方法资料链接列于正文末尾。
