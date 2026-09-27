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

`src/fundeval/` 按正文章节组织。当前已实现数据准备、收益衡量、风险调整指标、Alpha 回归与稳健性检验、Brinson 归因、择时模型、持续风险监控与下行及尾部风险，其余模块将陆续加入。

| 模块 | 对应章节 | 状态 |
| --- | --- | --- |
| `etl/`（schema、sources/files、clean、returns） | 一 数据准备 | 已实现（akshare、French 数据源待加入） |
| `returns.py`（TWR、年化、MWR/XIRR、超额收益） | 二 收益衡量 | 已实现 |
| `risk.py`（波动、回撤、Sharpe、IR、Treynor、M²） | 三 风险调整 | 已实现 |
| `alpha/`（regression：因子与 CAPM 回归、HAC 标准误；rolling：滚动 Alpha/Beta、滚动 IR、样本内外切分；fundamental：IR ≈ TC × IC × √BR） | 四 Alpha 来源 | 已实现 |
| `attribution/brinson.py`（单期 BHB / BF、多期 Cariño 链接与对账） | 五 第 2 节 | 已实现 |
| `attribution/timing.py`（Treynor–Mazuy、Henriksson–Merton） | 五 第 5 节 | 已实现 |
| `attribution/` 风格分析（style）、多因子分解（factor）、Campisi | 五 第 1、3、4 节 | 待实现 |
| `monitor.py`（滚动实现 TE（K 须显式给出）、风险倍数、z 值、Green/Yellow/Red 分区、连续 Red） | 七 持续监控 | 已实现（阈值为演示值，需按策略校准） |
| `tail.py`（下行偏差、Sortino、Calmar、历史模拟 VaR 与 ES） | 八 尾部风险 | 已实现 |
| `costs.py`、`report/` | 六、报告 | 待实现 |

```bash
pip install -e ".[test]"
pytest
```

依赖：pandas、numpy、scipy、statsmodels；可选 `excel`（openpyxl，读取 xlsx）与 `test`（pytest）。回归（`alpha/`、`attribution/timing.py`）使用 statsmodels。GitHub Actions（`.github/workflows/tests.yml`）在推送到 main 与 pull request 时于 Python 3.10、3.11、3.12 上运行全部测试。

```python
from fundeval import risk
from fundeval.etl.sources import files

df = files.load_returns(
    "tests/data/worked_example.csv",
    columns={"月份": "date", "组合收益": "portfolio", "基准收益": "benchmark", "无风险收益": "risk_free"},
)
risk.summary(df.portfolio, df.benchmark, df.risk_free, periods_per_year=12)

from fundeval import tail
tail.sortino_ratio(df.portfolio, mar=df.risk_free, periods_per_year=12)  # MAR 需事前指定
tail.historical_var(df.portfolio, confidence=0.95)

from fundeval import monitor
monitor.realized_tracking_error(df.portfolio - df.benchmark, window=6, periods_per_year=12)  # K 必须显式给出

from fundeval.alpha import capm_regression
res = capm_regression(df.portfolio, df.benchmark, df.risk_free, hac_lags=2)  # Newey–West 标准误
res.table(); res.annualized_alpha(12)  # 算术年化 α × K

from fundeval.attribution import brinson_single
brinson_single([0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.038, 0.01, -0.01], [0.03, 0.01, -0.005]).table()
```

收益一律以小数表示（0.02 即 2%）。比率使用同频算术均值与样本标准差（n−1），乘以 √K 年化；分母为零时返回 NaN，表示不适用。`tests/test_worked_example.py` 用第九部分的演示数值作为基准测试。

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
