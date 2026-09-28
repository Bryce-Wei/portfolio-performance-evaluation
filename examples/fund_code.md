# 用真实基金代码评价：命令示例

本页的每条命令都要**联网**：经 akshare 取基金净值与分红、基金概况、指数行情、无风险利率与汇率。需要先安装
`pip install -e ".[data,excel,plot]"`（`data` 为 akshare，`excel` 为 openpyxl，`plot` 为 matplotlib）。
不联网的示例见 [`quickstart.py`](quickstart.py)。

未安装、直接从源码运行时，把 `fundeval` 换成 `python -m fundeval`，并设置 `PYTHONPATH=src`
（Windows PowerShell：`$env:PYTHONPATH="src"`；cmd：`set PYTHONPATH=src`）。

下面的基金代码是开发时在本地实测过的：110020（易方达沪深300ETF联接A，指数型-股票）、000001（华夏成长混合，混合型-灵活）、
110011（易方达优质精选混合(QDII)，QDII-混合偏股，合同基准含港元计价的中证香港300 与“中债总指数”）。

## 一、单只基金报告

```bash
# 只给基金代码：--benchmark 默认 contract，取基金合同业绩比较基准并解析
# 110020 → 沪深300全收益 H00300 × 95% + 活期存款利率 × 5%
fundeval report --fund 110020 --start 2021-01-01 --end 2025-12-31 --out 110020.md

# 同上，显式写出 --benchmark contract，并生成图表（PNG 存到 110020_files/，须与 --out 一起用）
fundeval report --fund 110020 --benchmark contract --start 2021-01-01 --end 2025-12-31 \
    --charts --out 110020.md

# 合同基准有无法自动解析的成分时报错并列出；用 --benchmark-map 指定代码
# 110011 的“中债总指数”名称有歧义，不自动映射
fundeval report --fund 110011 --benchmark-map "中债总指数=cbond:composite" \
    --start 2021-01-01 --end 2025-12-31 --out 110011.xlsx

# 外币成分（110011 的中证香港300，港元）默认 --fx convert：按国家外汇管理局人民币汇率中间价换算为人民币收益
fundeval report --fund 110011 --benchmark-map "中债总指数=cbond:composite" --fx convert \
    --start 2021-01-01 --end 2025-12-31 --out 110011.md

# 风格分析按基金类型自动选择预设（110011 → cn_equity，000001 → cn_balanced），附 36 个月滚动权重
fundeval report --fund 000001 --style auto --style-window 36 \
    --start 2021-01-01 --end 2025-12-31 --charts --out 000001.md

# 多因子分解：A 股指数代理因子 MKT、SMB、HML（中证指数官网全收益指数），HAC 滞后 3 期、t 分布
fundeval report --fund 110020 --factors cn_index_proxy --hac-lags 3 --use-t \
    --start 2021-01-01 --end 2025-12-31 --out 110020_factors.md

# 不用合同基准，直接指定单个指数或“代码:权重”，指数默认换成全收益代码（000300 → H00300）
fundeval report --fund 110011 --benchmark "000300:0.8,H11001:0.2" \
    --start 2021-01-01 --end 2025-12-31 --out 110011_custom.md
```

## 二、多基金横向对比

```bash
# 各自按合同基准逐只评价；单只失败时记录原因并继续；不做综合打分，--sort 只按单一指标排序
fundeval compare --funds 110011,110020,000001 --start 2021-01-01 --end 2025-12-31 \
    --benchmark-map "中债总指数=cbond:composite" --style auto --factors cn_index_proxy \
    --sort sharpe --charts --out compare.md
```

对比表下会注明：不同类型的基金不宜直接比较，只比较同类基金；样本少于 36 个月的基金单独标注。

## 三、数据源不可达时怎么办

各数据源在不同网络环境下的可达性差别很大（开发时本地实测，akshare 1.18.97）：

| 数据源 | 用途 | 可达性 |
| --- | --- | --- |
| 中证指数官网（csindex） | 指数行情，全收益指数（H00300 等）只能从这里取 | 最稳定 |
| 东方财富基金净值 | 基金单位净值、分红、基金概况 | 通常可用；fundf10 页面偶尔超时 |
| 国债收益率（bond_zh_us_rate） | 无风险利率 cgb2y / cgb10y | 通常可用 |
| 东方财富指数行情 | 价格指数（000300 等） | 经常不可达 |
| 中债网站（yield.chinabond.com.cn） | 中债综合财富 / 全价指数 | 经常不可达 |
| Shibor（rate_interbank） | 无风险利率 shibor3m 等 | 约 10 页分页、单次 44–119 秒，经常超时 |

网络失败时命令输出一行中文错误与替代办法，返回码 2；数据源自动切换与使用旧缓存都会写入报告附注。替代办法：

```bash
# 指数只走中证指数官网（默认 auto 已在东方财富失败时改用中证官网，这里跳过东方财富的等待）
fundeval report --fund 110020 --index-source csindex --out 110020.md

# 无风险利率按顺序尝试：先国债 2 年，失败再用 Shibor 3M；或直接用常数年化利率
fundeval report --fund 110020 --rf cgb2y,shibor3m --out 110020.md
fundeval report --fund 110020 --rf 0.018 --out 110020.md

# 中债成分取数失败：用 --benchmark-map 指定中证官网可取的债券指数（会改变基准口径，报告注明“调用方指定”）
fundeval report --fund 000001 --benchmark-map "中债-综合全价(总值)=H11001" --out 000001.md
fundeval report --fund 110011 --benchmark-map "中债总指数=H11001" --out 110011.md

# 汇率取数失败：--fx none 不换算（口径写明“未做汇率换算，基准收益含汇率差异”）
fundeval report --fund 110011 --benchmark-map "中债总指数=cbond:composite" --fx none --out 110011.md

# 单个请求慢：调大单请求超时（默认 30 秒）与每次尝试的总时限（默认 300 秒）；缓存过旧时 --refresh 重新拉取
fundeval report --fund 110020 --timeout 60 --total-timeout 600 --refresh --out 110020.md
```

说明：

- 用 H11001（中证全债）替代中债指数会改变基准口径：编制方与样本不同，H11001 的收益类型未经核实，报告标为未知并附价格指数限定语。工具不会自动替换。
- 取数结果缓存在 `~/.fundeval/cache`（`--cache-dir` 修改，`--no-cache` 关闭）；重新拉取失败时回退到旧缓存，并在附注中注明使用的是哪天的缓存。
- 警告每条一行“警告：<消息>”输出到标准错误；加 `--quiet` 不输出，警告仍写入报告附注。
