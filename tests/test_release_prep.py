"""v0.2.0 发布前收尾：图例不遮挡曲线、CLI 警告格式与 --quiet、版本号单一来源与 --version、
离线示例 quickstart、docs/examples、CHANGELOG 与 README 结构。

全部离线；图表相关测试在未安装 matplotlib 时 skip。运行 quickstart 的测试在临时目录中完成，不写入仓库。
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fundeval
from fundeval import cli
from fundeval.etl.sources import akshare as aks
from fake_akshare import FakeAkshare

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).parent / "data"
COLUMNS = "月份=date,组合收益=portfolio,基准收益=benchmark,无风险收益=risk_free"


# ------------------------------ 一、图例不遮挡数据 ------------------------------


@pytest.fixture
def mpl():
    mpl = pytest.importorskip("matplotlib")
    from fundeval.report.charts import require_matplotlib

    require_matplotlib()
    return mpl


def _returns(n_series: int, periods: int = 36) -> dict[str, pd.Series]:
    idx = pd.date_range("2021-01-31", periods=periods, freq="ME")
    rng = np.random.default_rng(7)
    # 英文标签下图例名较长（与“代码 + 简称”的长度相当），不依赖中文字体
    names = ["110011 E Fund Quality Select Mixed (QDII)", "110020 E Fund CSI 300 ETF Feeder A", "000001 ChinaAMC Growth Mixed",
             "161725 Fund D", "005827 Fund E", "000002 Fund F", "000003 Fund G", "000004 Fund H"]
    return {names[i]: pd.Series(rng.normal(0.004, 0.04, periods), index=idx) for i in range(n_series)}


def _assert_legend_clear(fig) -> str:
    """保存图片后检查：图例的外接框与坐标区不相交（图例在坐标区之外），坐标区内任何数据点都不落在图例框内，
    且保存的外接框包含整个图例（没有被裁掉）。返回图例位置 "right" 或 "bottom"。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    renderer = fig.canvas.get_renderer()
    ax = next(a for a in fig.axes if a.get_legend() is not None)
    legend = ax.get_legend().get_window_extent(renderer)
    axes_box = ax.get_window_extent(renderer)
    assert not legend.overlaps(axes_box), (legend, axes_box)
    for line in ax.get_lines():
        xy = ax.transData.transform(line.get_xydata())
        inside = (xy[:, 0] >= legend.x0) & (xy[:, 0] <= legend.x1) & (xy[:, 1] >= legend.y0) & (xy[:, 1] <= legend.y1)
        assert not inside.any()
    saved = fig.get_tightbbox(renderer)  # 英寸；换算成像素后应包含图例
    dpi = fig.dpi
    assert saved.x0 * dpi <= legend.x0 + 1 and saved.x1 * dpi >= legend.x1 - 1
    assert saved.y0 * dpi <= legend.y0 + 1 and saved.y1 * dpi >= legend.y1 - 1
    if legend.x0 >= axes_box.x1:
        return "right"
    # 下方图例不压在横轴刻度标签上
    assert legend.y1 <= ax.xaxis.get_tightbbox(renderer).y0 + 1
    return "bottom"


def test_compare_legend_outside_axes_right_for_few_funds(mpl):
    import matplotlib.pyplot as plt

    from fundeval.report.charts import compare_figure

    fig = compare_figure(_returns(3), lang="en")
    try:
        assert _assert_legend_clear(fig) == "right"
    finally:
        plt.close(fig)


def test_compare_legend_below_axes_for_many_funds(mpl):
    import matplotlib.pyplot as plt

    from fundeval.report.charts import LEGEND_RIGHT_MAX, compare_figure

    fig = compare_figure(_returns(LEGEND_RIGHT_MAX + 2), lang="en")
    try:
        assert _assert_legend_clear(fig) == "bottom"
    finally:
        plt.close(fig)


def test_wealth_and_style_legends_outside_axes(mpl):
    import matplotlib.pyplot as plt

    from fundeval.report.charts import style_figure, wealth_figure

    r = _returns(2)
    data = pd.DataFrame({"portfolio": list(r.values())[0], "benchmark": list(r.values())[1]})
    weights = pd.DataFrame({"A": 0.5, "B": 0.3, "cash": 0.2}, index=data.index)
    for fig in (wealth_figure(data, "en"), style_figure(weights, 12, "en")):
        try:
            assert _assert_legend_clear(fig) == "right"
        finally:
            plt.close(fig)


def test_saved_compare_png_is_wider_than_axes_area(mpl, tmp_path):
    """保存到文件时按包含图例的外接框裁切：右侧图例使图片比默认画布更宽，图例没有被裁掉。"""
    from matplotlib.image import imread

    from fundeval.report.charts import DPI, _save, compare_figure

    path = _save(compare_figure(_returns(3), lang="en"), tmp_path / "compare.png")
    height, width = imread(path).shape[:2]
    assert width > 8 * DPI * 0.9 and width / height > 8 / 3.6 * 0.9


def test_rolling_chart_skipped_when_sample_equals_window(mpl, tmp_path, worked_example):
    """12 期样本只有 1 个 12 期窗口，滚动曲线只有一个点：不画空图，注明原因。"""
    from fundeval.report import evaluate
    from fundeval.report.charts import report_charts

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = report_charts(evaluate(worked_example, periods_per_year=12), tmp_path)
    assert "rolling" not in res.paths and "只有 1 个窗口" in res.skipped["rolling"]


def test_percent_axis_labels_distinct_on_narrow_range(mpl, worked_example):
    """回撤只有几个百分点时，纵轴刻度标签互不相同（不会出现两个“-1%”）。"""
    import matplotlib.pyplot as plt

    from fundeval.report.charts import drawdown_figure

    fig = drawdown_figure(worked_example, "en")
    try:
        fig.canvas.draw()
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
        assert len(labels) == len(set(labels)) > 2
    finally:
        plt.close(fig)


# ------------------------------ 二、CLI 警告格式与 --quiet ------------------------------


def _fund_args(tmp_path, out, *extra):
    return [
        "report", "--fund", "110011", "--benchmark", "000300:0.8,H11001:0.2",
        "--start", "2024-01-01", "--end", "2024-02-29", "--freq", "W", "--rf", "0.018",
        "--cache-dir", str(tmp_path / "cache"), "--out", str(out), *extra,
    ]


def test_cli_warnings_one_line_each_without_source_path(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(aks, "_ak", lambda: FakeAkshare())
    out = tmp_path / "fund.md"
    assert cli.main(_fund_args(tmp_path, out)) == 0
    err = capsys.readouterr().err
    assert ".py:" not in err and "RuntimeWarning" not in err and "return _cached_call(" not in err
    lines = [line for line in err.splitlines() if line.strip()]
    warning_lines = [line for line in lines if not line.startswith("报告已写入")]
    assert warning_lines and all(line.startswith("警告：") for line in warning_lines)
    assert len(warning_lines) == len(set(warning_lines))
    assert any("日增长率" in line for line in warning_lines)
    assert any("H11001" in line for line in warning_lines)


def test_cli_quiet_hides_warnings_but_keeps_report_notes(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(aks, "_ak", lambda: FakeAkshare())
    loud, quiet = tmp_path / "loud.md", tmp_path / "quiet.md"
    assert cli.main(_fund_args(tmp_path, loud)) == 0
    shown = [line.removeprefix("警告：") for line in capsys.readouterr().err.splitlines() if line.startswith("警告：")]
    assert cli.main(_fund_args(tmp_path, quiet, "--quiet")) == 0
    err = capsys.readouterr().err
    assert "警告" not in err and "Warning" not in err
    notes = quiet.read_text(encoding="utf-8").split("## 附注")[1].split("## 结论")[0]
    # 取数与评价阶段显示过的每条警告都在附注中（--quiet 不影响附注）
    for message in shown:
        assert message in " ".join(notes.split())
    assert "日增长率" in notes


def test_cli_quiet_compare(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(aks, "_ak", lambda: FakeAkshare())
    out = tmp_path / "cmp.md"
    code = cli.main([
        "compare", "--funds", "110011,110020", "--benchmark", "000300", "--start", "2024-01-01", "--end", "2024-02-29",
        "--freq", "W", "--rf", "0.018", "--cache-dir", str(tmp_path / "cache"), "--out", str(out), "--quiet",
    ])
    assert code == 0
    err = capsys.readouterr().err
    assert "警告" not in err and ".py:" not in err
    assert "日增长率" in out.read_text(encoding="utf-8").split("## 附注")[1]


def test_warning_log_deduplicates_and_flattens():
    buf = io.StringIO()
    log = cli.WarningLog(stream=buf)
    with log.capture():
        for _ in range(3):
            warnings.warn("重新拉取失败\n  已改用旧缓存", RuntimeWarning)
        warnings.warn("另一条", UserWarning)
    assert buf.getvalue().splitlines() == ["警告：重新拉取失败 已改用旧缓存", "警告：另一条"]
    assert log.since(0) == ["重新拉取失败 已改用旧缓存", "另一条"]
    quiet_buf = io.StringIO()
    quiet = cli.WarningLog(quiet=True, stream=quiet_buf)
    with quiet.capture():
        warnings.warn("不显示", RuntimeWarning)
    assert quiet_buf.getvalue() == "" and quiet.since(0) == ["不显示"]


def test_python_api_warnings_unchanged_after_cli(tmp_path):
    """CLI 退出后恢复原有的 warnings 设置：Python API 照常发出 RuntimeWarning。"""
    before = warnings.showwarning
    cli.main(["report", "--input", str(DATA / "worked_example.csv"), "--columns", COLUMNS, "--out", str(tmp_path / "r.md"), "--quiet"])
    assert warnings.showwarning is before
    from fundeval import returns

    with pytest.warns(RuntimeWarning):
        returns.irr([-100, 230, -132])  # 多个内部收益率根的提示（第二部分）


def test_cli_errors_still_printed_with_quiet(tmp_path, capsys):
    code = cli.main(["report", "--input", str(DATA / "worked_example.csv"), "--columns", COLUMNS,
                     "--out", str(tmp_path / "x.pdf"), "--quiet"])
    assert code == 2 and capsys.readouterr().err.startswith("错误：")


# ------------------------------ 三、版本号与 --version ------------------------------


def test_version_single_source():
    from fundeval._version import __version__

    assert fundeval.__version__ == __version__ == "0.2.0"
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^dynamic = \["version"\]', text, re.M)
    assert 'version = {attr = "fundeval._version.__version__"}' in text
    assert not re.search(r'^version = "', text, re.M)


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == "fundeval 0.2.0"


def test_version_works_from_source_without_package_metadata(tmp_path):
    """以 PYTHONPATH=src 运行源码（用户本机从 zip 解压后的方式）时没有已安装包的元数据：
    子进程先让 importlib.metadata 查不到任何包，再读取 __version__ 与 --version。"""
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    code = (
        "import importlib.metadata as m\n"
        "def missing(*a, **k):\n"
        "    raise m.PackageNotFoundError('fundeval')\n"
        "m.version = m.distribution = m.metadata = missing\n"
        "import fundeval\n"
        "from fundeval import cli\n"
        "print(fundeval.__version__)\n"
        "cli.main(['--version'])\n"
    )
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=tmp_path, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["0.2.0", "fundeval", "0.2.0"]
    out = subprocess.run([sys.executable, "-m", "fundeval", "--version"], env=env, cwd=tmp_path, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "fundeval 0.2.0"


# ------------------------------ 四、离线示例 ------------------------------


def _run_quickstart(tmp_path, *extra, encoding=None):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    if encoding:
        env["PYTHONIOENCODING"] = encoding
    return subprocess.run(
        [sys.executable, str(ROOT / "examples" / "quickstart.py"), "--out", str(tmp_path / "out"), *extra],
        env=env, cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )


def test_quickstart_runs_offline_in_temp_dir(tmp_path):
    before = sorted(p.name for p in ROOT.iterdir())
    res = _run_quickstart(tmp_path)
    assert res.returncode == 0, res.stderr
    # 关键指标与正文第九部分一致
    for value in ("10.2058%", "6.7390%", "1.1254", "2.2327", "3.0000%", "8.4668%"):
        assert value in res.stdout
    out = tmp_path / "out"
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "# 正文第九部分演示组合评价报告" in text and "## 结论" in text
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        assert "## 图表" not in text and "matplotlib" in res.stderr
    else:
        assert "## 图表" in text
        for name in ("wealth.png", "drawdown.png"):
            assert (out / "report_files" / name).stat().st_size > 1000
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        assert not (out / "report.xlsx").exists()
    else:
        assert (out / "report.xlsx").exists()
    # 不写入仓库
    assert sorted(p.name for p in ROOT.iterdir()) == before
    assert not (tmp_path / "quickstart_output").exists()


def test_quickstart_no_charts(tmp_path):
    res = _run_quickstart(tmp_path, "--no-charts")
    assert res.returncode == 0, res.stderr
    assert not (tmp_path / "out" / "report_files").exists()
    assert "## 图表" not in (tmp_path / "out" / "report.md").read_text(encoding="utf-8")


def test_quickstart_output_redirected_as_gbk(tmp_path):
    """中文 Windows 重定向输出时标准输出为 GBK：“M²”无法编码，脚本改用 ASCII 写法，不中断。"""
    res = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "quickstart.py"), "--out", str(tmp_path / "out"), "--no-charts"],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "gbk"},
        cwd=tmp_path, capture_output=True, timeout=300,
    )
    assert res.returncode == 0, res.stderr.decode("gbk", errors="replace")
    out = res.stdout.decode("gbk")
    assert "M^2" in out and "8.4668%" in out and "累计收益（组合）" in out
    # 写入文件的报告仍为 UTF-8，保留原字符
    assert "M²" in (tmp_path / "out" / "report.md").read_text(encoding="utf-8")


def _run_cli_gbk(tmp_path, *args):
    return subprocess.run(
        [sys.executable, "-m", "fundeval", *args],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "gbk"},
        cwd=tmp_path, capture_output=True, timeout=300,
    )


def test_cli_stdout_markdown_and_help_as_gbk(tmp_path):
    """--out 缺省时 Markdown 打印到标准输出（含“−”“²”），--help 含“−”：GBK 下都不中断。"""
    res = _run_cli_gbk(tmp_path, "report", "--input", str(DATA / "worked_example.csv"), "--columns", COLUMNS)
    assert res.returncode == 0, res.stderr.decode("gbk", errors="replace")
    text = res.stdout.decode("gbk")
    assert "| M^2 | 8.4668% |" in text and "n-1" in text and "√K" in text
    res = _run_cli_gbk(tmp_path, "report", "--help")
    assert res.returncode == 0 and "H00300 - rf" in res.stdout.decode("gbk")
    res = _run_cli_gbk(tmp_path, "report", "--input", str(tmp_path / "missing.csv"))
    assert res.returncode == 2 and res.stderr.decode("gbk").startswith("错误：")


def test_cli_warning_lines_on_gbk_stream():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="gbk")
    log = cli.WarningLog(stream=stream)
    with log.capture():
        warnings.warn("MKT = H00300 − rf，M² 缺失", RuntimeWarning)
    stream.flush()
    assert raw.getvalue().decode("gbk") == "警告：MKT = H00300 - rf，M^2 缺失\n"
    assert log.since(0) == ["MKT = H00300 − rf，M² 缺失"]  # 附注保留原字符


def test_docs_examples_committed_and_linked():
    report = ROOT / "docs" / "examples" / "report.md"
    text = report.read_text(encoding="utf-8")
    assert "10.2058%" in text and "1.1254" in text and "2.2327" in text
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    assert images, "示例报告应嵌入图表"
    for rel in images:
        assert (report.parent / rel).exists(), rel
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/examples/report.md" in readme


def test_ci_runs_quickstart():
    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert "examples/quickstart.py" in workflow


def test_fund_code_examples_cover_options():
    text = (ROOT / "examples" / "fund_code.md").read_text(encoding="utf-8")
    for flag in ("fundeval report", "fundeval compare", "--benchmark contract", "--benchmark-map", "--fx convert",
                 "--fx none", "--style auto", "--factors cn_index_proxy", "--charts", "--rf", "--index-source csindex"):
        assert flag in text, flag
    assert "联网" in text


# ------------------------------ 五、CHANGELOG 与 README ------------------------------


def test_changelog_structure():
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert re.search(r"^## \[?0\.2\.0\]?", text, re.M)
    section = text.split("## ", 2)[1] if text.startswith("## ") else text.split("\n## ", 2)[1]
    for heading in ("### 新增", "### 修正", "### 口径与限定"):
        assert heading in section
    for n in range(1, 11):
        assert f"#{n}" in text, f"PR #{n}"
    for key in ("全收益", "每10份", "单请求超时", "稳健性", "汇率"):
        assert key in text, key
    assert "第" in text and "部分" in text


def test_readme_structure():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    assert headings[0].startswith("## 快速开始")
    assert 'pip install -e ".[data,excel,plot]"' in text
    assert "examples/quickstart.py" in text
    quick = text.split("## 快速开始")[1].split("\n## ")[0]
    assert re.search(r"fundeval report --fund 110020 --charts --out \S+\.md", quick)
    assert any("章节对照" in h for h in headings)
    limits = text.split("## 口径与已知限制")[1].split("\n## ")[0]
    for key in ("全收益", "指数代理", "汇率", "数据源", "样本", "风格权重", "已扣"):
        assert key in limits, key
    assert text.index("## 快速开始") < text.index("## 口径与已知限制")
