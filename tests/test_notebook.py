"""examples/walkthrough.ipynb 始终能跑通：不依赖 Jupyter，直接读取 ipynb 的 JSON，在临时目录中按顺序执行代码单元。

notebook 不联网（联网单元由 RUN_ONLINE = False 控制），也不需要 matplotlib。执行时标准输出写入内存，
不受控制台编码（如中文 Windows 的 GBK）影响；文件一律按 UTF-8 读取。
"""

import contextlib
import io
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "examples" / "walkthrough.ipynb"


def load_notebook() -> dict:
    with NOTEBOOK.open(encoding="utf-8") as fh:
        return json.load(fh)


def code_cells(nb: dict) -> list[str]:
    return ["".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"]


def test_notebook_outputs_are_cleared():
    nb = load_notebook()
    assert nb["nbformat"] == 4
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None, "提交前请清空 notebook 输出"


def test_notebook_structure_follows_article_order():
    nb = load_notebook()
    headings = [
        "".join(c["source"]).splitlines()[0] for c in nb["cells"] if c["cell_type"] == "markdown"
    ]
    expected = ["读取与清洗", "数据质量", "单期收益", "收益与风险指标", "尾部风险", "Alpha 回归", "Brinson 单期例题",
                "持续监控", "evaluate 报告与三段结论", "联网示例"]
    positions = [next(i for i, h in enumerate(headings) if name in h) for name in expected]
    assert positions == sorted(positions)
    cells = code_cells(nb)
    assert "RUN_ONLINE = False" in cells[0]
    online = [i for i, c in enumerate(cells) if "if RUN_ONLINE:" in c]
    assert online == [len(cells) - 1]  # 联网单元集中在最后
    assert all("import akshare" not in c and "matplotlib" not in c for c in cells)


def test_notebook_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # 在临时目录中执行：notebook 不依赖当前目录，也不在仓库里写文件
    monkeypatch.setenv("FUNDEVAL_WORKED_EXAMPLE", str(ROOT / "tests" / "data" / "worked_example.csv"))
    namespace = {"__name__": "__main__"}
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        for i, source in enumerate(code_cells(load_notebook())):
            try:
                exec(compile(source, f"walkthrough.ipynb 代码单元 {i + 1}", "exec"), namespace)
            except Exception as exc:  # pragma: no cover - 失败时指明单元
                pytest.fail(f"walkthrough.ipynb 第 {i + 1} 个代码单元出错：{type(exc).__name__}: {exc}")
    text = out.getvalue()
    assert namespace["RUN_ONLINE"] is False and "跳过联网示例" in text
    assert "10.2058%" in text and "2.2327" in text and "8.4668%" in text
    assert "样本较短" in text and "观察到的表现" in text
    assert list(tmp_path.iterdir()) == []
