"""第一部分第 1 节：评价口径配置（fundeval.config）、CLI --config 与 evaluate / compare 的 config 参数。"""

import dataclasses
import warnings
from pathlib import Path

import pandas as pd
import pytest

from fundeval import cli
from fundeval.config import CONFIG_SECTIONS, EvaluationConfig, config_from_dict, load_config
from fundeval.etl.sources import akshare as aks
from fundeval.report import evaluate
from fundeval.report.compare import compare
from fake_akshare import FakeAkshare

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).parent / "data"
COLUMNS = "月份=date,组合收益=portfolio,基准收益=benchmark,无风险收益=risk_free"


def write(tmp_path, text, name="cfg.toml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ------------------------------ 默认值、读取与校验 ------------------------------


def test_defaults_match_cli_defaults():
    """EvaluationConfig 的默认值与 fundeval report 的命令行默认值一致。"""
    cfg = EvaluationConfig()
    ns = cli.build_parser().parse_args(["report", "--input", "x.csv"])
    for dest, value in cfg.cli_values().items():
        assert getattr(ns, dest) == value, dest
    assert cfg.k == 12 and cfg.confidence == 0.95 and cfg.robustness is True
    sub = cli.build_parser().parse_args(["compare", "--funds", "110011"])
    for dest, value in cfg.cli_values().items():
        if hasattr(sub, dest) and dest != "benchmark":  # compare 的 --benchmark 默认 contract，与 None 等价
            assert getattr(sub, dest) == value, dest


def test_example_config_loads():
    cfg = load_config(ROOT / "examples" / "config.toml")
    assert cfg.start == "2021-01-01" and cfg.end == "2025-12-31" and cfg.freq == "M"
    assert cfg.benchmark == "contract" and cfg.benchmark_map == {"中债总指数": "cbond:composite"}
    assert cfg.rf == "shibor3m,cgb2y" and cfg.hac_lags == 3 and cfg.use_t is True
    assert cfg.style == "auto" and cfg.factors == "cn_index_proxy"
    assert cfg.source.endswith("config.toml")
    # 示例覆盖每一节（[monitor] 以注释给出）
    text = (ROOT / "examples" / "config.toml").read_text(encoding="utf-8")
    for section in CONFIG_SECTIONS:
        assert f"[{section}]" in text, section


def test_unknown_keys_are_rejected_with_choices(tmp_path):
    with pytest.raises(ValueError, match=r"\[sample\] 中的未知键 'frequency'，可选：start、end、freq、periods_per_year"):
        load_config(write(tmp_path, '[sample]\nfrequency = "M"\n'))
    with pytest.raises(ValueError, match=r"未知的配置节 \[sampel\]，可选：\[sample\]"):
        load_config(write(tmp_path, '[sampel]\nfreq = "M"\n'))
    with pytest.raises(ValueError, match="须为表"):
        load_config(write(tmp_path, 'sample = "M"\n'))
    with pytest.raises(ValueError, match="不是有效的 TOML"):
        load_config(write(tmp_path, "[sample\n"))
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.toml")


@pytest.mark.parametrize(
    "text, pattern",
    [
        ('[sample]\nfreq = "M"\nperiods_per_year = 0\n', "K.*不小于 1"),
        ('[sample]\nfreq = "M"\nperiods_per_year = 52\n', "K 须为 12"),
        ('[sample]\nfreq = "X"\n', "freq 须为"),
        ('[sample]\nstart = "2025-01-01"\nend = "2024-01-01"\n', "不能晚于"),
        ('[sample]\nstart = "not a date"\n', "无法解析为日期"),
        ("[risk]\nconfidence = 1.0\n", r"\(0, 1\)"),
        ("[risk]\nconfidence = 0\n", r"\(0, 1\)"),
        ('[risk]\nmar = "x"\n', "须为数值"),
        ("[regression]\nhac_lags = -1\n", "不小于 0"),
        ("[regression]\nhac_lags = 1.5\n", "须为整数"),
        ('[regression]\nuse_t = "yes"\n', "true 或 false"),
        ('[factors]\npreset = "ff5"\n', "factors 须为"),
        ('[currency]\nfx = "auto"\n', "fx 须为"),
        ("[risk_free]\nrate = 1.8\n", "年化小数"),
        ('[risk_free]\nrate = 0.018\nsources = "cgb2y"\n', "不能同时给出"),
        ("[data]\ntimeout = 0\n", "正数"),
        ("[data]\ntolerance = -1\n", "非负"),
        ("[monitor]\ntarget_te = -0.01\n", "正数"),
        ("[style]\nwindow = 24\n", "须与 style 一起"),
        ("[benchmark]\ndeposit_rate = 0.35\n", "年化小数"),
    ],
)
def test_numeric_validation(tmp_path, text, pattern):
    with pytest.raises(ValueError, match=pattern):
        load_config(write(tmp_path, text))


def test_normalization():
    cfg = config_from_dict(
        {
            "sample": {"start": pd.Timestamp("2021-01-01").date(), "freq": "d", "periods_per_year": 244},
            "benchmark": {"spec": {"000300": 0.8, "H11001": 0.2}},
            "risk_free": {"rate": 0.018},
            "data": {"use_cache": False, "timeout": 10},
        }
    )
    assert cfg.start == "2021-01-01" and cfg.freq == "D" and cfg.k == 244
    assert cfg.benchmark == "000300:0.8,H11001:0.2" and cfg.rf == "0.018"
    assert cfg.cli_values()["no_cache"] is True and cfg.timeout == 10.0
    assert cfg.evaluate_kwargs() == {"confidence": 0.95, "robustness": True, "periods_per_year": 244}
    assert EvaluationConfig(rf=["cgb2y", "shibor3m"]).rf == "cgb2y,shibor3m"
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.freq = "M"
    with pytest.raises(ValueError, match="监控目标须同时给出"):
        EvaluationConfig(target_te=0.04).monitor_targets


# ------------------------------ evaluate(config=) ------------------------------


def test_evaluate_with_config_and_keyword_priority(worked_example):
    cfg = EvaluationConfig(hac_lags=2, use_t=True, confidence=0.9, robustness=False, fees="费用前收益",
                           target_active=0.03, target_te=0.04, window=6)
    rep = evaluate(worked_example, config=cfg)
    assert rep.capm.cov_type == "HAC" and "VaR（90%）" in " ".join(rep.metrics["label"])
    assert rep.robustness is None and rep.monitoring is not None
    assert rep.scope["费用口径"] == "费用前收益" and rep.scope["配置来源"] == "EvaluationConfig（Python 对象）"
    # 关键字参数优先：显式 hac_lags=None 回到 OLS，confidence 显式给出
    rep2 = evaluate(worked_example, hac_lags=None, confidence=0.95, config=cfg)
    assert rep2.capm.cov_type != "HAC" and "VaR（95%）" in " ".join(rep2.metrics["label"])
    # 不给 config 时行为不变
    plain = evaluate(worked_example)
    assert "配置来源" not in plain.scope and plain.robustness is not None


def test_evaluate_config_interval_and_k(worked_example):
    cfg = EvaluationConfig(start="2025-03-01", end="2025-10-31")
    rep = evaluate(worked_example, config=cfg)
    assert rep.n == 8 and rep.scope["样本起"] == "2025-03-31"
    assert evaluate(worked_example, config=EvaluationConfig()).periods_per_year == 12
    assert evaluate(worked_example, config=EvaluationConfig(freq="Q")).periods_per_year == 4  # K 取自 config
    assert evaluate(worked_example, periods_per_year=12, config=EvaluationConfig(freq="Q")).periods_per_year == 12


# ------------------------------ CLI --config ------------------------------


def test_cli_config_with_overrides(tmp_path):
    cfg = write(
        tmp_path,
        '[sample]\nstart = "2025-01-01"\nend = "2025-06-30"\n'
        "[regression]\nhac_lags = 2\nrobustness = false\n"
        "[risk]\nconfidence = 0.9\n"
        '[fees]\nbasis = "费用前收益"\n',
    )
    out = tmp_path / "r.md"
    args = ["report", "--input", str(DATA / "worked_example.csv"), "--columns", COLUMNS, "--config", str(cfg),
            "--end", "2025-09-30", "--out", str(out)]
    assert cli.main(args) == 0
    text = out.read_text(encoding="utf-8")
    assert f"| 配置来源 | {cfg}（命令行覆盖：--end） |" in text
    assert "| 样本止 | 2025-09-30 |" in text and "| 期数 | 9 |" in text  # --end 覆盖配置
    assert "| 费用口径 | 费用前收益 |" in text and "历史 VaR（90%）" in text
    assert "Newey–West，滞后 2" in text and "## 稳健性" not in text


def test_cli_config_no_overrides_and_errors(tmp_path, capsys):
    cfg = write(tmp_path, "[risk_free]\nrate = 0.012\n")
    out = tmp_path / "r.md"
    args = ["report", "--input", str(DATA / "worked_example.csv"), "--columns", "月份=date,组合收益=portfolio,基准收益=benchmark",
            "--config", str(cfg), "--out", str(out)]
    assert cli.main(args) == 0
    text = out.read_text(encoding="utf-8")
    assert "（命令行覆盖：无）" in text and "1.20%" in text
    bad = write(tmp_path, "[risk]\nconfidence = 2\n", "bad.toml")
    assert cli.main(["report", "--input", "x.csv", "--config", str(bad)]) == 2
    err = capsys.readouterr().err
    assert "错误：配置有误" in err and "(0, 1)" in err


def test_explicit_dests():
    dests = cli._explicit_dests(["report", "--input", "x.csv", "--freq=Q", "--no-cache", "--use-t"])
    assert {"input", "freq", "no_cache", "use_t", "command"} <= dests
    assert "rf" not in dests and "timeout" not in dests


# ------------------------------ compare(config=) ------------------------------


@pytest.fixture
def fake(monkeypatch):
    fake = FakeAkshare()
    monkeypatch.setattr(aks, "_ak", lambda: fake)
    monkeypatch.setattr(aks, "_sleep", lambda s: None)
    return fake


def test_compare_with_config(fake, tmp_path):
    cfg = EvaluationConfig(start="2024-01-01", end="2024-02-29", freq="W", rf=0.018, confidence=0.9,
                           benchmark_map={"中债总指数": "cbond:composite"}, cache_dir=str(tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = compare(["110020", "110011"], config=cfg)
        assert res.scope["配置来源"] == "EvaluationConfig（Python 对象）" and res.scope["频率"].startswith("W")
        assert not res.failures and all("VaR（90%）" in " ".join(r.metrics["label"]) for r in res.reports.values())
        # 显式给出的 freq 优先
        res_m = compare(["110020"], freq="M", end="2024-03-31", config=cfg)
        assert res_m.scope["频率"].startswith("M")
        with pytest.raises(ValueError, match="config 与 options 不能同时给出"):
            compare(["110020"], config=cfg, options=cfg.report_options())


def test_cli_compare_config(fake, tmp_path):
    cfg = write(tmp_path, f'[sample]\nstart = "2024-01-01"\nend = "2024-02-29"\nfreq = "W"\n'
                          f'[risk_free]\nrate = 0.018\n[data]\ncache_dir = "{tmp_path.as_posix()}/c"\n')
    out = tmp_path / "c.md"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert cli.main(["compare", "--funds", "110020", "--config", str(cfg), "--sort", "sharpe", "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert f"| 配置来源 | {cfg}（命令行覆盖：无） |" in text
