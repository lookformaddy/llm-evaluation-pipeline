from __future__ import annotations

import json
from pathlib import Path

from llm_eval_pipeline.cli import main

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "golden" / "support_qa_v1.jsonl"
BASE_CFG = ROOT / "examples" / "sut_config_baseline.json"
REG_CFG = ROOT / "examples" / "sut_config_regressed.json"
THRESHOLDS = ROOT / "examples" / "thresholds.json"


def test_validate_dataset_command(capsys):
    assert main(["validate-dataset", str(DATASET)]) == 0
    out = capsys.readouterr().out
    assert "OK: 12 cases" in out
    assert "content_hash:" in out


def test_run_command_writes_reports(tmp_path, capsys):
    out_dir = tmp_path / "out"
    rc = main([
        "run", "--dataset", str(DATASET), "--config", str(BASE_CFG),
        "--out", str(out_dir), "--history", str(tmp_path / "history.json"),
    ])
    assert rc == 0
    for name in ("report.json", "report.md", "report.html"):
        assert (out_dir / name).exists()
    report = json.loads((out_dir / "report.json").read_text())
    assert report["metrics"]["case_count"] == 12
    history = json.loads((tmp_path / "history.json").read_text())
    assert len(history) == 1
    assert "overall mean" in capsys.readouterr().out


def test_run_command_gate_pass_and_fail(tmp_path):
    baseline_dir = tmp_path / "baseline"
    assert main([
        "run", "--dataset", str(DATASET), "--config", str(BASE_CFG),
        "--out", str(baseline_dir),
    ]) == 0

    same_dir = tmp_path / "same"
    assert main([
        "run", "--dataset", str(DATASET), "--config", str(BASE_CFG),
        "--out", str(same_dir), "--baseline", str(baseline_dir / "report.json"),
        "--thresholds", str(THRESHOLDS),
    ]) == 0

    reg_dir = tmp_path / "regressed"
    assert main([
        "run", "--dataset", str(DATASET), "--config", str(REG_CFG),
        "--out", str(reg_dir), "--baseline", str(baseline_dir / "report.json"),
        "--thresholds", str(THRESHOLDS),
    ]) == 1


def test_demo_command(tmp_path, capsys):
    out_dir = tmp_path / "demo"
    assert main(["demo", "--out", str(out_dir)]) == 0
    assert (out_dir / "baseline" / "report.json").exists()
    assert (out_dir / "regressed" / "report.json").exists()
    assert (out_dir / "history.json").exists()
    out = capsys.readouterr().out
    assert "gate vs baseline: FAILED" in out


def test_history_command(tmp_path, capsys):
    hist = tmp_path / "history.json"
    main(["run", "--dataset", str(DATASET), "--out", str(tmp_path / "o"),
          "--history", str(hist)])
    capsys.readouterr()
    assert main(["history", str(hist)]) == 0
    assert "Trend history" in capsys.readouterr().out


def test_missing_dataset_is_a_clean_error(capsys):
    assert main(["validate-dataset", "/nonexistent/nope.jsonl"]) == 2
    assert "error:" in capsys.readouterr().err
