from __future__ import annotations

from llm_eval_pipeline import SUTConfig, Thresholds, compare_runs
from llm_eval_pipeline.history import append_history, entry_from_report, load_history
from llm_eval_pipeline.report import (
    render_history_markdown,
    render_html,
    render_markdown,
)


def _report(runner, cases, manifest):
    return runner.run(cases, manifest, SUTConfig(name="baseline"))


def test_markdown_report_sections(runner, cases, manifest):
    report = _report(runner, cases, manifest)
    gate = compare_runs(report, report, Thresholds())
    md = render_markdown(report, gate)
    assert "# Evaluation report" in md
    assert "## Scorer summary" in md
    assert "## Slice breakdown" in md
    assert "## Regression gate" in md
    assert "PASSED" in md
    assert "category=returns" in md
    assert "returns-window" in md


def test_html_report_is_self_contained(runner, cases, manifest):
    report = _report(runner, cases, manifest)
    page = render_html(report)
    assert page.startswith("<!DOCTYPE html>")
    assert "returns-window" in page
    assert "Scorer summary" in page
    assert "http" not in page.split("<style>")[0].replace("lang=", "")


def test_html_escapes_output(runner, cases, manifest):
    report = _report(runner, cases, manifest)
    report.case_results[0].output.output_text = "<script>alert(1)</script>"
    page = render_html(report)
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_history_roundtrip(tmp_path, runner, cases, manifest):
    report = _report(runner, cases, manifest)
    path = tmp_path / "history.json"
    assert load_history(path) == []
    append_history(path, entry_from_report(report))
    append_history(path, entry_from_report(report))
    entries = load_history(path)
    assert len(entries) == 2
    assert entries[0].dataset_name == "support-qa"
    assert entries[0].overall_mean == report.metrics.overall_mean
    md = render_history_markdown(entries)
    assert "Trend history" in md and report.run_id in md


def test_history_entry_records_gate(runner, cases, manifest):
    report = _report(runner, cases, manifest)
    gate = compare_runs(report, report, Thresholds())
    entry = entry_from_report(report, gate)
    assert entry.gate_passed is True
