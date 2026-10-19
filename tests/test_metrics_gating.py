from __future__ import annotations

from llm_eval_pipeline import SUTConfig, Thresholds, compare_runs
from llm_eval_pipeline.metrics import _percentile

BASELINE_CFG = SUTConfig(name="baseline", options={"max_facts": 3})
REGRESSED_CFG = SUTConfig(
    name="regressed",
    options={"max_facts": 1, "drop_citations": True, "retrieval_noise": True},
)


def test_percentile_helper():
    assert _percentile([], 95) == 0.0
    assert _percentile([10.0], 50) == 10.0
    assert _percentile([0.0, 100.0], 50) == 50.0
    assert _percentile([0.0, 100.0], 100) == 100.0


def test_run_metrics_shape(runner, cases, manifest):
    report = runner.run(cases, manifest, BASELINE_CFG)
    m = report.metrics
    assert m.case_count == 12
    assert set(m.per_scorer) == {
        "regex_match", "semantic_similarity", "groundedness", "llm_judge"
    }
    assert m.error_count == 0
    assert m.total_cost_usd > 0 and m.latency_p95_ms >= m.latency_p50_ms
    # Baseline configuration should be strong on the bundled golden set.
    assert m.per_scorer["groundedness"].mean >= 0.9
    assert m.per_scorer["regex_match"].mean >= 0.9


def test_slice_breakdowns(runner, cases, manifest):
    report = runner.run(cases, manifest, BASELINE_CFG)
    keys = {s.slice_key for s in report.metrics.slices}
    assert "category=returns" in keys and "difficulty=easy" in keys
    returns = next(s for s in report.metrics.slices if s.slice_key == "category=returns")
    assert returns.case_count == 2
    assert "groundedness" in returns.per_scorer


def test_gate_passes_identical_run(runner, cases, manifest):
    report = runner.run(cases, manifest, BASELINE_CFG)
    gate = compare_runs(report, report, Thresholds(default_max_drop=0.05))
    assert gate.passed


def test_gate_fails_on_regression(runner, cases, manifest):
    baseline = runner.run(cases, manifest, BASELINE_CFG)
    regressed = runner.run(cases, manifest, REGRESSED_CFG)
    gate = compare_runs(regressed, baseline, Thresholds(default_max_drop=0.05))
    assert not gate.passed
    failed_names = {c.name for c in gate.failures}
    assert any(n.startswith("groundedness") for n in failed_names)


def test_gate_min_score_failure(runner, cases, manifest):
    regressed = runner.run(cases, manifest, REGRESSED_CFG)
    gate = compare_runs(
        regressed, None, Thresholds(min_scores={"groundedness": 0.95})
    )
    assert not gate.passed
    assert any(c.name == "groundedness:min_score" for c in gate.failures)


def test_gate_latency_and_cost_caps(runner, cases, manifest):
    report = runner.run(cases, manifest, BASELINE_CFG)
    gate = compare_runs(
        report,
        None,
        Thresholds(max_latency_p95_ms=1.0, max_total_cost_usd=0.0001),
    )
    assert not gate.passed
    names = {c.name for c in gate.failures}
    assert names == {"latency_p95:max", "total_cost:max"}


def test_regressed_run_scores_lower(runner, cases, manifest):
    baseline = runner.run(cases, manifest, BASELINE_CFG)
    regressed = runner.run(cases, manifest, REGRESSED_CFG)
    assert regressed.metrics.overall_mean < baseline.metrics.overall_mean
    assert (
        regressed.metrics.per_scorer["groundedness"].mean
        < baseline.metrics.per_scorer["groundedness"].mean
    )
