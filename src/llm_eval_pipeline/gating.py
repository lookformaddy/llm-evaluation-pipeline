"""Regression gating: compare a run against a baseline and decide CI.

The gate fails (exit code 1 in the CLI) when any of these hold:

* a scorer's mean is below its configured minimum (``min_scores``);
* a scorer's mean dropped versus baseline by more than its allowed
  ``max_drop`` (per-scorer or ``default_max_drop``);
* p95 latency exceeds ``max_latency_p95_ms``;
* total cost exceeds ``max_total_cost_usd``.
"""

from __future__ import annotations

from .schemas import GateCheck, GateResult, RunReport, Thresholds


def compare_runs(
    current: RunReport,
    baseline: RunReport | None,
    thresholds: Thresholds,
) -> GateResult:
    checks: list[GateCheck] = []
    cur = current.metrics
    assert cur is not None, "current run has no metrics; run metrics.aggregate first"

    base_means: dict[str, float] = {}
    if baseline is not None and baseline.metrics is not None:
        base_means = {
            name: agg.mean for name, agg in baseline.metrics.per_scorer.items()
        }

    for name, agg in sorted(cur.per_scorer.items()):
        minimum = thresholds.min_scores.get(name)
        if minimum is not None:
            checks.append(
                GateCheck(
                    name=f"{name}:min_score",
                    passed=agg.mean >= minimum,
                    current=agg.mean,
                    threshold=minimum,
                    message=(
                        f"{name} mean {agg.mean:.3f} "
                        f"{'>=' if agg.mean >= minimum else '<'} minimum {minimum:.3f}"
                    ),
                )
            )
        if baseline is not None and name in base_means:
            allowed = thresholds.max_drop.get(name, thresholds.default_max_drop)
            drop = base_means[name] - agg.mean
            checks.append(
                GateCheck(
                    name=f"{name}:max_drop",
                    passed=drop <= allowed,
                    current=agg.mean,
                    baseline=base_means[name],
                    threshold=allowed,
                    message=(
                        f"{name} changed {base_means[name]:.3f} -> {agg.mean:.3f} "
                        f"(drop {drop:.3f}, allowed {allowed:.3f})"
                    ),
                )
            )

    if thresholds.max_latency_p95_ms is not None:
        checks.append(
            GateCheck(
                name="latency_p95:max",
                passed=cur.latency_p95_ms <= thresholds.max_latency_p95_ms,
                current=cur.latency_p95_ms,
                threshold=thresholds.max_latency_p95_ms,
                message=f"latency p95 {cur.latency_p95_ms:.1f}ms vs max {thresholds.max_latency_p95_ms:.1f}ms",
            )
        )
    if thresholds.max_total_cost_usd is not None:
        checks.append(
            GateCheck(
                name="total_cost:max",
                passed=cur.total_cost_usd <= thresholds.max_total_cost_usd,
                current=cur.total_cost_usd,
                threshold=thresholds.max_total_cost_usd,
                message=f"total cost ${cur.total_cost_usd:.4f} vs max ${thresholds.max_total_cost_usd:.4f}",
            )
        )
    if cur.error_count:
        checks.append(
            GateCheck(
                name="sut_errors",
                passed=False,
                current=float(cur.error_count),
                message=f"{cur.error_count} case(s) errored in the SUT",
            )
        )

    return GateResult(passed=all(c.passed for c in checks), checks=checks)


def exit_code(gate: GateResult | None) -> int:
    if gate is None:
        return 0
    return 0 if gate.passed else 1
