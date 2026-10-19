"""Aggregate metrics and slice breakdowns for a run."""

from __future__ import annotations

from collections import defaultdict

from .schemas import (
    AggregateMetrics,
    RunReport,
    ScorerAggregate,
    SliceAggregate,
)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = pct / 100.0 * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def aggregate(report: RunReport) -> AggregateMetrics:
    results = report.case_results
    n = len(results)

    per_scorer: dict[str, ScorerAggregate] = {}
    scorer_names = sorted({name for r in results for name in r.scores})
    for name in scorer_names:
        vals = [r.scores[name].score for r in results if name in r.scores]
        if not vals:
            continue
        per_scorer[name] = ScorerAggregate(
            mean=sum(vals) / len(vals),
            min=min(vals),
            max=max(vals),
            pass_rate=sum(
                1 for r in results if name in r.scores and r.scores[name].passed
            )
            / len(vals),
        )

    overall = (
        sum(r.mean_score for r in results) / n if n else 0.0
    )

    # Slice breakdown: group by every tag key present on any case.
    slice_groups: dict[str, list] = defaultdict(list)
    for r in results:
        for key, value in sorted(r.case.tags.items()):
            slice_groups[f"{key}={value}"].append(r)
    slices: list[SliceAggregate] = []
    for slice_key in sorted(slice_groups):
        group = slice_groups[slice_key]
        per: dict[str, float] = {}
        for name in scorer_names:
            vals = [r.scores[name].score for r in group if name in r.scores]
            if vals:
                per[name] = round(sum(vals) / len(vals), 4)
        slices.append(
            SliceAggregate(
                slice_key=slice_key,
                case_count=len(group),
                mean_score=round(sum(r.mean_score for r in group) / len(group), 4),
                per_scorer=per,
            )
        )

    latencies = [r.output.latency_ms for r in results]
    total_cost = sum(r.output.cost_usd for r in results)
    return AggregateMetrics(
        case_count=n,
        overall_mean=round(overall, 4),
        per_scorer={
            k: ScorerAggregate(
                mean=round(v.mean, 4), min=round(v.min, 4),
                max=round(v.max, 4), pass_rate=round(v.pass_rate, 4),
            )
            for k, v in per_scorer.items()
        },
        slices=slices,
        latency_p50_ms=round(_percentile(latencies, 50), 3),
        latency_p95_ms=round(_percentile(latencies, 95), 3),
        latency_mean_ms=round(sum(latencies) / n, 3) if n else 0.0,
        total_cost_usd=round(total_cost, 6),
        mean_cost_usd=round(total_cost / n, 6) if n else 0.0,
        error_count=sum(1 for r in results if r.output.error),
    )
