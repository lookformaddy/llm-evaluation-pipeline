"""Human-readable run reports: Markdown and self-contained HTML."""

from __future__ import annotations

import html

from .schemas import GateResult, HistoryEntry, RunReport


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_markdown(report: RunReport, gate: GateResult | None = None) -> str:
    m = report.metrics
    assert m is not None
    lines: list[str] = []
    lines.append(f"# Evaluation report — run `{report.run_id}`")
    lines.append("")
    lines.append(f"- **Dataset:** {report.dataset.name} v{report.dataset.version} "
                 f"(`{report.dataset.content_hash[:12]}…`, {m.case_count} cases)")
    lines.append(f"- **System under test:** {report.sut_config.name} "
                 f"({report.sut_config.model})")
    lines.append(f"- **Overall mean score:** {m.overall_mean:.3f}")
    lines.append(f"- **Latency:** p50 {m.latency_p50_ms:.1f} ms · "
                 f"p95 {m.latency_p95_ms:.1f} ms · mean {m.latency_mean_ms:.1f} ms")
    lines.append(f"- **Cost:** total ${m.total_cost_usd:.4f} · "
                 f"mean ${m.mean_cost_usd:.4f}/case")
    if m.error_count:
        lines.append(f"- **SUT errors:** {m.error_count}")
    lines.append("")

    lines.append("## Scorer summary")
    lines.append("")
    lines.append("| Scorer | Mean | Min | Max | Pass rate |")
    lines.append("|---|---|---|---|---|")
    for name, agg in sorted(m.per_scorer.items()):
        lines.append(
            f"| {name} | {agg.mean:.3f} | {agg.min:.3f} | {agg.max:.3f} | {agg.pass_rate:.0%} |"
        )
    lines.append("")

    if m.slices:
        lines.append("## Slice breakdown")
        lines.append("")
        lines.append("| Slice | Cases | Mean | " +
                     " | ".join(sorted(m.per_scorer)) + " |")
        lines.append("|---|---|---|" + "---|" * len(m.per_scorer))
        for s in m.slices:
            per = " | ".join(_fmt(s.per_scorer.get(name)) for name in sorted(m.per_scorer))
            lines.append(f"| {s.slice_key} | {s.case_count} | {s.mean_score:.3f} | {per} |")
        lines.append("")

    if gate is not None:
        lines.append("## Regression gate")
        lines.append("")
        lines.append(f"**Gate: {'PASSED' if gate.passed else 'FAILED'}**")
        lines.append("")
        lines.append("| Check | Result | Current | Baseline | Threshold |")
        lines.append("|---|---|---|---|---|")
        for c in gate.checks:
            lines.append(
                f"| {c.name} | {'✅ pass' if c.passed else '❌ FAIL'} | "
                f"{_fmt(c.current)} | {_fmt(c.baseline)} | {_fmt(c.threshold)} |"
            )
        lines.append("")

    lines.append("## Cases")
    lines.append("")
    lines.append("| Case | Mean | Output (truncated) | Failing scorers |")
    lines.append("|---|---|---|---|")
    for r in report.case_results:
        out = r.output.output_text.replace("|", "\\|").replace("\n", " ")
        if len(out) > 90:
            out = out[:87] + "…"
        failing = ", ".join(n for n, s in r.scores.items() if not s.passed) or "—"
        lines.append(f"| {r.case.id} | {r.mean_score:.3f} | {out} | {failing} |")
    lines.append("")
    return "\n".join(lines)


def render_html(report: RunReport, gate: GateResult | None = None) -> str:
    m = report.metrics
    assert m is not None
    scorer_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{agg.mean:.3f}</td>"
        f"<td>{agg.min:.3f}</td><td>{agg.max:.3f}</td><td>{agg.pass_rate:.0%}</td></tr>"
        for name, agg in sorted(m.per_scorer.items())
    )
    slice_rows = "".join(
        f"<tr><td>{html.escape(s.slice_key)}</td><td>{s.case_count}</td>"
        f"<td>{s.mean_score:.3f}</td></tr>"
        for s in m.slices
    )
    case_rows = "".join(
        "<tr>"
        f"<td>{html.escape(r.case.id)}</td>"
        f"<td>{r.mean_score:.3f}</td>"
        f"<td>{html.escape(r.output.output_text)}</td>"
        f"<td>{html.escape(', '.join(f'{n}: {s.score:.2f}' for n, s in sorted(r.scores.items())))}</td>"
        f"<td>{r.output.latency_ms:.0f} ms</td>"
        f"<td>${r.output.cost_usd:.4f}</td>"
        "</tr>"
        for r in report.case_results
    )
    gate_html = ""
    if gate is not None:
        checks = "".join(
            f"<tr><td>{html.escape(c.name)}</td>"
            f"<td class='{'ok' if c.passed else 'bad'}'>"
            f"{'PASS' if c.passed else 'FAIL'}</td>"
            f"<td colspan='3'>{html.escape(c.message)}</td></tr>"
            for c in gate.checks
        )
        gate_html = (
            f"<h2>Regression gate: {'PASSED' if gate.passed else 'FAILED'}</h2>"
            f"<table><tr><th>Check</th><th>Result</th><th>Detail</th></tr>{checks}</table>"
        )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Evaluation report {html.escape(report.run_id)}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1c2a25; }}
table {{ border-collapse: collapse; margin: 1rem 0; width: 100%; }}
th, td {{ border: 1px solid #ccd; padding: 0.35rem 0.6rem; text-align: left;
          vertical-align: top; font-size: 0.92rem; }}
th {{ background: #eef3f0; }}
.ok {{ color: #137333; font-weight: 600; }} .bad {{ color: #b3261e; font-weight: 600; }}
.meta {{ color: #555; }}
</style></head><body>
<h1>Evaluation report — run {html.escape(report.run_id)}</h1>
<p class="meta">Dataset: {html.escape(report.dataset.name)} v{html.escape(report.dataset.version)}
(hash {html.escape(report.dataset.content_hash[:12])}…) · SUT: {html.escape(report.sut_config.name)}
({html.escape(report.sut_config.model)}) · {m.case_count} cases</p>
<p><strong>Overall mean:</strong> {m.overall_mean:.3f} ·
<strong>Latency p50/p95:</strong> {m.latency_p50_ms:.1f} / {m.latency_p95_ms:.1f} ms ·
<strong>Total cost:</strong> ${m.total_cost_usd:.4f}</p>
<h2>Scorer summary</h2>
<table><tr><th>Scorer</th><th>Mean</th><th>Min</th><th>Max</th><th>Pass rate</th></tr>{scorer_rows}</table>
<h2>Slices</h2>
<table><tr><th>Slice</th><th>Cases</th><th>Mean</th></tr>{slice_rows}</table>
{gate_html}
<h2>Cases</h2>
<table><tr><th>Case</th><th>Mean</th><th>Output</th><th>Scores</th><th>Latency</th><th>Cost</th></tr>{case_rows}</table>
</body></html>
"""


def render_history_markdown(entries: list[HistoryEntry]) -> str:
    if not entries:
        return "# Trend history\n\nNo runs recorded yet.\n"
    lines = [
        "# Trend history",
        "",
        "| Recorded | Run | Dataset | SUT | Overall | p95 latency | Cost | Gate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        gate = "—" if e.gate_passed is None else ("✅" if e.gate_passed else "❌")
        lines.append(
            f"| {e.recorded_at:%Y-%m-%d %H:%M} | {e.run_id} | {e.dataset_name} "
            f"| {e.sut_name} | {e.overall_mean:.3f} | {e.latency_p95_ms:.0f} ms "
            f"| ${e.total_cost_usd:.4f} | {gate} |"
        )
    lines.append("")
    return "\n".join(lines)
