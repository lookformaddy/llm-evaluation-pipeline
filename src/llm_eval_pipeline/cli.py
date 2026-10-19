"""Command line interface.

Subcommands
-----------
``validate-dataset``  Validate a JSONL golden dataset and print its
                      version fingerprint.
``run``               Execute an evaluation, write JSON/Markdown/HTML
                      reports, optionally gate against a baseline report,
                      and append to the trend history. Exit code is 1
                      when the gate fails — that is what makes it CI-grade.
``demo``              Self-contained demo: baseline run (gate passes
                      against itself), then a deliberately regressed
                      SUT configuration (gate fails). No API keys needed.
``history``           Print the trend history as Markdown.

Examples
--------
    llm-eval validate-dataset data/golden/support_qa_v1.jsonl
    llm-eval run --dataset data/golden/support_qa_v1.jsonl \
        --config examples/sut_config_baseline.json --out out/
    llm-eval demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import report as report_mod
from .dataset import build_manifest, content_hash, load_dataset
from .gating import compare_runs, exit_code
from .history import append_history, entry_from_report, load_history
from .runner import EvalRunner
from .schemas import RunReport, SUTConfig, Thresholds
from .scorers import default_scorers
from .sut import KnowledgeBaseSUT

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_KB = PACKAGE_ROOT.parents[1] / "data" / "kb.json"
DEFAULT_DATASET = PACKAGE_ROOT.parents[1] / "data" / "golden" / "support_qa_v1.jsonl"


def _load_config(path: str | None, name_override: str | None = None) -> SUTConfig:
    if path:
        config = SUTConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))
    else:
        config = SUTConfig()
    if name_override:
        config = config.model_copy(update={"name": name_override})
    return config


def _run_eval(args) -> tuple[RunReport, object | None]:
    cases = load_dataset(args.dataset)
    manifest = build_manifest(
        cases,
        name=args.dataset_name,
        version=args.dataset_version,
        description="golden dataset",
    )
    config = _load_config(getattr(args, "config", None))
    kb_path = Path(args.kb) if getattr(args, "kb", None) else DEFAULT_KB
    sut = KnowledgeBaseSUT.from_file(kb_path)
    runner = EvalRunner(sut, default_scorers())
    report = runner.run(cases, manifest, config)

    gate = None
    if getattr(args, "baseline", None):
        baseline = RunReport.model_validate_json(
            Path(args.baseline).read_text(encoding="utf-8")
        )
        thresholds = Thresholds()
        if getattr(args, "thresholds", None):
            thresholds = Thresholds.model_validate_json(
                Path(args.thresholds).read_text(encoding="utf-8")
            )
        gate = compare_runs(report, baseline, thresholds)
    return report, gate


def _write_outputs(report: RunReport, gate, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(
        report_mod.render_markdown(report, gate), encoding="utf-8"
    )
    (out_dir / "report.html").write_text(
        report_mod.render_html(report, gate), encoding="utf-8"
    )


def cmd_validate_dataset(args) -> int:
    cases = load_dataset(args.dataset)
    print(f"OK: {len(cases)} cases")
    print(f"content_hash: {content_hash(cases)}")
    tag_keys = sorted({k for c in cases for k in c.tags})
    print(f"slice keys: {', '.join(tag_keys) or '(none)'}")
    return 0


def cmd_run(args) -> int:
    report, gate = _run_eval(args)
    out_dir = Path(args.out)
    _write_outputs(report, gate, out_dir)
    if args.history:
        append_history(args.history, entry_from_report(report, gate))
    m = report.metrics
    print(f"run {report.run_id}: {m.case_count} cases, overall mean {m.overall_mean:.3f}")
    for name, agg in sorted(m.per_scorer.items()):
        print(f"  {name:<22} mean {agg.mean:.3f}  pass {agg.pass_rate:.0%}")
    print(f"  latency p95 {m.latency_p95_ms:.0f} ms · total cost ${m.total_cost_usd:.4f}")
    print(f"reports written to {out_dir}/ (report.json, report.md, report.html)")
    if gate is not None:
        print(f"GATE: {'PASSED' if gate.passed else 'FAILED'}")
        for check in gate.failures:
            print(f"  FAIL {check.name}: {check.message}")
    return exit_code(gate)


def cmd_demo(args) -> int:
    workdir = Path(args.out)
    workdir.mkdir(parents=True, exist_ok=True)
    cases = load_dataset(DEFAULT_DATASET)
    manifest = build_manifest(cases, "support-qa", "1.0.0", "demo golden dataset")
    sut = KnowledgeBaseSUT.from_file(DEFAULT_KB)
    runner = EvalRunner(sut, default_scorers())

    thresholds = Thresholds(
        min_scores={"groundedness": 0.8, "semantic_similarity": 0.6},
        default_max_drop=0.05,
    )

    print("=== Demo 1/2: baseline configuration ===")
    baseline_cfg = SUTConfig(name="baseline", options={"max_facts": 3})
    baseline = runner.run(cases, manifest, baseline_cfg, run_id="demo-baseline")
    gate_ok = compare_runs(baseline, baseline, thresholds)
    _write_outputs(baseline, gate_ok, workdir / "baseline")
    append_history(workdir / "history.json", entry_from_report(baseline, gate_ok))
    print(f"overall mean {baseline.metrics.overall_mean:.3f} — gate vs itself: "
          f"{'PASSED' if gate_ok.passed else 'FAILED'}")

    print("\n=== Demo 2/2: regressed configuration "
          "(fewer facts, citations dropped, retrieval noise) ===")
    regressed_cfg = SUTConfig(
        name="regressed",
        options={"max_facts": 1, "drop_citations": True, "retrieval_noise": True},
    )
    regressed = runner.run(cases, manifest, regressed_cfg, run_id="demo-regressed")
    gate_bad = compare_runs(regressed, baseline, thresholds)
    _write_outputs(regressed, gate_bad, workdir / "regressed")
    append_history(workdir / "history.json", entry_from_report(regressed, gate_bad))
    print(f"overall mean {regressed.metrics.overall_mean:.3f} — gate vs baseline: "
          f"{'PASSED' if gate_bad.passed else 'FAILED'}")
    for check in gate_bad.failures:
        print(f"  FAIL {check.name}: {check.message}")
    print(f"\nArtifacts in {workdir}/ (baseline/, regressed/, history.json)")
    # The demo succeeds when it demonstrates both outcomes.
    return 0 if (gate_ok.passed and not gate_bad.passed) else 1


def cmd_history(args) -> int:
    entries = load_history(args.history)
    print(report_mod.render_history_markdown(entries))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate-dataset", help="validate a JSONL golden dataset")
    p_val.add_argument("dataset")
    p_val.set_defaults(func=cmd_validate_dataset)

    p_run = sub.add_parser("run", help="run an evaluation and write reports")
    p_run.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p_run.add_argument("--dataset-name", default="support-qa")
    p_run.add_argument("--dataset-version", default="1.0.0")
    p_run.add_argument("--config", help="SUT config JSON (default: built-in baseline)")
    p_run.add_argument("--kb", help="knowledge base JSON (default: bundled demo KB)")
    p_run.add_argument("--out", default="out")
    p_run.add_argument("--baseline", help="baseline report.json to gate against")
    p_run.add_argument("--thresholds", help="thresholds JSON for the gate")
    p_run.add_argument("--history", help="trend history JSON to append to")
    p_run.set_defaults(func=cmd_run)

    p_demo = sub.add_parser("demo", help="baseline vs regressed demo")
    p_demo.add_argument("--out", default="demo-out")
    p_demo.set_defaults(func=cmd_demo)

    p_hist = sub.add_parser("history", help="print trend history")
    p_hist.add_argument("history")
    p_hist.set_defaults(func=cmd_history)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
