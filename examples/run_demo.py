"""Programmatic usage example: run the pipeline from Python code.

    python examples/run_demo.py

Mirrors `llm-eval demo` but shows the API a team would embed in its own
tooling: build a runner, execute, gate, render.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_eval_pipeline import (  # noqa: E402
    EvalRunner,
    KnowledgeBaseSUT,
    SUTConfig,
    Thresholds,
    build_manifest,
    compare_runs,
    default_scorers,
    load_dataset,
)
from llm_eval_pipeline.report import render_markdown  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    cases = load_dataset(ROOT / "data" / "golden" / "support_qa_v1.jsonl")
    manifest = build_manifest(cases, "support-qa", "1.0.0")
    sut = KnowledgeBaseSUT.from_file(ROOT / "data" / "kb.json")
    runner = EvalRunner(sut, default_scorers())

    baseline = runner.run(cases, manifest, SUTConfig(name="baseline"))
    regressed = runner.run(
        cases,
        manifest,
        SUTConfig(
            name="regressed",
            options={"max_facts": 1, "drop_citations": True},
        ),
    )
    thresholds = Thresholds(
        min_scores={"groundedness": 0.8}, default_max_drop=0.05
    )
    gate = compare_runs(regressed, baseline, thresholds)

    print(f"baseline overall:  {baseline.metrics.overall_mean:.3f}")
    print(f"regressed overall: {regressed.metrics.overall_mean:.3f}")
    print(f"gate: {'PASSED' if gate.passed else 'FAILED'}")
    for failure in gate.failures:
        print(f"  FAIL {failure.name}: {failure.message}")

    out = ROOT / "out" / "example-report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(baseline), encoding="utf-8")
    print(f"baseline report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
