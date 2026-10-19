"""Example: wrap ANY existing LLM application as a SystemUnderTest.

The pipeline only needs the ``SystemUnderTest`` protocol. This example
wraps a plain Python function (standing in for "call our support bot
service") and fills latency/cost accounting from the app response —
exactly how a team would adapt a LangChain/LlamaIndex/home-grown app.

    python examples/custom_sut_example.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_eval_pipeline import EvalRunner, build_manifest, load_dataset  # noqa: E402
from llm_eval_pipeline.schemas import CaseOutput, EvalCase, SUTConfig  # noqa: E402
from llm_eval_pipeline.scorers import default_scorers  # noqa: E402
from llm_eval_pipeline.sut import count_tokens  # noqa: E402


def my_existing_app(prompt: str) -> dict:
    """Stand-in for a real app call. Returns text + citations."""
    # ... in production: call your chain/agent/service here ...
    return {
        "text": "You can return any unused item within 30 days of delivery "
                "for a full refund. [doc:returns-policy]",
        "citations": ["returns-policy"],
    }


class MyAppSUT:
    name = "my-existing-app"

    def run(self, case: EvalCase, config: SUTConfig) -> CaseOutput:
        start = time.perf_counter()
        resp = my_existing_app(case.input)
        latency_ms = (time.perf_counter() - start) * 1000.0
        in_tok = count_tokens(case.input)
        out_tok = count_tokens(resp["text"])
        cost = (
            in_tok / 1000 * config.input_cost_per_1k_tokens
            + out_tok / 1000 * config.output_cost_per_1k_tokens
        )
        return CaseOutput(
            case_id=case.id,
            output_text=resp["text"],
            citations=resp["citations"],
            latency_ms=round(latency_ms, 3),
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=round(cost, 6),
        )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cases = load_dataset(root / "data" / "golden" / "support_qa_v1.jsonl")[:2]
    manifest = build_manifest(cases, "support-qa", "1.0.0")
    runner = EvalRunner(MyAppSUT(), default_scorers())
    report = runner.run(cases, manifest, SUTConfig(name="my-app"))
    for r in report.case_results:
        print(r.case.id, "->", {k: round(s.score, 3) for k, s in r.scores.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
