"""Systems under test (SUT).

A SUT is anything implementing the :class:`SystemUnderTest` protocol:
given an :class:`EvalCase` and an :class:`SUTConfig`, it returns a
:class:`CaseOutput` with the produced text, citations, a step trace, and
latency/cost accounting.

Ships with :class:`KnowledgeBaseSUT` — a deterministic, offline
retrieval-and-compose assistant over a small synthetic knowledge base.
It needs no API keys, which makes the whole pipeline (and its test
suite) reproducible anywhere, including CI. Its config knobs
(``max_facts``, ``drop_citations``, ``retrieval_noise``) exist so teams
can model realistic regressions — a prompt change that truncates
answers, a retrieval change that drops citations — and watch the gate
react.

To evaluate a real LLM application, wrap it in this protocol: call your
app, map its response into ``CaseOutput`` (filling tokens/latency/cost
from the provider response), and the rest of the pipeline is unchanged.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .embedder import tokenize
from .schemas import CaseOutput, EvalCase, SUTConfig, TraceEvent


def count_tokens(text: str) -> int:
    """Cheap deterministic token approximation (~4 chars/token, min 1)."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


@runtime_checkable
class SystemUnderTest(Protocol):
    name: str

    def run(self, case: EvalCase, config: SUTConfig) -> CaseOutput:
        ...


class KnowledgeBaseSUT:
    """Deterministic retrieval-and-compose assistant over a JSON KB.

    KB format: ``{doc_id: {"title": str, "text": str, "facts": [str]}}``.
    Retrieval ranks *allowed* docs by token overlap with the question;
    composition emits the top document's facts as the answer and cites
    the document.
    """

    name = "knowledge-base-assistant"

    def __init__(self, knowledge_base: dict[str, dict[str, Any]]) -> None:
        if not knowledge_base:
            raise ValueError("knowledge base must not be empty")
        self.kb = knowledge_base

    @classmethod
    def from_file(cls, path: str | Path) -> "KnowledgeBaseSUT":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data)

    # -- retrieval --------------------------------------------------------

    def _rank_docs(self, question: str, allowed: list[str]) -> list[tuple[str, int]]:
        """Rank allowed docs by token overlap; returns (doc_id, overlap)."""
        q_tokens = set(tokenize(question))
        scored: list[tuple[int, str]] = []
        for doc_id in allowed:
            doc = self.kb.get(doc_id)
            if doc is None:
                continue
            d_tokens = set(tokenize(doc["title"] + " " + doc["text"]))
            overlap = len(q_tokens & d_tokens)
            scored.append((overlap, doc_id))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [(doc_id, overlap) for overlap, doc_id in scored]

    # -- main entry point -------------------------------------------------

    def run(self, case: EvalCase, config: SUTConfig) -> CaseOutput:
        trace: list[TraceEvent] = []
        opts = config.options
        max_facts = int(opts.get("max_facts", 3))
        drop_citations = bool(opts.get("drop_citations", False))
        retrieval_noise = bool(opts.get("retrieval_noise", False))

        allowed = case.available_docs or list(self.kb.keys())
        ranked_pairs = self._rank_docs(case.input, allowed)
        # A document with zero token overlap is not a retrieval result:
        # the assistant abstains instead of answering from an unrelated doc.
        if ranked_pairs and ranked_pairs[0][1] == 0:
            ranked_pairs = []
        ranked = [doc_id for doc_id, _ in ranked_pairs]
        trace.append(
            TraceEvent(step="retrieve", detail=f"candidates={ranked}", latency_ms=18.0)
        )

        if retrieval_noise and len(ranked) > 1:
            # Model a retrieval regression: the best document is demoted.
            ranked = ranked[1:] + ranked[:1]
            trace.append(TraceEvent(step="retrieve_noise", detail="top doc demoted"))

        top_doc_id = ranked[0] if ranked else None
        facts: list[str] = []
        citations: list[str] = []
        if top_doc_id is not None:
            facts = list(self.kb[top_doc_id].get("facts", []))[:max_facts]
            if not drop_citations:
                citations = [top_doc_id]
            trace.append(
                TraceEvent(
                    step="compose",
                    detail=f"doc={top_doc_id} facts={len(facts)}",
                    latency_ms=42.0,
                )
            )

        if facts:
            body = " ".join(f.capitalize() + "." for f in facts)
        else:
            body = "I don't have information about that in the help center."
        output_text = body
        if citations:
            output_text = body + " " + " ".join(f"[doc:{c}]" for c in citations)

        input_tokens = count_tokens(case.input) + sum(
            count_tokens(self.kb[d]["text"]) for d in (ranked[:1])
        )
        output_tokens = count_tokens(output_text)
        latency_ms = (
            config.base_latency_ms
            + config.latency_ms_per_output_token * output_tokens
        )
        cost_usd = (
            input_tokens / 1000.0 * config.input_cost_per_1k_tokens
            + output_tokens / 1000.0 * config.output_cost_per_1k_tokens
        )
        trace.append(
            TraceEvent(
                step="accounting",
                detail=f"in={input_tokens} out={output_tokens} cost=${cost_usd:.5f}",
                latency_ms=latency_ms,
            )
        )

        return CaseOutput(
            case_id=case.id,
            output_text=output_text,
            citations=citations,
            latency_ms=round(latency_ms, 3),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost_usd, 6),
            trace=trace,
        )
