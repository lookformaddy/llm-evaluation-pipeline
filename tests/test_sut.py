from __future__ import annotations

from llm_eval_pipeline.schemas import EvalCase, SUTConfig

RETURNS_CASE = EvalCase(
    id="t-returns",
    input="How many days do I have to return an unused item?",
    available_docs=["returns-policy", "shipping-info"],
    reference_facts=["return any unused item within 30 days of delivery"],
)


def test_baseline_output_is_grounded_and_cited(sut):
    out = sut.run(RETURNS_CASE, SUTConfig())
    assert "30 days" in out.output_text
    assert out.citations == ["returns-policy"]
    assert "[doc:returns-policy]" in out.output_text
    assert [t.step for t in out.trace] == ["retrieve", "compose", "accounting"]


def test_output_is_deterministic(sut):
    a = sut.run(RETURNS_CASE, SUTConfig(name="a"))
    b = sut.run(RETURNS_CASE, SUTConfig(name="b"))
    assert a.output_text == b.output_text
    assert a.latency_ms == b.latency_ms
    assert a.cost_usd == b.cost_usd


def test_accounting_is_positive_and_consistent(sut):
    out = sut.run(RETURNS_CASE, SUTConfig())
    assert out.input_tokens > 0 and out.output_tokens > 0
    assert out.cost_usd > 0 and out.latency_ms > 0


def test_drop_citations_knob(sut):
    cfg = SUTConfig(options={"drop_citations": True})
    out = sut.run(RETURNS_CASE, cfg)
    assert out.citations == []
    assert "[doc:" not in out.output_text


def test_max_facts_knob_limits_answer(sut):
    one = sut.run(RETURNS_CASE, SUTConfig(options={"max_facts": 1}))
    three = sut.run(RETURNS_CASE, SUTConfig(options={"max_facts": 3}))
    assert len(one.output_text) < len(three.output_text)


def test_retrieval_noise_demotes_best_doc(sut):
    noisy = sut.run(RETURNS_CASE, SUTConfig(options={"retrieval_noise": True}))
    assert noisy.citations == ["shipping-info"]


def test_unknown_question_fallback(sut):
    case = EvalCase(id="t-unknown", input="Do you sell gift wrapping?",
                    available_docs=[])
    out = sut.run(case, SUTConfig())
    assert "don't have information" in out.output_text
    assert out.citations == []


def test_empty_kb_rejected():
    from llm_eval_pipeline import KnowledgeBaseSUT

    try:
        KnowledgeBaseSUT({})
    except ValueError:
        return
    raise AssertionError("empty KB should be rejected")
