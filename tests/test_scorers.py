from __future__ import annotations

from llm_eval_pipeline.schemas import CaseOutput, EvalCase
from llm_eval_pipeline.scorers import (
    ExactMatchScorer,
    GroundednessScorer,
    JudgeScorer,
    OfflineRubricJudge,
    RegexMatchScorer,
    SemanticSimilarityScorer,
)


def out(text: str, citations=None) -> CaseOutput:
    return CaseOutput(case_id="c", output_text=text, citations=citations or [])


CASE = EvalCase(
    id="c",
    input="How long do I have to return an item?",
    expected_output="You can return any unused item within 30 days of delivery for a full refund.",
    expected_patterns=["30 days", "refund"],
    reference_facts=[
        "return any unused item within 30 days of delivery",
        "full refund to the original payment method",
    ],
    available_docs=["returns-policy"],
)

GOOD = (
    "Return any unused item within 30 days of delivery. "
    "Full refund to the original payment method. [doc:returns-policy]"
)


# --- exact match -----------------------------------------------------------

def test_exact_match_pass_and_normalization():
    s = ExactMatchScorer()
    exact = out("  you can return any unused item within 30 days of delivery "
                "for a full refund  ")
    assert s.score(CASE, exact).passed
    assert s.score(CASE, out("something else")).score == 0.0


# --- regex -----------------------------------------------------------------

def test_regex_match_full_and_partial():
    s = RegexMatchScorer()
    assert s.score(CASE, out(GOOD)).score == 1.0
    partial = s.score(CASE, out("You have 30 days."))
    assert partial.score == 0.5 and not partial.passed


def test_regex_match_no_patterns_is_pass():
    case = EvalCase(id="c2", input="q")
    assert RegexMatchScorer().score(case, out("anything")).passed


# --- semantic similarity ---------------------------------------------------

def test_semantic_similarity_paraphrase_vs_unrelated():
    s = SemanticSimilarityScorer()
    para = s.score(CASE, out("Items may be returned within 30 days of delivery "
                             "and you will receive a full refund."))
    unrel = s.score(CASE, out("Loyalty points expire after twelve months."))
    assert para.score > unrel.score
    assert para.passed and not unrel.passed


# --- groundedness ----------------------------------------------------------

def test_groundedness_full_coverage_passes():
    res = GroundednessScorer().score(CASE, out(GOOD, ["returns-policy"]))
    assert res.score == 1.0 and res.passed
    assert res.details["facts_covered"] == 2


def test_groundedness_missing_fact_lowers_score():
    res = GroundednessScorer().score(
        CASE, out("Return any unused item within 30 days of delivery.",
                  ["returns-policy"])
    )
    assert res.score == 0.5
    assert res.details["missing_facts"]


def test_groundedness_hallucinated_citation_penalized():
    res = GroundednessScorer().score(CASE, out(GOOD, ["made-up-doc"]))
    assert res.score == 0.5
    assert not res.passed
    assert any("hallucinated" in p for p in res.details["penalties"])


def test_groundedness_no_citation_penalty():
    res = GroundednessScorer().score(CASE, out(GOOD, []))
    assert res.score == 0.8
    assert "no_citations" in res.details["penalties"]


# --- judge -----------------------------------------------------------------

def test_offline_judge_is_deterministic():
    judge = OfflineRubricJudge()
    a = judge.judge(CASE, out(GOOD, ["returns-policy"]), ["relevance", "completeness"])
    b = judge.judge(CASE, out(GOOD, ["returns-policy"]), ["relevance", "completeness"])
    assert a.dimensions == b.dimensions
    assert set(a.dimensions) == {"relevance", "completeness"}


def test_judge_scorer_good_beats_bad():
    s = JudgeScorer()
    good = s.score(CASE, out(GOOD, ["returns-policy"]))
    bad = s.score(CASE, out("Banana banana banana."))
    assert good.score > bad.score
    assert good.details["judge"] == "OfflineRubricJudge"
    assert "dimensions" in good.details


def test_judge_scorer_accepts_custom_judge():
    class FixedJudge:
        def judge(self, case, output, rubric):
            from llm_eval_pipeline.scorers import JudgeVerdict

            return JudgeVerdict({d: 0.9 for d in rubric}, "fixed")

    res = JudgeScorer(judge=FixedJudge()).score(CASE, out("whatever"))
    assert res.score == 0.9 and res.passed
