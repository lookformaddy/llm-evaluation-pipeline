# Architecture

## Design goals

1. **Deterministic by default.** Given the same dataset version and SUT
   configuration, a run must produce the same scores. That is what makes
   a regression gate meaningful in CI. Randomness, wall-clock latency,
   and network calls are pushed to the edges (SUT adapters), and the
   bundled reference implementation has none of them.
2. **Everything validated.** Datasets, configs, reports, and thresholds
   are pydantic models. Bad input fails at the boundary with a precise
   error, never mid-run.
3. **Small protocols, many implementations.** Three protocols carry the
   whole design — `SystemUnderTest`, `Scorer`, `Judge` (plus `Embedder`
   for semantic scoring). Each can be replaced without touching the
   engine.
4. **Failure isolation.** A SUT exception on one case, or a scorer
   exception on one (case, scorer) pair, is recorded on that result and
   surfaced in metrics (`error_count`) — it never aborts the run. The
   gate treats SUT errors as an automatic failure.

## Data flow

```
JSONL file ──load_dataset──▶ list[EvalCase] ──content_hash──▶ DatasetManifest
                                                              │
SUTConfig (JSON) ────────────────────────────────────────────▶│
                                                              ▼
                                        EvalRunner.run(cases, manifest, config)
                                          for each case:
                                            SUT.run(case, config) ─▶ CaseOutput
                                                (+ trace, tokens, latency, cost)
                                            scorer.score(case, output) ─▶ ScoreResult × N
                                                              │
                                                              ▼
                                        metrics.aggregate ─▶ RunReport
                                                              │
                          baseline RunReport ─▶ gating.compare_runs ─▶ GateResult
                                                              │
                          report.render_markdown / render_html
                          history.append_history (JSON trend log)
```

## Key decisions

### Dataset versioning by content hash

A golden dataset is code: it is reviewed, edited, and debated. Naming
versions by hand ("v3-final-really") drifts from content. Instead,
`dataset.content_hash` computes a SHA-256 over the canonical JSONL
encoding of the parsed cases (sorted keys, normalized formatting).
Formatting-only changes do not create new versions; any semantic change
does. Every `RunReport` and `HistoryEntry` carries the hash, so any
number in any report can be traced to the exact dataset that produced
it.

### The offline SUT is a real (if small) RAG system

`KnowledgeBaseSUT` performs retrieval (token-overlap ranking over the
case's allowed documents, with abstention on zero overlap), composition
(fact selection), citation emission, and trace/accounting — the same
shape as a production RAG assistant, minus the LLM. Its config knobs
model the three most common real-world regression classes:

| Knob | Models |
|---|---|
| `max_facts` | Prompt/output truncation — answers get thinner |
| `drop_citations` | A template change that stops emitting sources |
| `retrieval_noise` | A retriever/embedding change that demotes the best document |

Because these knobs move specific scorers in predictable directions
(groundedness and judge first, semantic similarity with them), the demo
and the test suite can assert not just *that* the gate fails but *why*.

### Scorers are independent and composable

Each scorer sees only `(case, output)` and returns a `ScoreResult`
(score in [0, 1], pass flag, structured `details`). Aggregation never
mixes raw scorer internals; slices and gate rules operate on scorer
*names*, so adding a scorer (e.g. a toxicity classifier or a
format/schema checker) automatically gives it metrics, slices, and
gating with zero engine changes.

### Groundedness scoring

`GroundednessScorer` combines three signals a RAG team actually cares
about:

- **Coverage** — fraction of the case's reference facts whose content
  tokens appear in the output (≥70% token overlap counts as present).
- **Citation presence** — grounded facts with no citation are
  discounted (×0.8): unverifiable claims are weaker evidence.
- **Citation validity** — citing a document outside the case's allowed
  set is treated as a hallucinated citation (×0.5, and the case cannot
  pass): inventing sources is the failure mode this harness exists to
  catch.

### The judge interface, honestly stubbed

LLM-as-judge is powerful and dangerous: an uncalibrated judge is a
second unmonitored model in the loop. The pipeline therefore defines a
narrow `Judge` protocol (case + output + rubric → per-dimension grades
+ rationale) and ships `OfflineRubricJudge`, a deterministic heuristic
stub whose only job is to exercise the interface in CI. The README and
roadmap are explicit that production use means implementing a real
judge and calibrating it against human labels; the harness makes that
swap a one-class change.

### Reports and history are files, on purpose

Reports render to JSON (full fidelity, machine-consumable), Markdown
(code review, PR comments), and a single self-contained HTML file (no
external assets — safe to attach to a CI run or email). History is one
JSON array of run summaries: small enough to commit, diffable in
review, and trivially exportable to a warehouse later.

## Extension points

| To… | Implement / change |
|---|---|
| Evaluate a real app | A class with `run(case, config) -> CaseOutput` (see `examples/custom_sut_example.py`) |
| Use a real embedding model | The `Embedder` protocol, injected into `SemanticSimilarityScorer` |
| Use a real judge model | The `Judge` protocol, injected into `JudgeScorer` |
| Add a quality signal | A `Scorer` class; register it in your runner's scorer list |
| Gate differently | A `Thresholds` JSON; rules live in `gating.compare_runs` |
| Store history elsewhere | Replace `history.append_history` — entries are plain pydantic models |

## Non-goals

- **Load/performance testing** of LLM endpoints (latency here is a
  quality-adjacent signal, not a benchmark).
- **Human annotation workflows** (labeling UIs, adjudication) — datasets
  are files; author them wherever you like.
- **Online/production monitoring** — this harness is the pre-deployment
  gate; production drift monitoring is a sibling system.
