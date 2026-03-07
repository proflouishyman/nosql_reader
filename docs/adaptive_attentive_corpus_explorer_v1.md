# Adaptive Attentive Corpus Explorer
## Implementation Plan v1.1 (Build Baseline)

Date: 2026-03-04
Branch: `codex/feature-incremental-explorer`
Mode flag: `TIER0_NOTEBOOK_MODE=legacy|adaptive`
Current implementation reference: `/Users/louishyman/coding/nosql/nosql_reader/docs/adaptive_explorer_current_logic_v5_3.md`

> Note (2026-03-07): This document remains the baseline implementation plan.
> The current branch behavior now includes additional implemented logic not present
> in this original baseline (hierarchy stabilization, evidence-snippet drilldown,
> run comparison API/UI, and graph-aware benchmark rubric). See the reference doc above.

---

## 1. Scope and Objectives

This implementation adds an adaptive question-graph notebook that supports:

1. Seed-first (`origin=seed`) and emergent (`origin=emergent`) question growth in parallel.
2. Per-batch attentive reading driven by active hypotheses.
3. Periodic question defragmentation (dedupe, clustering, promotion, change/continuity detection).
4. Evidence-aware synthesis that explicitly surfaces contested/high-tension findings.

Backward compatibility is required:

1. Legacy Tier 0 flow remains default.
2. Existing API route remains `/api/rag/explore_corpus`.
3. Existing report fields remain present; adaptive fields are additive.

---

## 2. Final Build Corrections (Required)

These corrections are mandatory and are incorporated into implementation.

1. Defrag budget metrics capture order:
- Capture `budget.used` and `budget.skipped` into local variables before calling `budget.reset()`.

2. Non-mutating decision logging:
- `RelationDecisionLog` must not call `budget.request(...)` inside logging.
- Budget skip booleans are computed in decision flow and passed into log objects.

3. Merge dedupe strength rule:
- For duplicate evidence keys, keep stronger link (`strength` max), not first-seen link.

4. Canonical pointer for absorbed nodes:
- Add `canonical_node_id` to `QuestionNode` (or equivalent lookup map) so evidence routing to canonical owner is deterministic.

5. Seed decay unit alignment:
- Compare seed decay against docs-read unit, not defrag interval index.
- If interval-based behavior is desired, use separate config variable.

6. Consistent LLM response contract:
- Use the codebase-standard `response = llm.generate(...)`, check `response.success`, parse `response.content`.
- Avoid direct parse of raw response object.

---

## 3. Files

### New files

1. `app/historian_agent/question_graph.py`
2. `app/historian_agent/relation_engine.py`
3. `app/historian_agent/defrag_engine.py`
4. `app/historian_agent/incremental_explorer.py`

### Modified files

1. `app/config.py`
2. `app/routes.py`
3. `app/historian_agent/research_notebook.py`
4. `app/historian_agent/question_synthesis.py`
5. `app/templates/corpus_explorer.html`
6. `app/static/corpus_explorer.js`
7. `SOLUTIONS.MD`

---

## 4. Data Contracts

### QuestionNode additions

Required fields:

1. `origin`
2. `question_type`
3. `tension_score`
4. `evidence_owner`
5. `canonical_node_id`
6. `schema_version` on graph container

Rules:

1. `what` question type allowed only at `micro`.
2. Meso/macro promotions with `what` are skipped, not fatal.
3. Absorbed node must have `evidence_owner=False` and `canonical_node_id=<owner_id>`.

### EvidenceLink uniqueness

Enforced key:

1. `(question_id, doc_id, block_id, evidence_type)`

Dedup behavior:

1. If duplicate key exists, keep higher `strength`.

---

## 5. Runtime Control and Resilience

1. Add weighted per-interval LLM budget (`light=1`, `medium=2`, `heavy=3`).
2. All adaptive LLM calls must have timeout tiers from config.
3. All LLM failures/timeouts degrade gracefully (skip/retry later; no run abort).
4. Defrag and relation engines must be safe in budget-exhausted mode.

---

## 6. Route and UI Behavior

1. `/api/rag/explore_corpus` accepts `mode` payload field.
2. `mode=adaptive` uses incremental explorer singleton.
3. Existing strategies remain; new UI option `Attentive Reading (experimental)` sets `mode=adaptive`.
4. If adaptive mode is selected from UI, base sampling strategy defaults to `balanced` unless user specifies one.

---

## 7. Implementation Phases

1. Phase 1: Graph models + schema version + merge invariants + serialization.
2. Phase 2: Relation engine + decision logs + why/how enforcement + seed guard.
3. Phase 3: Defrag engine + LLMBudget + promotion/change-continuity logic.
4. Phase 4: Incremental explorer orchestration and notebook integration.
5. Phase 5: Route/UI wiring and report payload extension.
6. Phase 6: Synthesis integration from graph (including tension summaries).
7. Phase 7: Test matrix and comparative runs.

---

## 8. Testing Plan

### Unit tests

1. Merge invariants and canonical routing.
2. Evidence dedupe with higher-strength retention.
3. Seed guard behavior (material seed advantage required).
4. Why/how enforcement fallback behavior.
5. Defrag budget accounting and no-post-reset metric loss.
6. Schema migration from older graph payloads.

### Integration runs

1. Adaptive 10-doc smoke run.
2. Adaptive 100-doc run.
3. Legacy 100-doc run (same scope and strategy) for comparison.
4. Adaptive 500-doc run.

### Comparison metrics (100-doc)

1. Runtime seconds.
2. Total questions and by-level counts.
3. Multi-axis macro question share.
4. Factoid leakage above micro level.
5. High-tension node count.
6. Seed status counts (confirmed/complicated/unconfirmed).

---

## 9. Acceptance Criteria

1. Zero meso/macro `what` questions.
2. No evidence attached to non-canonical absorbed nodes.
3. Decision log produced for all non-trivial relation actions.
4. Defrag budget metrics are non-zero and accurate when calls occur.
5. Seed questions are explicitly retained and status-labeled.
6. Adaptive runtime for 100-doc and 500-doc runs completes without exceptions.
7. Legacy mode remains unaffected.

---

## 10. Rollback and Compatibility

1. `TIER0_NOTEBOOK_MODE=legacy` remains default.
2. Adaptive-specific payload keys are additive only.
3. If adaptive fails at runtime, route returns error details without mutating legacy state.
4. Stored adaptive graph includes `schema_version` and migration path.
