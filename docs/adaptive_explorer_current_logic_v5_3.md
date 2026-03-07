# Adaptive Explorer Current Logic and Algorithms (v5.3)

Date: 2026-03-07  
Branch: `codex/feature-incremental-explorer`  
Status: Implemented on branch (adaptive mode)

---

## 1. Scope

This document captures the **currently implemented** adaptive explorer logic and algorithms, including the recent improvements for:

1. explicit macro->meso->micro hierarchy stabilization,
2. question-graph drill-down with evidence snippets,
3. run-to-run compare APIs/UI,
4. graph-aware benchmark rubric scoring.

It reflects behavior in:

1. `app/historian_agent/defrag_engine.py`
2. `app/historian_agent/incremental_explorer.py`
3. `app/rag_base.py`
4. `app/routes.py`
5. `app/static/corpus_explorer.js`
6. `scripts/benchmark_adaptive_prompt_variants.py`

---

## 2. End-to-End Adaptive Flow

High-level adaptive flow:

1. Consultation + brief normalization (lens, hypotheses, axes, sort order).
2. Seed graph initialization (`seed` + `emergent` node support).
3. Per-batch attentive reading with evidence linking.
4. Periodic defrag (dedupe, promotion, change/continuity, hierarchy stabilization).
5. Export graph summary + hierarchy tree + quality metrics.
6. UI drill-down renders macro->meso->micro and associated documents/evidence blocks.

---

## 3. Algorithm: Hierarchy Stabilization Pass

Location: `defrag_engine._stabilize_hierarchy_links(...)`

Problem addressed:

1. Graphs could hold evidence-rich nodes without explicit parent links across levels.
2. UI and traceability needed explicit canonical macro->meso->micro paths.

Algorithm:

1. Work only on canonical owner nodes (`evidence_owner=True`).
2. Build current canonical graph adjacency (`outgoing`, `incoming`).
3. Build direct evidence doc sets per node.
4. Compute scope-doc closure (node docs + descendant docs).
5. For each orphan meso (no macro parent):
   - score each macro candidate by `(shared_docs, overlap_ratio, lexical_similarity)`,
   - skip candidates that create cycles,
   - require either shared docs or lexical fallback threshold (`>= 0.55`),
   - add `decomposes_into` edge + decision log (`action="hierarchy_link"`).
6. For each orphan micro (no meso parent):
   - same scoring pattern against meso nodes,
   - lexical fallback threshold (`>= 0.50`),
   - add canonical hierarchy edge + decision log.

Safety/contract details:

1. Cycle prevention is enforced before adding any edge.
2. Confidence is bounded (`0.55` to `0.95`) for inferred links.
3. Every inferred link is auditable in `decision_log`.

---

## 4. Algorithm: Question Graph Tree Builder

Location: `incremental_explorer._build_graph_tree(...)`

Purpose:

1. Build a UI-ready hierarchical tree from canonical nodes and evidence coverage.
2. Preserve evidence traceability to document/block level.

Algorithm:

1. Build canonical node maps and explicit canonical edges.
2. Build direct-doc sets per node from `EvidenceLink`.
3. Compute scope docs via descendant closure.
4. Select macro nodes, ranked by scope evidence size.
5. For each macro:
   - select meso candidates by macro/meso shared docs,
   - keep overlap counts/ratios and explicit-link flag.
6. For each macro+meso pair:
   - select micro candidates by triple overlap (macro ∩ meso ∩ micro-direct),
   - emit shared doc ids and associated document rows.
7. For each micro document row:
   - include `doc_id`, filename/source metadata,
   - include `block_ids`,
   - include `block_details` (block id + snippet),
   - include evidence types.

Output shape:

1. `question_graph.tree` = list of macro entries
2. each macro has `meso[]`
3. each meso has `micro[]`
4. each micro has `documents[]`

---

## 5. Algorithm: Evidence Block Snippet Resolution

Location: `rag_base.DocumentStore.fetch_block_snippets(...)`

Purpose:

1. Turn block ids (e.g., `doc_id::b12`) into short evidence snippets for UI drill-down.

Lookup cascade:

1. Parse block ids into `(doc_id, block_index)`.
2. Query chunk collection by `document_id` once per doc set.
3. If `block_index` exists, prefer matching chunk index.
4. Fallback to first chunk for that document.
5. Fallback to parent document text fields (`ocr_text`, `content`, `text`, `summary`).
6. Normalize whitespace and clip to max chars (`320` default).

Contract:

1. Always returns a dict keyed by requested block id.
2. Empty string is returned if no snippet source is found.

---

## 6. Graph Quality Metrics (Run-Level)

Location: `incremental_explorer._compute_graph_quality_metrics(...)`

Metrics:

1. `traceability_rate`
   - fraction of macros with at least one meso path that reaches a micro with direct docs.
2. `duplicate_micro_rate`
   - normalized duplicate rate from micro question text normalization.
3. `macro_with_zero_direct_evidence`
   - count of macro nodes with zero direct evidence links.
4. `avg_docs_per_macro`
   - average direct evidence link count across macro nodes.

Payload:

1. included at `question_graph.quality_metrics` in report export.

---

## 7. Run Comparison API and Logic

Route: `POST /api/rag/exploration_notebooks/compare`  
Location: `routes.py`

Purpose:

1. Compare two saved runs for node shape and thematic/evidence overlap.

Process:

1. Validate both paths are under `NOTEBOOK_SAVE_DIR`.
2. For notebook artifact paths, prefer sibling report file when available.
3. Load both payloads as JSON objects.
4. Extract compare features:
   - total nodes/by-level,
   - macro themes from `question_graph.tree`,
   - macro-scope doc ids from sample/shared/document rows.
5. Theme overlap:
   - exact normalized text match first,
   - fuzzy fallback with `SequenceMatcher` threshold `0.84`.
6. Doc overlap:
   - compute shared doc count and Jaccard-style shared coverage ratio.

Return contract:

1. `run_a` summary
2. `run_b` summary
3. `comparison` block:
   - `node_delta`
   - `shared_macro_theme_count`
   - `shared_macro_themes`
   - `shared_macro_doc_count`
   - `shared_doc_coverage_ratio`

---

## 8. Benchmark Rubric Algorithm (Quality-First)

Location: `scripts/benchmark_adaptive_prompt_variants.py`

Base text-quality proxies:

1. `quality_analytic_ratio`
2. `quality_factoid_count`
3. `quality_vague_count`

Graph-quality proxies:

1. `quality_traceability_rate`
2. `quality_duplicate_micro_rate`
3. `quality_macro_zero_direct_evidence`
4. `quality_avg_docs_per_macro`

Composite rubric score:

1. `quality_rubric_score` in `[0,100]`
2. weighted blend favoring analytic + traceable + low-duplicate behavior
3. penalties for factoid/vague leakage and unsupported macro growth

The benchmark output now includes:

1. per-run rubric score,
2. rubric breakdown fields,
3. expanded table columns for fast A/B/C scanning.

---

## 9. UI Behavior (Current)

Question Graph Explorer:

1. three columns (`Macro`, `Meso`, `Micro`) with breadcrumb state,
2. per-node evidence counters (direct/scope/shared),
3. associated-doc panel for selected micro,
4. expandable doc rows with block snippets.

Run Comparison:

1. users select exactly two saved runs,
2. compare panel shows node deltas, shared themes, shared doc coverage,
3. macro theme lists shown side-by-side.

---

## 10. Current Known Limits

1. Early runs with very small document counts may have low traceability.
2. Fuzzy theme overlap is lexical; semantic clustering for compare is not yet implemented.
3. Macro direct evidence counts can remain low when evidence lives mostly at micro level.
4. Long full-corpus runs should be monitored via async status endpoints/logs.

---

## 11. Suggested Next Algorithmic Extensions

1. Add semantic theme matching (embedding-based) in compare API.
2. Add confidence tiers on inferred hierarchy links for UI filtering.
3. Add explicit per-node provenance path serialization (macro->meso->micro->block IDs).
4. Add evaluator fixtures from historian gold/weak question sets directly into benchmark scoring.

