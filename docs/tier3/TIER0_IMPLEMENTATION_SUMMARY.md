# Tier 0 Corpus Exploration - Implementation Summary

## What Tier 0 Does
Tier 0 performs **systematic corpus exploration** before question‑answering. It reads the corpus in structured batches, accumulates knowledge in a notebook, and generates **typed, validated research questions**.

## Key Features
- **Stratified sampling** (temporal, genre, biographical, collection)
- **Full-corpus streaming** (no sampling, processes every document)
- **Notebook‑style document objects** with semantic blocks
- **Batch coverage guarantee** (no doc skipping when `batch_max_chars` is reached)
- **Strict year extraction** (optional date-label extraction when `year` field missing)
- **Closed‑world batch analysis** (no outside knowledge)
- **Inductive batch questions** (explicitly forbid single‑person factoids)
- **Notebook question filtering** (factoid rejection + dedup before notebook insert)
- **Evidence alignment** (block IDs validated/repairable)
- **Repair guard** (skip repair on very small batches)
- **Typed question generation** (6 historiographic categories)
- **Adversarial validation** (answerability/significance/specificity/evidence)
- **JSON repair** (strict reformatting when models drift from JSON)
- **Answerability precheck** (lightweight retrieval thresholding)
- **Evidence thresholding** (skip low‑evidence patterns/questions)
- **Pattern merging** (high‑similarity patterns merged to accumulate evidence)
- **Time-scope guard** (remove unsupported time windows)
- **Contradiction typing** (name/ID/date vs true conflicts)
- **Group indicators** (race/gender/class/ethnicity/origin/occupation; explicit mentions only)
- **Question synthesis** (LLM‑derived buckets + hierarchy + gap analysis + contradiction/temporal questions)
- **Cronon‑style framing** (purpose statement, why‑then/why‑there, terms/assumptions)
- **Narrative synthesis** (closed‑world, multi‑paragraph historian narrative)
- **Recursive synthesis** (drill-down leaf answers → theme summaries → long-form essay)
- **Notebook synthesis macros** (theme-level macro paragraphs from patterns + contradictions)
- **Evidence briefs** (theme evidence summaries passed to editor)
- **Evidence‑dense paragraphs** (min citations per body paragraph; weak leaves become gaps)
- **Multi‑pass editor** (structure → evidence → style, using configurable editor model)
- **Semantic assignment + caching** (theme/ question embeddings cached)
- **Semantic clustering** (similar questions grouped under each theme)
- **Synthesis checkpoints** (skip recomputation when inputs unchanged)
- **Notebook persistence** (timestamped saves)
- **Doc-level cache** (per-doc findings stored in Mongo with model+prompt signature)
- **Cache refresh mode** (rebuild blocks when doc cache is empty or stale)

## Recursive Essay Method (History Essay Guidelines)
Recursive synthesis now follows the method in `docs/historypdf.pdf`:
- **Selection + interpretation** drive scope (no pretense of total coverage).
- **Sources vs evidence**: sources are raw; evidence is selected, quoted, and argued.
- **Interpretive questions** (why/how) are favored over descriptive ones.
- **Paragraph structure**: topic sentence + supporting evidence + analysis.
- **Historical writing conventions**: past tense, specificity, avoid presentism.
- **Thesis + purpose statement** (Cronon “I am studying… because… in order to…”).
- **Counterargument & limits** (explicitly note alternatives, contradictions, gaps).

## File Map
**Core:**
- `app/historian_agent/corpus_explorer.py`
- `app/historian_agent/stratification.py`
- `app/historian_agent/semantic_chunker.py`
- `app/historian_agent/research_notebook.py`
- `app/historian_agent/tier0_models.py`
- `app/historian_agent/tier0_utils.py`

**Question Quality:**
- `app/historian_agent/question_models.py`
- `app/historian_agent/question_typology.py`
- `app/historian_agent/question_validator.py`
- `app/historian_agent/question_pipeline.py`

## API Endpoints
- `POST /api/rag/explore_corpus`
- `GET /api/rag/exploration_report`

## Configuration (via `.env`)
All Tier 0 settings are under `APP_CONFIG.tier0` (see `docs/tier3/QUICKSTART.md`).
Full corpus mode can be enabled with `TIER0_FULL_CORPUS=1` or by using `strategy: "full"` in the request.

Key new toggles:
- `TIER0_NOTEBOOK_SYNTHESIS=1` (enable theme macros + evidence briefs)
- `TIER0_PATTERN_CITATION_COUNT=3` (doc_ids per pattern for macros)
- `TIER0_PARAGRAPH_MIN_EVIDENCE=2` (min citations per body paragraph)
- `TIER0_ESSAY_REVISE=1` (enable editor passes)
- `TIER0_ESSAY_REVISE_PASSES=3` (structure/evidence/style)
- `TIER0_ESSAY_REVISE_MODEL=llama3.3:latest`
- `TIER0_ESSAY_REVISE_PROFILE=editor`
- `TIER0_DOC_CACHE_MODE=refresh` (force rebuild of cached doc blocks)

## Output Shape (Example)
```json
{
  "corpus_map": {"statistics": {...}, "archive_notes": "..."},
  "questions": [
    {
      "question": "Why did injury rates spike during 1923-1925 labor disputes?",
      "type": "causal",
      "validation": {"score": 85, "status": "good"}
    }
  ],
  "patterns": [...],
  "entities": [...],
  "contradictions": [...],
  "group_indicators": [...],
  "question_synthesis": {
    "grand_narrative": {...},
    "narrative": {...},
    "recursive_synthesis": {...},
    "themes": [...],
    "contradiction_questions": [...],
    "temporal_questions": [...],
    "hierarchy": {...},
    "gaps": [...]
  }
}
```
