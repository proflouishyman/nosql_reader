# Tier 0 Corpus Exploration - Integration Guide

## Overview
Tier 0 is the systematic corpus exploration layer that runs *before* question answering. It discovers patterns, contradictions, and research questions by reading the corpus in structured batches and accumulating knowledge in a persistent notebook.

## Current Architecture (Already Wired)
**Core files:**
- `app/historian_agent/corpus_explorer.py` (orchestrator)
- `app/historian_agent/stratification.py` (sampling + document objects)
- `app/historian_agent/semantic_chunker.py` (Notebook‑style blocks)
- `app/historian_agent/research_notebook.py` (persistent notebook)
- `app/historian_agent/tier0_models.py` (DocumentObject / DocumentBlock)
- `app/historian_agent/tier0_utils.py` (JSON parsing, logging, saving)

**Question quality pipeline:**
- `app/historian_agent/question_models.py`
- `app/historian_agent/question_typology.py`
- `app/historian_agent/question_validator.py`
- `app/historian_agent/question_pipeline.py`
- `app/historian_agent/question_answerability.py`

**Question synthesis:**
- `app/historian_agent/question_synthesis.py`

## Endpoints (Wired in `app/routes.py`)
- `POST /api/rag/explore_corpus`
- `GET /api/rag/exploration_report`

Example:
```bash
curl -X POST http://localhost:5006/api/rag/explore_corpus \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "balanced",
    "total_budget": 2000,
    "year_range": [1920, 1930]
  }'
```

Full corpus (streaming, no sampling):
```bash
curl -X POST http://localhost:5006/api/rag/explore_corpus \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "full",
    "total_budget": 100000
  }'
```

## How the Tier 0 Loop Works
1. **Stratify** the corpus by time, genre, people, and collections
2. **Build document objects** with semantic blocks (closed‑world evidence)
3. **LLM batch analysis** extracts entities/patterns/contradictions/questions
4. **Notebook accumulates** findings across batches
5. **Question pipeline** generates typed questions and validates them
6. **Answerability precheck** filters questions with too little evidence
7. **Question synthesis** (LLM‑derived buckets) builds a hierarchy, contradiction questions, temporal questions, and flags gaps

## Configuration (All in `.env`)
Tier 0 is configured through `APP_CONFIG.tier0` which reads from `.env`. See `docs/tier3/QUICKSTART.md` for the full list of variables.

## Output Shape
The endpoint returns:
- `corpus_map` (stats + archive notes)
- `questions` (typed + validated)
- `question_synthesis` (hierarchy + gaps)
- `notebook_synthesis` (theme macros + evidence briefs)
- `patterns`, `entities`, `contradictions`, `group_indicators`
- `notebook_path` (if auto‑save enabled)

## Notes
- Closed‑world enforcement is on by default (`TIER0_STRICT_CLOSED_WORLD=1`).
- Batch statistics are computed in code, not by the LLM.
- Question generation uses typology + adversarial validation + answerability precheck.
- Contradictions include a lightweight type classification (name variant, ID conflict, date conflict, etc.).
- Group indicators are extracted only when explicitly stated; the system prefers false negatives over false positives.
- Cronon‑style synthesis adds purpose statements and why‑then/why‑there framing.
- Notebook synthesis adds theme macro paragraphs and evidence briefs for the editor.
- Repair loops are skipped on very small batches (`TIER0_REPAIR_MIN_DOCS`).
- Question generation and validation include JSON‑repair passes (format‑only, no new facts).
- Pattern merging uses a conservative similarity threshold (`TIER0_PATTERN_MERGE_THRESHOLD`) to accumulate evidence without over‑merging.
- Full corpus mode streams documents without `$sample` and can be forced by `TIER0_FULL_CORPUS=1`.
