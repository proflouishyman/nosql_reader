# Tier 0 Corpus Exploration - Quick Start

## What’s Included
Tier 0 is already integrated into the codebase with:
- Notebook-style corpus exploration
- Semantic chunking and closed-world batch analysis
- Typed question generation + adversarial validation
- Notebook persistence and logs

## Configure (.env)
All Tier 0 settings are loaded from `.env` via `APP_CONFIG.tier0`.

Minimum settings to review:
```
TIER0_EXPLORATION_BUDGET=2000
TIER0_EXPLORATION_STRATEGY=balanced
TIER0_FULL_CORPUS=0
TIER0_LLM_CACHE_ENABLED=1
TIER0_LLM_CACHE_DIR=/app/logs/llm_cache
TIER0_SYNTHESIS_CHECKPOINT_DIR=/app/logs/synthesis_checkpoints
TIER0_EXTRACT_DATES_STRICT=1
TIER0_GROUP_INDICATOR_MIN_DOCS=3
TIER0_BATCH_MAX_CHARS=60000
TIER0_SEMANTIC_CHUNKING=1
TIER0_BLOCK_MAX_CHARS=2000
TIER0_MAX_BLOCKS_PER_DOC=12
TIER0_MIN_ENTITIES_PER_BATCH=5
TIER0_MIN_PATTERNS_PER_BATCH=2
TIER0_REPAIR_ATTEMPTS=1
TIER0_REPAIR_MIN_DOCS=6
TIER0_STRICT_CLOSED_WORLD=1

NOTEBOOK_SAVE_DIR=/app/logs/corpus_exploration
NOTEBOOK_AUTO_SAVE=1
TIER0_LOG_DIR=/app/logs/tier0
TIER0_DEBUG_MODE=0

TIER0_QUESTION_PER_TYPE=4
TIER0_QUESTION_MIN_EVIDENCE_DOCS=3
TIER0_QUESTION_MIN_SCORE=50
TIER0_QUESTION_MIN_SCORE_REFINE=50
TIER0_QUESTION_MAX_REFINEMENTS=2
TIER0_QUESTION_TARGET_COUNT=12
TIER0_QUESTION_MIN_COUNT=8
TIER0_QUESTION_ENFORCE_TYPE_DIVERSITY=1
TIER0_QUESTION_MIN_TYPES=3

TIER0_ANSWERABILITY_MIN_DOCS=5
TIER0_ANSWERABILITY_MAX_DOCS=200
TIER0_ANSWERABILITY_TOP_K=50
TIER0_SYNTHESIS_ENABLED=1
TIER0_SYNTHESIS_DYNAMIC=1
TIER0_SYNTHESIS_SEMANTIC_ASSIGNMENT=1
TIER0_SYNTHESIS_EMBED_PROVIDER=ollama
TIER0_SYNTHESIS_EMBED_MODEL=qwen3-embedding:0.6b
TIER0_SYNTHESIS_EMBED_CACHE=/app/logs/embedding_cache.pkl
TIER0_SYNTHESIS_EMBED_TIMEOUT=120
TIER0_SYNTHESIS_ASSIGN_MIN_SIM=0.2
TIER0_SYNTHESIS_DEDUPE_THRESHOLD=0.86
TIER0_SYNTHESIS_CLUSTER_THRESHOLD=0.78
TIER0_SYNTHESIS_THEME_MERGE_THRESHOLD=0.84
TIER0_SYNTHESIS_THEME_COUNT=5
TIER0_SYNTHESIS_MIN_THEMES=4
TIER0_SYNTHESIS_MAX_QUESTION_SAMPLE=24
TIER0_SYNTHESIS_MAX_PATTERN_SAMPLE=12
TIER0_SYNTHESIS_NARRATIVE_ENABLED=1
TIER0_SYNTHESIS_NARRATIVE_MAX_THEMES=5
TIER0_RECURSIVE_ENABLED=1
TIER0_RECURSIVE_MIN_DOCS=5
TIER0_RECURSIVE_MAX_DOCS=15
TIER0_RECURSIVE_MAX_DEPTH=3
TIER0_RECURSIVE_SUBQUESTION_COUNT=3
TIER0_LEAF_ANSWERS_COLLECTION=tier0_leaf_answers
TIER0_RUNS_COLLECTION=tier0_runs
TIER0_RUNS_STORE_NOTEBOOK=1
TIER0_DOC_CACHE_ENABLED=1
TIER0_DOC_CACHE_MODE=use   # use | refresh | rebuild | off
TIER0_DOC_CACHE_COLLECTION=tier0_doc_cache
TIER0_DOC_CACHE_PROMPT_VERSION=v1

# Doc cache stores per-document findings (entities, patterns, contradictions, group indicators)
# keyed by model + prompt version + chunking config, so later runs can resynthesize without re-reading text.
TIER0_PATTERN_MERGE_THRESHOLD=0.9

# Ollama performance tuning (optional)
OLLAMA_NUM_CTX=131072
OLLAMA_NUM_GPU=99
OLLAMA_NUM_BATCH=128
```

## Endpoints
Tier 0 is wired in `app/routes.py`:
- `POST /api/rag/explore_corpus`
- `GET /api/rag/exploration_report`

Example:
```bash
curl -X POST http://localhost:5006/api/rag/explore_corpus \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "balanced",
    "total_budget": 200,
    "year_range": [1923, 1925]
  }'
```

Full corpus (streamed, long-running):
```bash
curl -X POST http://localhost:5006/api/rag/explore_corpus \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "full",
    "total_budget": 100000
  }'
```
Notes:
- `strategy: "full"` streams documents without sampling.
- You can also force full-corpus behavior by setting `TIER0_FULL_CORPUS=1` in `.env`.

## Expected Output (Shape)
```json
{
  "corpus_map": {"statistics": {"total_documents_read": 200, ...}, "archive_notes": "..."},
  "questions": [
    {
      "question": "Why did injury rates spike during 1923-1925 labor disputes?",
      "type": "causal",
      "validation": {"score": 85, "status": "good", ...},
      "answerability_precheck": {"doc_count": 22, "status": "ok"}
    }
  ],
  "question_synthesis": {
    "grand_narrative": {"question": "...", "themes": ["..."]},
    "themes": [...],
    "contradiction_questions": [...],
    "temporal_questions": [...],
    "hierarchy": {...},
    "gaps": [...]
  },
  "patterns": [...],
  "entities": [...],
  "contradictions": [...],
  "group_indicators": [...],
  "notebook_path": "/app/logs/corpus_exploration/.../notebook_YYYYMMDD_HHMMSS.json"
}
```

## Focused Exploration
```bash
curl -X POST http://localhost:5006/api/rag/explore_corpus \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "temporal",
    "total_budget": 300,
    "year_range": [1923, 1925]
  }'
```
