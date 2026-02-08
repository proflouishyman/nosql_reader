# Question Synthesis (Tier 0)

## Purpose
After question generation, Tier 0 now synthesizes a **research agenda**:
- Buckets questions into themes
- Builds a hierarchy (grand narrative → theme → sub‑questions)
- Clusters semantically similar questions under each theme
- Adds micro‑level questions from contradictions
- Adds temporal questions from coverage statistics
- Flags missing areas (gaps)
- Promotes Cronon‑style framing (scope, purpose, why‑then/why‑there)
- Surfaces group‑difference questions only when explicit evidence supports them

This turns exploratory questions into an integrative research plan.

## Module
`app/historian_agent/question_synthesis.py`

## Output Shape
```json
{
    "question_synthesis": {
    "grand_narrative": {
      "question": "...",
      "purpose_statement": "I am studying ... because ... in order to ...",
      "scope": {"time": "...", "place": "...", "actors": ["..."]},
      "why_then": "...",
      "why_there": "...",
      "assumptions_to_check": ["..."],
      "terms_to_define": ["..."],
      "themes": ["..."]
    },
    "theme_definitions": [
      {
        "key": "medical_surveillance",
        "name": "Medical Surveillance & Control",
        "description": "...",
        "keywords": ["medical", "exam", "certificate"],
        "scope": {"time": "...", "place": "...", "actors": ["..."]}
      }
    ],
    "themes": [
        {
          "theme": "Medical Certification as Labor Control",
          "overview_question": {...},
        "sub_questions": [...],
        "question_clusters": [
          {
            "representative_question": {...},
            "related_questions": [...]
          }
        ],
        "inductive_logic": {
          "patterns_observed": ["..."],
          "contradictions_observed": 2,
          "inference": "...",
          "historical_argument": "..."
        },
        "evidence_base": {
          "pattern_count": 12,
          "question_count": 5,
          "contradiction_count": 2,
          "confidence_distribution": {"high": 3, "medium": 6, "low": 3}
        }
      }
    ],
    "group_difference_questions": [
      {
        "question": "...",
        "groups": ["..."],
        "time_window": "...",
        "evidence_basis": "..."
      }
    ],
    "contradiction_questions": [
      {
        "question": "...",
        "contradiction_type": "name_variant_or_ocr",
        "claims": ["...", "..."],
        "sources": ["block_id", "block_id"]
      }
    ],
    "temporal_questions": [
      {
        "question": "...",
        "evidence_basis": "density_by_decade"
      }
    ],
    "hierarchy": {
      "level_1_grand": {...},
      "level_2_thematic": [...],
      "level_3_specific": {"theme_key": [{"representative_question": {...}, "related_questions": [...]}]},
      "level_3_group_comparisons": [...],
      "level_4_micro": [...]
    },
    "gaps": [
      {
        "gap": "Demographic coverage",
        "missing_evidence": ["women", "race", "age"],
        "suggested_questions": ["..."]
      }
    ]
  }
}
```

## Theme Buckets (Dynamic)
By default, the synthesizer uses an LLM to **derive theme buckets from the questions + patterns**.
This ensures buckets reflect the actual archive rather than fixed assumptions.

If the LLM fails, it falls back to a stable static set. Themes are merged when they overlap
and expanded when too few are produced.

Assignment is keyword‑based using the generated theme keywords.

## Gap Analysis
The system checks for missing axes (demographic, group differences, economic, labor relations, family impact, comparative context) and proposes gap questions if absent in the notebook.

## Group Differences (High Precision)
Group‑difference questions are only produced when explicit group indicators are found
in the text, with minimum evidence thresholds. This favors **false negatives** over
false positives.

## Configuration
Controlled by:
```
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
TIER0_GROUP_INDICATOR_MIN_DOCS=3
```

LLM response caching (for fast iteration):
```
TIER0_LLM_CACHE_ENABLED=1
TIER0_LLM_CACHE_DIR=/app/logs/llm_cache
```

Synthesis checkpointing (skip recompute when inputs unchanged):
```
TIER0_SYNTHESIS_CHECKPOINT_DIR=/app/logs/synthesis_checkpoints
```
