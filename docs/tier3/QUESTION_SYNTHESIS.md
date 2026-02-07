# Question Synthesis (Tier 0)

## Purpose
After question generation, Tier 0 now synthesizes a **research agenda**:
- Buckets questions into themes
- Builds a hierarchy (grand narrative → theme → sub‑questions)
- Flags missing areas (gaps)
- Promotes Cronon‑style framing (scope, purpose, why‑then/why‑there)
- Surfaces group‑difference questions when evidence supports them

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
      "themes": [
        {
          "theme": "Medical Certification as Labor Control",
          "overview_question": {...},
        "sub_questions": [...],
        "inductive_logic": {
          "patterns_observed": ["..."],
          "inference": "..."
        },
        "evidence_base": {
          "pattern_count": 12,
          "question_count": 5,
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

If the LLM fails, it falls back to a stable static set.

Assignment is keyword‑based using the generated theme keywords.

## Gap Analysis
The system checks for missing axes (demographic, group differences, economic, labor relations, family impact, comparative context) and proposes gap questions if absent in the notebook.

## Configuration
Controlled by:
```
TIER0_SYNTHESIS_ENABLED=1
TIER0_SYNTHESIS_DYNAMIC=1
TIER0_SYNTHESIS_THEME_COUNT=5
TIER0_SYNTHESIS_MAX_QUESTION_SAMPLE=24
TIER0_SYNTHESIS_MAX_PATTERN_SAMPLE=12
```
