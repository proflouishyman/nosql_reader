# Question Quality Update

## What Changed
Tier 0 now uses **historical question typology** plus **adversarial validation** to improve quality and answerability. The pipeline runs automatically inside `corpus_explorer.py`.

## New Modules
- `app/historian_agent/question_models.py`
- `app/historian_agent/question_typology.py`
- `app/historian_agent/question_validator.py`
- `app/historian_agent/question_pipeline.py`

## Pipeline (High Level)
1. Generate candidates by type (typology)
2. Validate each question (0–100 score)
3. Answerability precheck (light retrieval)
4. Refine borderline questions
5. Evidence/time-scope filters
6. Deduplicate, enforce diversity
5. Return top N

## Question Types
All 6 historiographic types are generated:
1. Causal
2. Comparative
3. Change‑over‑time
4. Distributional
5. Institutional
6. Scope conditions

## Validation Criteria
Each question is scored across:
- Answerability
- Historical significance
- Specificity
- Evidence‑based grounding

## Configuration
All thresholds and counts are configurable in `.env`:
```
TIER0_QUESTION_PER_TYPE=4
TIER0_QUESTION_MIN_EVIDENCE_DOCS=5
TIER0_QUESTION_MIN_SCORE=60
TIER0_QUESTION_MIN_SCORE_REFINE=50
TIER0_QUESTION_MAX_REFINEMENTS=2
TIER0_QUESTION_TARGET_COUNT=12
TIER0_QUESTION_MIN_COUNT=8
TIER0_QUESTION_ENFORCE_TYPE_DIVERSITY=1
TIER0_QUESTION_MIN_TYPES=3
TIER0_ANSWERABILITY_MIN_DOCS=5
TIER0_ANSWERABILITY_MAX_DOCS=200
TIER0_ANSWERABILITY_TOP_K=50
```

## Example Output (Shape)
```json
{
  "question": "Why did injury rates spike during 1923-1925 labor disputes?",
  "type": "causal",
  "validation": {"score": 85, "status": "good"},
  "answerability_precheck": {"doc_count": 22, "status": "ok"},
  "evidence_count": 47
}
```
