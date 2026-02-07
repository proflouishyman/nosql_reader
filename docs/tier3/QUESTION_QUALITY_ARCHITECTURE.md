# Question Quality Architecture

## Layers
1. **Models**: `question_models.py`
2. **Typology**: `question_typology.py`
3. **Validation**: `question_validator.py`
4. **Pipeline**: `question_pipeline.py`

## Design Principle
**Lists internally, dicts externally**
- Internal: dataclasses and `List[Question]`
- External: `List[Dict]` via `to_dict()`

## Flow
```
ResearchNotebook
  → TypedQuestionGenerator (by historiographic type)
  → QuestionValidator (adversarial scoring + refinement)
  → Answerability Precheck (light retrieval)
  → Evidence/Time Guard (evidence threshold + time-scope sanitization)
  → QuestionGenerationPipeline (filter + rank)
  → JSON for API
```

## Configuration
Controlled via `APP_CONFIG.tier0` / `.env`:
- Per‑type generation count
- Validation thresholds
- Refinement limits
- Target output count
- Diversity enforcement
