# Prompting Best-Practices Web Review (Adaptive Explorer)

Date: 2026-03-05  
Scope: Prompt quality improvements for adaptive corpus exploration

## Primary Sources Reviewed

- OpenAI Prompt Engineering Guide: <https://platform.openai.com/docs/guides/prompt-engineering>
- OpenAI Best Practices (API): <https://help.openai.com/en/articles/6654000-best-practices-for-prompting>
- Anthropic Prompt Engineering Overview: <https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview>
- Google Vertex AI Prompt Design Strategies: <https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/prompt-design-strategies>

## Cross-Source Consensus

1. Put task instructions first and separate sections with clear delimiters.
2. Define strict output contracts (schema, required fields, no extra prose).
3. Use specific positive instructions, not only prohibitions.
4. Provide small good/bad examples to sharpen behavior.
5. Keep tasks decomposed and ordered (step-by-step workflow).
6. Prefer deterministic settings for format-critical stages.

## Implemented Changes

### Prompt Variant `v4`

Implemented in `app/historian_agent/adaptive_prompts.py`.

- Added explicit ordered instruction block (`INSTRUCTIONS (follow in order)`).
- Added closed-world constraints in tagged sections.
- Added question quality rubric and good/bad examples.
- Kept strict JSON schema identical to existing contract.
- Added stronger wording for evidence anchoring and cross-document value.

### Why/How Enforcement Prompt (`v4`)

- Added short decision examples to reduce inconsistent classification.
- Reinforced scope-preservation constraints (actors/place/period unchanged).

### Promotion and Change/Continuity Prompts (`v4`)

- Added explicit requirement to preserve evidentiary tension.
- Added anti-generic wording constraints and corpus-answerability checks.

### Seed Extraction Prompt (`v4`)

- Added broad-but-falsifiable rule.
- Added distinct-dimensions preference and anti-fabrication constraints.

## Model/Prompt Experiment Harness Updates

- Added run payload support for `ledger_model` override in adaptive mode.
- Updated benchmark script (`scripts/benchmark_adaptive_prompt_variants.py`) to run:
  - prompt variants across one or more models,
  - optional auto-model selection from local `ollama list`,
  - matrix reporting with requested/effective model recorded.

## Evaluation Metrics to Track

1. Macro/meso growth with stable micro traceability.
2. Seed confirmation rate (confirmed vs unconfirmed seeds).
3. Promotion count per interval (non-zero expected in successful runs).
4. Ratio of high-value cross-document questions vs single-doc factoids.
5. Decision-log density and edge diversity (merge/decompose/generalize balance).
