# Historian Question Generation Framework (Archive-First)

Date: 2026-03-05
Purpose: Encode researcher guidance into adaptive prompt/evaluation design.

## Core Process

1. Observation: identify recurring forms, anomalies, categories, and silences.
2. Puzzle: convert oddities into questions that require explanation.
3. System: move from document-level puzzle to institutional/power explanation.
4. Pattern: aggregate repeated evidence across documents.
5. Absence: treat meaningful archival silence as evidence.

## Five Question Types

- Causal: why did an outcome occur?
- Institutional: how did organizational systems operate?
- Social Structure: how did hierarchy and inequality operate?
- Change Over Time: how/why did patterns shift or persist?
- Experience/Meaning: how did actors interpret their world?

## Quality Checks

A strong question should satisfy all or most:
- could produce competing explanations,
- implies mechanism or causal process,
- needs multiple sources to answer,
- remains bounded in scope.

## Common Failure Modes

- Factoid recall: when/who/what year/how many.
- Vague pseudo-analytic prompts: "what impact", "what role", "what was life like".
- Over-broad framing that forces list-like summaries instead of arguments.

## Implementation Mapping

- Prompt variant `v5` in `app/historian_agent/adaptive_prompts.py` applies this process.
- Benchmark quality proxies in `scripts/benchmark_adaptive_prompt_variants.py` track:
  - analytic ratio,
  - factoid count,
  - vague-question count.
