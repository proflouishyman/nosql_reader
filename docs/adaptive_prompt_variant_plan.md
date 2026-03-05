# Adaptive Prompt Variant Plan

## Purpose

This document defines prompt variants for adaptive corpus exploration so we can
run repeatable A/B/C tests and select the variant that best supports historian
style question aggregation.

## Scope

Prompt variants are run-scoped and currently apply to:

1. Batch attentive extraction (entities, patterns, contradictions, questions)
2. Why/how enforcement for question framing
3. Promotion prompts (micro -> meso, meso -> macro)
4. Change/continuity prompt
5. Seed extraction prompt

## Variant Keys

- `v1`: Baseline behavior (conservative, closest to prior prompt style)
- `v2`: Mechanism/comparison emphasis with stronger axis-aware question framing
- `v3`: Question-thread emphasis (micro to macro ladder) with stricter anti-trivia filter

## Runtime Controls

- Config default: `TIER0_ADAPTIVE_PROMPT_VARIANT=v1`
- Per-run override in API payload: `prompt_variant` (`v1|v2|v3`)

The override is stored in `exploration_metadata.prompt_variant` for auditability.

## Benchmark Method

Use the benchmark script:

```bash
python scripts/benchmark_adaptive_prompt_variants.py \
  --base-url http://localhost:5001 \
  --documents 100 \
  --variants v1,v2,v3 \
  --sort-order archival
```

Output JSON is saved under:

- `logs/prompt_variant_benchmarks/`

## Selection Criteria

Choose the prompt variant that maximizes:

1. Emergent node formation (`graph_emergent_nodes`)
2. Meso/macro growth (`graph_meso_nodes + graph_macro_nodes`, excluding seed-only wins)
3. Graph structure (`graph_edges`, `decision_log_count`)
4. Promotion activity (`defrag_promotions_total`)
5. Seed grounding (`seed_questions_confirmed`)

Reject variants that:

1. Collapse to seed-only graphs
2. Inflate micro-only nodes with low promotions
3. Increase runtime significantly with no structural graph gain
