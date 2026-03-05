# Adaptive Explorer Design v3 (Inductive Historian Workflow)

Date: 2026-03-05
Status: Active implementation/testing
Supersedes: `/Users/louishyman/coding/nosql/nosql_reader/docs/adaptive_prompt_variant_plan.md`

## 1. Synthesis of New Perspectives

This revision integrates three core historian insights:

1. **Scale jump is the key skill**
   Local observations must scale to system-level questions.
2. **Question generation is inductive and iterative**
   `documents -> observations -> patterns -> puzzles -> questions -> interpretation`.
3. **Archival questions are not yet historical questions**
   Strong research requires transforming record-level puzzles into explanations
   about institutions, power, and historical change.

## 2. Design Rewrite (What Changes)

### A. Two-level question model (new requirement)

- **Level A: Archival question generation**
  Questions tied to record structure, category use, anomalies, and omissions.
- **Level B: Historical question synthesis**
  Questions explaining larger systems that produced those archival patterns.

Implementation impact:
- `v6` prompt variant explicitly encodes this two-level process.
- Promotion logic is treated as archival->historical transformation, not just text broadening.

### B. Four-signal scan (hard requirement)

Each batch must explicitly scan for:
- repetition,
- anomaly,
- change over time,
- absence/silence.

Implementation impact:
- Signals are required in prompt instructions before question generation.
- Future extension: emit explicit `signal_type` labels in structured output.

### C. Three-reader internal role split (prompt-level ensemble)

Prompt instructs internal passes as:
1. pattern finder,
2. anomaly detector,
3. question generator.

Implementation impact:
- `v6` prompt includes this role split without changing output schema.
- Future extension: optional explicit intermediate outputs for audit mode.

### D. Evidence-carrying chain (traceability invariant)

Every higher-level question must remain traceable to concrete evidence blocks.

Implementation impact:
- Keep `evidence_blocks` mandatory in question output.
- Reject promotions that lose mechanism/scope or evidence grounding.

## 3. Actionable Implementation Plan

### Phase 1 — Prompt layer (now)

1. Add/maintain `v6` across:
   - batch analysis,
   - why/how enforcement,
   - promotion,
   - change/continuity,
   - seed extraction.
2. Preserve strict JSON contracts and closed-world constraints.

### Phase 2 — Question transformation instrumentation (next)

1. Add optional metadata fields (guarded, backward-compatible):
   - `question_origin_level`: archival|historical,
   - `question_family`: causal|institutional|social_structure|change_over_time|experience_meaning,
   - `signal_type`: repetition|anomaly|change|absence.
2. Persist these fields into decision/audit logs.

### Phase 3 — Quality scoring (in progress)

1. Keep structural graph metrics.
2. Keep quality proxies:
   - analytic ratio,
   - factoid count,
   - vague-question count.
3. Add curated evaluator set from historian-provided examples:
   - strong historical questions,
   - weak factoid questions,
   - pseudo-analytic weak questions.

### Phase 4 — Promotion robustness (next)

1. Strengthen skip-on-thin-evidence behavior.
2. Add penalty for generic macro rewrites.
3. Require at least one explicit mechanism condition in promoted prompts.

### Phase 5 — Overnight iterative testing (active)

1. Compare `v4` vs `v5` vs `v6`.
2. Compare models by quality-first policy.
3. Keep qwen3.5 watcher active and run matrix when model is ready.

## 4. Testing Matrix

## Fast loop (quality smoke)
```bash
python scripts/benchmark_adaptive_prompt_variants.py \
  --documents 10 \
  --variants v4,v5,v6 \
  --models qwen2.5:32b,gemma3:12b,llama3.1:8b \
  --sort-order archival
```

## Medium loop (aggregation behavior)
```bash
python scripts/benchmark_adaptive_prompt_variants.py \
  --documents 25 \
  --variants v4,v5,v6 \
  --models qwen2.5:32b,gemma3:12b \
  --sort-order archival
```

## qwen3.5 readiness run
Triggered by watcher; log file:
- `/Users/louishyman/coding/nosql/nosql_reader/logs/prompt_variant_benchmarks/qwen35_watch.log`

## 5. Acceptance Criteria (v3)

1. Macro questions are evidence-traceable through meso/micro nodes.
2. `v6` improves or matches analytic ratio without increasing factoid/vague counts.
3. Seed confirmation does not regress significantly versus `v4/v5`.
4. Promotion counts remain non-zero with bounded generic-promote failures.
5. At least one model family shows stable historian-grade behavior at 25 docs.

## 6. Explicit Risks

1. Over-scaling risk: turning good archival questions into generic macro prompts.
2. Under-scaling risk: remaining in local/document-level questions.
3. Model variance: some models collapse to seed-only unless prompt is highly constrained.
4. Absence detection fragility: true silence vs ordinary non-mention.

## 7. Immediate Next Tasks

1. Finish `v6` benchmark pass and compare against `v4/v5`.
2. Add evaluator fixtures from historian examples into benchmark script inputs.
3. Tune promotion/seed thresholds if `v6` raises analytic quality but lowers seed confirmation.
