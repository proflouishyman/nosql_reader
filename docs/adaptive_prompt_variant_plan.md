# Adaptive Prompt Variant Plan v2 (Historian-Inductive Synthesis)

## Why the Plan Changed

The original plan focused on graph growth metrics. The revised plan incorporates
historian practice directly:

1. Questions emerge from documents (not pre-specified hypotheses alone).
2. Reading is inductive and iterative (documents -> patterns -> puzzles -> questions).
3. Archives are analyzed as institutional artifacts (including absences/silences).
4. Strong questions are explanatory, bounded, and debatable.

This shifts us from "generate more nodes" to "generate better historical questions."

## Revised Core Model

### Inductive pipeline (required behavior)

1. Extract document-level observations.
2. Detect four archival signals across documents:
   - repetition
   - anomaly
   - change over time
   - absence
3. Convert signals into puzzles.
4. Convert puzzles into historian-grade questions.
5. Accumulate evidence and promote only when evidence supports abstraction.

### Question quality target

Questions should primarily fall into:

1. Causal
2. Institutional
3. Social Structure
4. Change Over Time
5. Experience/Meaning

## Prompt Variant Strategy

Variants now represent intentional reasoning styles:

- `v1`: baseline legacy-compatible
- `v2`: mechanism/comparison emphasis
- `v3`: micro->macro thread emphasis
- `v4`: strict contract + anti-trivia formatting discipline
- `v5`: historian-inductive process with explicit four-signal scan

## Runtime Controls

- `TIER0_ADAPTIVE_PROMPT_VARIANT` controls default.
- API payload `prompt_variant` overrides per run.
- API payload `ledger_model` overrides model per run.
- Both values are recorded in `exploration_metadata`.

## Actionable Implementation Backlog

### A. Prompt/Extraction (implemented + next)

1. Keep `v5` as historian-inductive baseline for current experiments.
2. Add explicit output fields (future):
   - `signal_type`: repetition|anomaly|change|absence
   - `question_family`: causal|institutional|social_structure|change_over_time|experience_meaning
3. Add hard rejection for weak pseudo-analytic templates unless rewritten.

### B. Graph Aggregation (next)

1. Preserve independent threads when low-confidence links only.
2. Increase promotion gating for generic meso/macro rewrites.
3. Add lineage trace export per macro question:
   macro -> meso -> micro -> evidence blocks.

### C. Evaluation Harness (implemented + next)

1. Keep structure metrics:
   - emergent nodes, meso/macro nodes, promotions, seed confirmation.
2. Keep quality proxies (implemented):
   - `quality_analytic_ratio`
   - `quality_factoid_count`
   - `quality_vague_count`
3. Add curated gold/bad benchmark set (next):
   - good historical questions
   - bad factoid/trivia
   - pseudo-analytic weak forms

### D. Model A/B/C (in progress)

1. Run matrix by prompt variant x model family.
2. Track failure/timeout rates separately from quality.
3. Select by quality-first, speed-second policy.

## Benchmark Method (current)

```bash
python scripts/benchmark_adaptive_prompt_variants.py \
  --base-url http://localhost:5001 \
  --documents 25 \
  --variants v3,v4,v5 \
  --models qwen2.5:32b,gemma3:12b,llama3.1:8b \
  --sort-order archival
```

Output:

- `/Users/louishyman/coding/nosql/nosql_reader/logs/prompt_variant_benchmarks/`

## Acceptance Criteria (revised)

### Must pass

1. Non-trivial emergent graph growth (not seed-only collapse).
2. Meso/macro promotions are present and evidence-grounded.
3. Macro questions are traceable to micro evidence.
4. Factoid/vague question counts stay low.
5. Seed questions are explicitly confirmed or unconfirmed.

### Preferred

1. Higher analytic ratio with similar or better seed confirmation.
2. Stable behavior across at least two model families.
3. Runtime remains within practical iteration bounds.

## Immediate Next Steps

1. Benchmark `v5` against `v3` and `v4` at 25 docs and 100 docs.
2. Promote `v5` only if quality proxies improve without seed-collapse regression.
3. Build curated evaluator set from historian-provided examples and wire into benchmark report.
