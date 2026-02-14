#!/usr/bin/env bash
set -euo pipefail

NOTEBOOK="${NOTEBOOK_PATH:-/app/logs/corpus_exploration/exploration_20260209_012130/20260209_012130_notebook.json}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTDIR="/app/logs/synthesis_matrix_${RUN_TAG}"

mkdir -p "/Users/louishyman/coding/nosql/nosql_reader/app/logs/synthesis_matrix_${RUN_TAG}"

MODELS=("llama3.1:8b" "llama3.3:latest" "gpt-oss:20b")
TARGETS=(12 24 36)

for model in "${MODELS[@]}"; do
  model_tag="$(echo "$model" | tr ":/" "__")"
  for target in "${TARGETS[@]}"; do
    sample="$target"
    pattern="$target"
    out="${OUTDIR}/synth_${model_tag}_q${target}_s${sample}_p${pattern}.json"
    docker compose -f /Users/louishyman/coding/nosql/nosql_reader/docker-compose.yml exec -T app sh -lc \
      "PYTHONPATH=/app:/app/historian_agent \
      NOTEBOOK_PATH=${NOTEBOOK} \
      OUT_PATH=${out} \
      REGEN_QUESTIONS=1 \
      LLM_MODEL=${model} \
      LLM_FAST_MODEL=${model} \
      VERIFIER_MODEL=${model} \
      TIER0_QUESTION_TARGET_COUNT=${target} \
      TIER0_QUESTION_PER_TYPE=8 \
      TIER0_QUESTION_MIN_SCORE=60 \
      TIER0_SYNTHESIS_MAX_QUESTION_SAMPLE=${sample} \
      TIER0_SYNTHESIS_MAX_PATTERN_SAMPLE=${pattern} \
      python /app/run_synthesis_once.py"
  done
done

echo "matrix done: ${OUTDIR}"
