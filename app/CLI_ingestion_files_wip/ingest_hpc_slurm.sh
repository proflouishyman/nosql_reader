#!/bin/bash
#SBATCH --job-name=image_ingest
#SBATCH --output=logs/ingest_%A_%a.out
#SBATCH --error=logs/ingest_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-9  # Adjust based on number of subdirectories

# ============================================================
# SLURM Batch Script for Image Ingestion on HPC
# ============================================================
#
# This script processes images in parallel across multiple compute nodes
# using SLURM job arrays. Each array task processes one subdirectory.
#
# Setup:
#   1. Create logs directory: mkdir -p logs
#   2. Adjust --array range based on number of subdirectories
#   3. Set BASE_DIR to your image root directory
#   4. Configure OLLAMA_URL or OpenAI settings
#
# Submit:
#   sbatch ingest_hpc_slurm.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/ingest_*.out
#
# ============================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# ============================================================
# CONFIGURATION
# ============================================================

# Base directory containing subdirectories of images
BASE_DIR="/path/to/image/collections"

# Provider: "ollama" or "openai"
PROVIDER="ollama"

# Ollama configuration (if using ollama)
OLLAMA_URL="http://gpu-node-01:11434"
OLLAMA_MODEL="llama3.2-vision:latest"

# OpenAI configuration (if using openai)
# OPENAI_API_KEY="sk-..."  # Or use --api-key-file
# OPENAI_MODEL="gpt-4o-mini"

# Processing options
REPROCESS=""  # Set to "--reprocess" to reprocess existing files

# Python environment
CONDA_ENV="nosql_reader"  # Or path to virtualenv

# ============================================================
# SETUP
# ============================================================

echo "============================================================"
echo "SLURM Job Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "============================================================"

# Activate Python environment
if [ -n "${CONDA_ENV:-}" ]; then
    echo "Activating conda environment: $CONDA_ENV"
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate "$CONDA_ENV"
else
    echo "No conda environment specified"
fi

# Set working directory
cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"

# ============================================================
# GET SUBDIRECTORY FOR THIS TASK
# ============================================================

# List all subdirectories
mapfile -t SUBDIRS < <(find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

# Get subdirectory for this array task
if [ ${SLURM_ARRAY_TASK_ID} -ge ${#SUBDIRS[@]} ]; then
    echo "⚠️  Task ID $SLURM_ARRAY_TASK_ID exceeds number of subdirectories (${#SUBDIRS[@]})"
    exit 0
fi

SUBDIR="${SUBDIRS[$SLURM_ARRAY_TASK_ID]}"

echo ""
echo "Processing subdirectory: $SUBDIR"
echo ""

# ============================================================
# RUN INGESTION
# ============================================================

# Build command
CMD="python3 $SCRIPT_DIR/ingest_cli.py \"$SUBDIR\""

if [ "$PROVIDER" = "ollama" ]; then
    CMD="$CMD --provider ollama"
    CMD="$CMD --ollama-url \"$OLLAMA_URL\""
    CMD="$CMD --model \"$OLLAMA_MODEL\""
elif [ "$PROVIDER" = "openai" ]; then
    CMD="$CMD --provider openai"
    CMD="$CMD --model \"$OPENAI_MODEL\""
    if [ -n "${OPENAI_API_KEY:-}" ]; then
        CMD="$CMD --api-key \"$OPENAI_API_KEY\""
    fi
fi

if [ -n "$REPROCESS" ]; then
    CMD="$CMD $REPROCESS"
fi

# Add summary file
SUMMARY_FILE="logs/summary_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
CMD="$CMD --summary-file \"$SUMMARY_FILE\""

echo "Command: $CMD"
echo ""

# Execute
eval $CMD

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Task completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "============================================================"

exit $EXIT_CODE
