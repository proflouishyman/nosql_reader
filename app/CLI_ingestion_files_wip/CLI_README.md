# Image Ingestion CLI Tools

Command-line tools for batch processing document images with OCR and structured data extraction.

## Overview

This suite provides three scripts for different use cases:

1. **`ingest_cli.py`** - Single directory processing
2. **`ingest_parallel.py`** - Multiple directories in parallel (workstation)
3. **`ingest_hpc_slurm.sh`** - HPC batch processing with SLURM job arrays

## Architecture

### Two-Stage Pipeline (Ollama)
```
image.jpg → Vision Model (OCR) → image.jpg.ocr.txt
          → Text Model (JSON) → image.jpg.json
          → MongoDB
```

### Single-Stage Pipeline (OpenAI)
```
image.jpg → GPT-4 Vision → image.jpg.json → MongoDB
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Make scripts executable
chmod +x ingest_cli.py ingest_parallel.py ingest_hpc_slurm.sh
```

## Quick Start

### 1. Process a Single Directory

```bash
# Basic usage (Ollama default)
python ingest_cli.py /path/to/images

# With custom Ollama server
python ingest_cli.py /path/to/images \
    --ollama-url http://server:11434 \
    --model llama3.2-vision:90b

# With OpenAI
python ingest_cli.py /path/to/images \
    --provider openai \
    --api-key sk-...

# Reprocess everything
python ingest_cli.py /path/to/images --reprocess
```

### 2. Parallel Processing (Workstation)

```bash
# Process all subdirectories with 4 workers
python ingest_parallel.py /path/to/collections \
    --subdirs \
    --workers 4

# Process specific directories
python ingest_parallel.py \
    /data/archive1 \
    /data/archive2 \
    /data/archive3 \
    --workers 3
```

### 3. HPC Batch Processing

```bash
# Edit configuration in script
nano ingest_hpc_slurm.sh

# Submit job array
sbatch ingest_hpc_slurm.sh

# Monitor jobs
squeue -u $USER
watch -n 5 squeue -u $USER

# Check logs
tail -f logs/ingest_*.out
```

## Command Reference

### ingest_cli.py

Process a single directory with detailed control.

```bash
python ingest_cli.py <directory> [options]
```

**Required:**
- `directory` - Path to images

**Provider Options:**
- `--provider {ollama,openai}` - AI provider (default: from env or ollama)
- `--model MODEL` - Model name
- `--ollama-url URL` - Ollama server URL
- `--api-key KEY` - OpenAI API key
- `--api-key-file PATH` - File containing API key

**Processing Options:**
- `--reprocess` - Regenerate all files (ignore existing .ocr.txt and .json)
- `--dry-run` - Scan and report, don't process
- `--prompt TEXT` - Custom structuring prompt
- `--prompt-file PATH` - File containing prompt

**Output:**
- `--summary-file PATH` - Write summary JSON
- `--verbose` - Enable verbose logging

### ingest_parallel.py

Process multiple directories in parallel.

```bash
python ingest_parallel.py <dir1> [dir2 ...] [options]
```

**Arguments:**
- `directories` - One or more directories to process

**Parallel Options:**
- `--workers N` - Number of parallel workers (default: 4)
- `--subdirs` - If single dir given, process all subdirectories
- `--max-depth N` - Subdirectory depth (default: 1)

**Other options same as `ingest_cli.py`**

### ingest_hpc_slurm.sh

SLURM batch script for HPC clusters.

**Configuration (edit script):**
- `BASE_DIR` - Root directory with subdirectories
- `PROVIDER` - ollama or openai
- `OLLAMA_URL` - Ollama server URL
- `OLLAMA_MODEL` - Vision model name
- `REPROCESS` - Set to "--reprocess" if needed
- `CONDA_ENV` - Python environment name

**SLURM Options:**
- `--array=0-N` - Process N+1 subdirectories
- `--cpus-per-task=N` - CPUs per job
- `--mem=XG` - Memory per job
- `--time=HH:MM:SS` - Max runtime

## Examples

### Example 1: Local Testing with Dry Run

```bash
# Check what would be processed
python ingest_cli.py /data/test_batch --dry-run

# Output:
# ============================================================
# SCAN RESULTS: /data/test_batch
# ============================================================
#   Total images found: 1,234
#
#   Breakdown by extension:
#     .jpg: 800 files
#     .png: 400 files
#     .tiff: 34 files
#
#   Existing intermediate files:
#     .ocr.txt files: 450
#     .json files: 450
```

### Example 2: Process with Custom Prompt

```bash
# Create custom prompt
cat > my_prompt.txt << 'EOF'
Extract text and structure as JSON with these fields:
{
  "document_type": "string",
  "date": "string", 
  "names": ["string"],
  "amounts": [{"value": "string", "currency": "string"}]
}
EOF

# Use custom prompt
python ingest_cli.py /data/invoices \
    --prompt-file my_prompt.txt
```

### Example 3: Parallel Processing with Summary

```bash
# Process 5 directories in parallel, save summary
python ingest_parallel.py \
    /data/archive_2020 \
    /data/archive_2021 \
    /data/archive_2022 \
    /data/archive_2023 \
    /data/archive_2024 \
    --workers 5 \
    --summary-file results/batch_summary.json

# Check summary
cat results/batch_summary.json | jq '.total_ingested'
```

### Example 4: HPC Job Array Setup

```bash
# Count subdirectories
NUM_DIRS=$(find /archive/collections -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "Found $NUM_DIRS subdirectories"

# Edit SLURM script
nano ingest_hpc_slurm.sh
# Set: #SBATCH --array=0-$((NUM_DIRS-1))
# Set: BASE_DIR="/archive/collections"

# Create logs directory
mkdir -p logs

# Submit
sbatch ingest_hpc_slurm.sh
# Output: Submitted batch job 123456

# Monitor progress
squeue -u $USER -o "%.10i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# Check individual logs
tail -f logs/ingest_123456_0.out
```

### Example 5: Reprocessing Failed Jobs

```bash
# Find directories that failed
grep -l "exit code: 1" logs/*.out | \
    sed 's/.*ingest_.*_\([0-9]*\)\.out/\1/' > failed_tasks.txt

# Get failed directory paths
BASE=/archive/collections
while read task_id; do
    find "$BASE" -mindepth 1 -maxdepth 1 -type d | sort | sed -n "$((task_id+1))p"
done < failed_tasks.txt > failed_dirs.txt

# Reprocess failed directories
python ingest_parallel.py $(cat failed_dirs.txt) \
    --workers 4 \
    --reprocess
```

## Performance Tips

### Ollama Optimization

1. **Use appropriate models:**
   - Small batches: `llama3.2-vision:11b`
   - Large batches: `llama3.2-vision:90b` (if GPU has VRAM)

2. **Tune context size:**
   - Edit `OLLAMA_STAGE2_CONTEXT_SIZE` in `image_ingestion.py`
   - Default: 8192 tokens
   - Increase for complex documents: 16384 or 32768

3. **Image preprocessing:**
   - Automatically resizes to 1600px max dimension
   - Adjust `MAX_IMAGE_SIDE` in `image_ingestion.py` if needed

### HPC Optimization

1. **Resource allocation:**
   ```bash
   #SBATCH --cpus-per-task=4    # Ollama can use multiple cores
   #SBATCH --mem=16G             # 16GB typical for vision models
   #SBATCH --gres=gpu:1          # Use GPU if available
   ```

2. **Array size:**
   - Match number of subdirectories
   - Each task processes one subdirectory completely

3. **Checkpointing:**
   - The pipeline automatically saves `.ocr.txt` files
   - Failed jobs can resume without redoing OCR
   - Use `--reprocess` to force regeneration

### Parallel Processing

1. **Worker count:**
   - Local: `--workers` = CPU cores - 1
   - Shared server: Limit to fair share
   - I/O bound: Can exceed core count

2. **Memory:**
   - Ollama: ~4GB per worker
   - OpenAI: Minimal (just Python process)

## File Structure

```
your_images/
├── image1.jpg
├── image1.jpg.ocr.txt      # Intermediate OCR (Ollama only)
├── image1.jpg.json         # Final structured data
├── image2.png
├── image2.png.ocr.txt
└── image2.png.json
```

## Environment Variables

The scripts respect these environment variables:

- `HISTORIAN_AGENT_MODEL_PROVIDER` - Default provider (ollama/openai)
- `HISTORIAN_AGENT_MODEL` - Default model name
- `HISTORIAN_AGENT_PROMPT` - Default structuring prompt
- `HISTORIAN_AGENT_OLLAMA_BASE_URL` - Ollama server URL
- `OLLAMA_BASE_URL` - Fallback Ollama URL
- `OLLAMA_DEFAULT_MODEL` - Fallback Ollama model
- `OPENAI_DEFAULT_MODEL` - Default OpenAI model
- `OPENAI_API_KEY` - OpenAI API key
- `OPENAI_API_KEY_FILE` - Path to API key file

Example variables are documented in:
`/Users/louishyman/coding/nosql/nosql_reader_cleanup/docs/ENV_EXAMPLES.md`

Example values:
```bash
export HISTORIAN_AGENT_MODEL_PROVIDER=ollama
export HISTORIAN_AGENT_MODEL=llama3.2-vision:latest
export OLLAMA_BASE_URL=http://gpu-server:11434
```

## Troubleshooting

### "Failed to connect to Ollama"
- Check Ollama is running: `curl http://localhost:11434/api/tags`
- Verify URL: `--ollama-url http://correct-host:11434`
- Check firewall rules on HPC

### "OpenAI API key required"
- Set via `--api-key` or `--api-key-file`
- Or set `OPENAI_API_KEY` environment variable
- Or write to `~/.config/nosql_reader/openai_api_key.txt`

### "OCR text too short"
- Image may be unreadable (blank, corrupted, wrong format)
- Check original image quality
- Adjust `MIN_OCR_TEXT_LENGTH` (default: 50 chars)

### SLURM job fails immediately
- Check logs: `cat logs/ingest_*.err`
- Verify Python environment activation
- Test command manually: `bash ingest_hpc_slurm.sh`

### Out of memory errors
- Reduce worker count: `--workers 2`
- Increase SLURM memory: `#SBATCH --mem=32G`
- Use smaller vision model
- Process fewer images per batch

## Performance Benchmarks

Typical processing times (per image):

| Provider | Model | OCR | Structure | Total |
|----------|-------|-----|-----------|-------|
| Ollama | llama3.2-vision:11b | 16s | 60s | ~76s |
| Ollama | llama3.2-vision:90b | 30s | 90s | ~120s |
| OpenAI | gpt-4o | - | 5s | ~5s |
| OpenAI | gpt-4o-mini | - | 3s | ~3s |

*Note: OpenAI is single-stage, so no separate OCR time*

**Reprocessing with cached OCR (Ollama):**
- Structure only: ~60s (saves 16-30s per image)

## License

MIT License - See LICENSE file for details.
