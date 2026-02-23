# Image Ingestion CLI - Quick Reference

## 🚀 Common Commands

### Single Directory Processing

```bash
# Basic (Ollama default)
python ingest_cli.py /data/images

# Dry run (see what would happen)
python ingest_cli.py /data/images --dry-run

# Reprocess everything
python ingest_cli.py /data/images --reprocess

# OpenAI instead
python ingest_cli.py /data/images --provider openai --api-key sk-...

# Custom Ollama server
python ingest_cli.py /data/images --ollama-url http://gpu-server:11434
```

### Parallel Processing

```bash
# Process all subdirectories with 4 workers
python ingest_parallel.py /data/collections --subdirs --workers 4

# Process specific directories
python ingest_parallel.py /data/batch1 /data/batch2 --workers 2

# With summary file
python ingest_parallel.py /data/images --subdirs --summary-file results.json
```

### Check Status

```bash
# Check progress
python check_status.py /data/images

# Check multiple directories
python check_status.py /data/batch1 /data/batch2

# Show incomplete files
python check_status.py /data/images --show-incomplete

# Export detailed report
python check_status.py /data/images --report status.json
```

## 🖥️ HPC / SLURM

```bash
# Setup
mkdir -p logs
nano ingest_hpc_slurm.sh  # Edit configuration

# Submit
sbatch ingest_hpc_slurm.sh

# Monitor
squeue -u $USER
watch -n 5 squeue -u $USER

# Check logs
tail -f logs/ingest_*.out
ls -lh logs/ingest_*.err

# Cancel
scancel JOBID
scancel -u $USER  # Cancel all your jobs
```

## 📁 File Structure

```
images/
├── doc1.jpg              # Original image
├── doc1.jpg.ocr.txt      # OCR text (Ollama only, cached)
└── doc1.jpg.json         # Final structured data
```

## ⚙️ Environment Setup

```bash
# Edit master config in project root
nano ../../.env

# Or use in Python
from dotenv import load_dotenv
load_dotenv()
```

Reference example variables live in:
`/Users/louishyman/coding/nosql/nosql_reader/docs/ENV_EXAMPLES.md` <!-- Updated docs path to the main repository. -->

## 🔍 Debugging

```bash
# Verbose logging
python ingest_cli.py /data/images --verbose

# Test single image manually
python -c "
import image_ingestion as ing
from pathlib import Path

config = ing.ModelConfig(
    provider='ollama',
    model='llama3.2-vision:latest',
    prompt=ing.DEFAULT_PROMPT
)

# Test OCR
ocr = ing._call_ollama_stage1_ocr(
    Path('test.jpg'), 
    config
)
print(ocr)
"

# Check Ollama connection
curl http://localhost:11434/api/tags

# List available models
python -c "import image_ingestion as ing; print(ing.ollama_models())"
```

## 📊 Performance

**Typical timing per image:**
- Ollama OCR: 15-30s
- Ollama Structure: 60-90s
- OpenAI: 3-5s (single stage)

**Optimization:**
- Use `--reprocess` cautiously (expensive)
- `.ocr.txt` files enable cheap re-structuring
- Parallel workers = CPU cores - 1

## 🔑 API Keys

```bash
# OpenAI via environment
export OPENAI_API_KEY=sk-...

# Or via file (more secure)
mkdir -p ~/.config/nosql_reader
echo "sk-..." > ~/.config/nosql_reader/openai_api_key.txt
chmod 600 ~/.config/nosql_reader/openai_api_key.txt

# Or via command line
python ingest_cli.py /data/images --provider openai --api-key sk-...
```

## 🐛 Common Issues

**"Failed to connect to Ollama"**
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags
# If remote: --ollama-url http://server:11434
```

**"OpenAI API key required"**
```bash
# Set key
export OPENAI_API_KEY=sk-...
# Or use --api-key flag
```

**"OCR text too short"**
```bash
# Image unreadable - check original quality
# Skip with: edit MIN_OCR_TEXT_LENGTH in image_ingestion.py
```

**Out of memory**
```bash
# Reduce workers
python ingest_parallel.py /data/images --workers 2
# Or increase SLURM memory
#SBATCH --mem=32G
```

## 📈 Workflow Examples

### New Batch
```bash
# 1. Dry run to check
python ingest_cli.py /data/new_batch --dry-run

# 2. Process single directory (test)
python ingest_cli.py /data/new_batch/test_subset

# 3. If looks good, process all in parallel
python ingest_parallel.py /data/new_batch --subdirs --workers 4

# 4. Check status
python check_status.py /data/new_batch --subdirs
```

### HPC Large Scale
```bash
# 1. Prep
mkdir -p logs
cp ingest_hpc_slurm.sh my_batch.sh
nano my_batch.sh  # Edit BASE_DIR

# 2. Count directories
NUM=$(find /data/archive -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "Processing $NUM directories"

# 3. Submit
sbatch --array=0-$((NUM-1)) my_batch.sh

# 4. Monitor
watch -n 30 'squeue -u $USER | tail -20'

# 5. Check status
python check_status.py /data/archive --subdirs --summary-only
```

### Re-run Failures
```bash
# 1. Find incomplete
python check_status.py /data/batch --show-incomplete --report status.json

# 2. Extract failed directories
cat status.json | jq -r '.directories[] | select(.completion_rate < 100) | .directory' > failed.txt

# 3. Reprocess
cat failed.txt | xargs python ingest_parallel.py --workers 4 --reprocess
```

## 📋 Exit Codes

- `0` - Success, all processed
- `1` - Some failures occurred
- `130` - Interrupted (Ctrl+C)

## 💡 Tips

1. **Always dry-run first** on new datasets
2. **Save summaries** for audit trails: `--summary-file`
3. **Cache OCR** - don't use `--reprocess` unless needed
4. **Monitor HPC** - check logs regularly
5. **Spot-check results** - verify JSON quality periodically
