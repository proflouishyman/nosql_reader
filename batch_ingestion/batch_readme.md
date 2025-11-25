# Batch Image Processing System

Efficient batch processing of historical documents using OpenAI's gpt-5-mini vision model with automatic retry for failed extractions.

## Overview

This system processes images in three stages:

1. **Initial Processing** (`batch_upload_images.py`) - Process all images with `"detail": "low"` for cost efficiency
2. **Result Validation** (`batch_download_images.py`) - Download results, validate JSON, track failures
3. **Retry Failed** (`batch_retry_failed.py`) - Reprocess failed images with `"detail": "high"` for better accuracy

## Features

- ✅ **Cost-optimized**: Start with low-detail processing (~$0.02 per 1000 images)
- ✅ **Extensive logging**: Every operation logged to timestamped files
- ✅ **Image preprocessing**: Validates size, format, dimensions before processing
- ✅ **JSON validation**: Checks for required fields, empty content, parse errors
- ✅ **Automatic retry**: Failed images tracked for high-detail reprocessing
- ✅ **Robust error handling**: Graceful fallbacks, detailed error messages
- ✅ **Progress tracking**: tqdm progress bars + detailed console output
- ✅ **Batch management**: Automatic batching under 95MB limit

## Setup

### 1. Install Dependencies

```bash
pip install openai pillow tqdm
```

### 2. Configure Paths

Edit the configuration sections in each script:

```python
# Update these paths for your environment
API_KEY_FILE = os.path.expanduser("~/api_key.txt")  # Your OpenAI API key
IMAGE_DIR = "./images"  # Directory containing images
PROMPT_FILE = "./vision_prompt.txt"  # Extraction prompt
BATCH_DIR = "./batch"  # Batch tracking directory
```

### 3. Create Vision Prompt

Create `vision_prompt.txt` with your extraction instructions. Example:

```
Perform OCR on the document to extract all text. Then, analyze the text to provide a summary of the content. Structure the extracted data into a STRICT JSON format. Use double quotes for strings. Return the extracted information as a JSON object with the following structure:
{
    "ocr_text": "string",
    "summary": "string",
    "sections": [...]
}
```

### 4. Add API Key

Create a file with your OpenAI API key:

```bash
echo "sk-your-api-key-here" > ~/api_key.txt
chmod 600 ~/api_key.txt
```

## Usage

### Step 1: Initial Processing (Low Detail)

Process all images without existing JSON files:

```bash
python batch_upload_images.py
```

**What it does:**
- Scans `IMAGE_DIR` recursively for images
- Skips images that already have `.json` files
- Validates image size, format, dimensions
- Creates batches under 95MB
- Uploads batches to OpenAI
- Saves batch tracking files to `./batch/`

**Output:**
- Batch tracking files: `./batch/{batch_id}.txt`
- Logs: `./batch/logs/upload_YYYYMMDD_HHMMSS.log`

**Cost:** ~$0.02 per 1000 images @ 138 tokens each

### Step 2: Download and Validate Results

Poll batch status and download completed results:

```bash
python batch_download_images.py
```

**What it does:**
- Polls OpenAI for batch completion (every 60 seconds)
- Downloads completed batch results
- Extracts individual JSON files
- Validates JSON structure and content
- Tracks failed images to `./batch/failed_images.txt`
- Moves completed batches to `./batch/completed/`

**Output:**
- JSON files: `{image_path}.json` (next to original images)
- Failed images list: `./batch/failed_images.txt`
- Logs: `./batch/logs/download_YYYYMMDD_HHMMSS.log`

**Success rate:** Expect ~90-95% valid JSON with low detail

### Step 3: Retry Failed Images (High Detail)

Reprocess failed images with higher detail:

```bash
python batch_retry_failed.py
```

**What it does:**
- Reads `./batch/failed_images.txt`
- Validates images still exist
- Removes old invalid JSON files
- Creates new batches with `"detail": "high"`
- Uploads retry batches to OpenAI

**Output:**
- New batch tracking files: `./batch/{batch_id}.txt`
- Logs: `./batch/logs/retry_YYYYMMDD_HHMMSS.log`

**Cost:** ~$0.35 per 1000 images @ 2350 tokens each

**Then run download again:**
```bash
python batch_download_images.py
```

## Directory Structure

```
.
├── batch_upload_images.py       # Initial processing script
├── batch_download_images.py     # Download and validation script
├── batch_retry_failed.py        # High-detail retry script
├── vision_prompt.txt            # Extraction prompt
├── images/                      # Your images to process
│   ├── folder1/
│   │   ├── page001.png
│   │   └── page001.png.json    # Generated JSON
│   └── folder2/
│       └── ...
└── batch/                       # Batch tracking
    ├── {batch_id}.txt          # Pending batches
    ├── failed_images.txt       # Failed images list
    ├── completed/              # Completed batch metadata
    │   └── {batch_id}.txt
    ├── batch_return/           # Downloaded JSONL files
    │   └── batch_{batch_id}.jsonl
    ├── batch_json_results/     # Extracted results per batch
    │   └── {batch_id}/
    └── logs/                   # All logs
        ├── upload_*.log
        ├── download_*.log
        └── retry_*.log
```

## Logging

All scripts generate comprehensive logs:

### Console Output
- INFO level: Progress, summaries, important events
- WARNING level: Skipped files, validation failures
- ERROR level: Processing failures, API errors

### Log Files
- DEBUG level: Detailed operation traces
- Timestamped filenames: `upload_20250101_120000.log`
- Located in: `./batch/logs/`

**View logs:**
```bash
# Latest upload log
tail -f ./batch/logs/upload_*.log

# Search for errors
grep ERROR ./batch/logs/*.log

# Failed images
cat ./batch/failed_images.txt
```

## JSON Validation

The download script validates JSON for:

1. **Parse Errors**: Must be valid JSON
2. **Required Fields**: Must contain `ocr_text` and `summary`
3. **Empty Content**: OCR text must not be empty
4. **Minimum Length**: OCR text must be at least 10 characters

**Invalid JSON handling:**
- Saved with `_validation_error` field for inspection
- Image path added to `failed_images.txt`
- Can be reprocessed with high detail

## Cost Estimation

### For 1000 Historical Documents (1800x2400px average):

**Initial Pass (Low Detail):**
- Tokens: 138 per image = 138,000 total
- Cost: $0.02 (input) + $0.06 (output) = **$0.08**

**Retry 10% with High Detail:**
- Tokens: 2,350 per image = 235,000 total
- Cost: $0.035 (input) + $0.11 (output) = **$0.145**

**Total: ~$0.23 for 1000 images**

Compare to high-detail-only approach: **$14.50** (63x more expensive!)

## Troubleshooting

### Issue: "API key file not found"
**Solution:** Create API key file at specified path:
```bash
echo "sk-your-key" > ~/api_key.txt
```

### Issue: "Image too large"
**Solution:** Images over 20MB are skipped. Resize before processing:
```bash
mogrify -resize 4096x4096\> -quality 90 *.jpg
```

### Issue: Batch stuck in "validating" or "in_progress"
**Solution:** Normal for large batches. OpenAI processes within 24 hours. Script will keep polling.

### Issue: High failure rate (>20%)
**Solution:** 
- Check image quality (blurry, low resolution)
- Verify prompt is clear and specific
- Images may need high detail from the start (edit `DETAIL_LEVEL` in upload script)

### Issue: "No images need processing"
**Solution:** Script skips images with existing `.json` files. To reprocess:
```bash
# Remove all JSON files
find ./images -name "*.json" -delete
```

## Advanced Configuration

### Change Batch Size

```python
MAX_BATCH_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB (smaller batches)
```

### Adjust Polling Frequency

```python
RETRY_DELAY = 120  # Check every 2 minutes instead of 1
```

### Change Model

```python
MODEL = "gpt-4o-mini"  # Use older model
DETAIL_LEVEL = "auto"  # Let OpenAI decide
```

### Customize Validation

Edit `validate_json()` in `batch_download_images.py`:

```python
def validate_json(json_str: str) -> tuple[bool, str, dict]:
    # Add your custom validation logic
    ...
```

## Best Practices

1. **Start small**: Test with 10-20 images before processing thousands
2. **Monitor logs**: Check logs regularly during initial batches
3. **Verify prompt**: Ensure prompt produces desired JSON structure
4. **Check costs**: Use OpenAI dashboard to monitor spending
5. **Backup originals**: Keep original images safe before processing
6. **Version control**: Track changes to prompt and configuration

## FAQ

**Q: Can I process images in parallel?**  
A: OpenAI processes batches in parallel automatically. Don't run multiple upload scripts simultaneously.

**Q: How long does processing take?**  
A: Typically 6-12 hours for 1000 images, depending on OpenAI's queue.

**Q: Can I cancel a batch?**  
A: Use OpenAI API or dashboard to cancel. Then remove the tracking file from `./batch/`.

**Q: What if my computer shuts down during polling?**  
A: Safe! Batch tracking files persist. Just run download script again.

**Q: Can I change the JSON structure?**  
A: Yes! Edit `vision_prompt.txt` and adjust validation in `validate_json()`.

## Support

For issues or questions:
1. Check logs in `./batch/logs/`
2. Review OpenAI batch status on dashboard
3. Verify image format and size requirements
4. Ensure API key has sufficient credits

## License

Use freely for your historical document processing needs!