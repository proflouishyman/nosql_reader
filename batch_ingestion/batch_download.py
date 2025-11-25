"""
Batch Image Download Script for OpenAI Vision API
Downloads completed batch results, validates JSON, and tracks failures
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

from openai import OpenAI

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths - UPDATE THESE FOR YOUR ENVIRONMENT
API_KEY_FILE = os.path.expanduser("~/api_key.txt")
BATCH_DIR = "./batch"
COMPLETED_DIR = os.path.join(BATCH_DIR, "completed")
OUTPUT_DIR = os.path.join(BATCH_DIR, "batch_return")
EXTRACTION_DIR = os.path.join(BATCH_DIR, "batch_json_results")
LOG_DIR = os.path.join(BATCH_DIR, "logs")
FAILED_LIST_FILE = os.path.join(BATCH_DIR, "failed_images.txt")

# Polling settings
RETRY_LIMIT = 45  # Check status up to 45 times
RETRY_DELAY = 60  # Wait 60 seconds between checks

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging() -> logging.Logger:
    """Configure comprehensive logging."""
    os.makedirs(LOG_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"download_{timestamp}.log")
    
    logger = logging.getLogger("batch_download")
    logger.setLevel(logging.DEBUG)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

logger = setup_logging()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def read_api_key(api_key_file: str) -> str:
    """Read the OpenAI API key from a file."""
    logger.info(f"Reading API key from: {api_key_file}")
    try:
        with open(api_key_file, "r") as file:
            key = file.read().strip()
        logger.debug("API key loaded successfully")
        return key
    except FileNotFoundError:
        logger.error(f"API key file not found: {api_key_file}")
        raise
    except Exception as e:
        logger.error(f"Error reading API key: {e}")
        raise

def get_batch_status(api_key: str, batch_id: str) -> object:
    """Get the status of a batch."""
    logger.debug(f"Checking status of batch: {batch_id}")
    try:
        client = OpenAI(api_key=api_key)
        batch = client.batches.retrieve(batch_id)
        return batch
    except Exception as e:
        logger.error(f"Error retrieving batch status: {e}")
        raise

def download_file(api_key: str, file_id: str, output_dir: str, filename: str) -> str:
    """Download a file from OpenAI and save it to the specified directory."""
    logger.info(f"Downloading file {file_id} as {filename}")
    
    try:
        client = OpenAI(api_key=api_key)
        response = client.files.content(file_id)
        
        os.makedirs(output_dir, exist_ok=True)
        
        output_file_path = os.path.join(output_dir, f"{filename}.jsonl")
        with open(output_file_path, "wb") as output_file:
            for chunk in response.iter_bytes():
                output_file.write(chunk)
        
        file_size_mb = os.path.getsize(output_file_path) / (1024 * 1024)
        logger.info(f"File downloaded successfully ({file_size_mb:.2f} MB): {output_file_path}")
        return output_file_path
        
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise

def serialize_batch(batch: object) -> Dict:
    """Serialize the batch object into a JSON-serializable dictionary."""
    return {
        "id": batch.id,
        "object": batch.object,
        "endpoint": batch.endpoint,
        "errors": str(batch.errors) if batch.errors else None,
        "input_file_id": batch.input_file_id,
        "completion_window": batch.completion_window,
        "status": batch.status,
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
        "created_at": batch.created_at,
        "in_progress_at": batch.in_progress_at,
        "expires_at": batch.expires_at,
        "completed_at": batch.completed_at,
        "failed_at": batch.failed_at,
        "expired_at": batch.expired_at,
        "request_counts": {
            "total": batch.request_counts.total,
            "completed": batch.request_counts.completed,
            "failed": batch.request_counts.failed
        },
        "metadata": batch.metadata
    }

def read_batch_ids(batch_dir: str) -> List[str]:
    """Read batch IDs from filenames in the specified directory."""
    logger.info(f"Reading batch IDs from: {batch_dir}")
    
    batch_ids = []
    if not os.path.exists(batch_dir):
        logger.warning(f"Batch directory does not exist: {batch_dir}")
        return batch_ids
    
    for filename in os.listdir(batch_dir):
        if filename.endswith(".txt"):
            batch_id = filename.replace(".txt", "")
            batch_ids.append(batch_id)
    
    logger.info(f"Found {len(batch_ids)} pending batches")
    return batch_ids

# ============================================================================
# JSON VALIDATION AND EXTRACTION
# ============================================================================

def validate_json(json_str: str) -> tuple[bool, str, dict]:
    """
    Validate JSON structure and content.
    Returns: (is_valid, error_message, parsed_json)
    """
    try:
        # Try to parse JSON
        data = json.loads(json_str)
        
        # Check for required fields (based on your prompt structure)
        required_fields = ["ocr_text", "summary"]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return False, f"Missing required fields: {missing_fields}", data
        
        # Check if OCR text is empty
        if not data.get("ocr_text", "").strip():
            return False, "OCR text is empty", data
        
        # Check for common error patterns
        ocr_text = data.get("ocr_text", "")
        if len(ocr_text) < 10:
            return False, "OCR text too short (possible extraction failure)", data
        
        # Validation passed
        return True, "", data
        
    except json.JSONDecodeError as e:
        return False, f"JSON parse error: {e}", {}
    except Exception as e:
        return False, f"Validation error: {e}", {}

def clean_json_response(content: str) -> str:
    """Remove markdown code fences if present."""
    content = content.strip()
    
    # Remove ```json and ``` markers
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    
    if content.endswith("```"):
        content = content[:-3]
    
    # Find first '{' and last '}'
    first_brace = content.find('{')
    last_brace = content.rfind('}')
    
    if first_brace != -1 and last_brace != -1:
        content = content[first_brace:last_brace + 1]
    
    return content.strip()

def extract_json_lines(
    result_file: str,
    output_dir: str,
    failed_images: Set[str]
) -> tuple[int, int, int]:
    """
    Extract each line from the result file into individual JSON files.
    Returns: (total_count, valid_count, invalid_count)
    """
    logger.info(f"Extracting JSON from: {result_file}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    total_count = 0
    valid_count = 0
    invalid_count = 0
    
    try:
        with open(result_file, "r") as file:
            for line_num, line in enumerate(file, 1):
                total_count += 1
                
                try:
                    # Parse the batch result line
                    data = json.loads(line)
                    custom_id = data.get("custom_id", "unknown_id")
                    
                    # Reconstruct original image path
                    original_path = custom_id.replace("|", os.sep)
                    
                    # Get the response content
                    response_body = data.get("response", {}).get("body", {})
                    choices = response_body.get("choices", [])
                    
                    if not choices:
                        logger.warning(f"No choices in response for: {custom_id}")
                        failed_images.add(original_path)
                        invalid_count += 1
                        continue
                    
                    content = choices[0].get("message", {}).get("content", "")
                    if not content:
                        logger.warning(f"Empty content for: {custom_id}")
                        failed_images.add(original_path)
                        invalid_count += 1
                        continue
                    
                    # Clean and validate JSON
                    cleaned_content = clean_json_response(content)
                    is_valid, error_msg, parsed_json = validate_json(cleaned_content)
                    
                    if not is_valid:
                        logger.warning(f"Invalid JSON for {custom_id}: {error_msg}")
                        failed_images.add(original_path)
                        invalid_count += 1
                        
                        # Still save the invalid JSON for inspection
                        output_file_path = original_path + ".json"
                        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                        with open(output_file_path, "w") as output_file:
                            json.dump({"_validation_error": error_msg, "raw_content": cleaned_content}, output_file, indent=2)
                        
                        continue
                    
                    # Save valid JSON
                    output_file_path = original_path + ".json"
                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                    
                    with open(output_file_path, "w") as output_file:
                        json.dump(parsed_json, output_file, indent=2)
                    
                    valid_count += 1
                    logger.debug(f"Saved valid JSON: {output_file_path}")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse line {line_num}: {e}")
                    invalid_count += 1
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")
                    invalid_count += 1
        
        logger.info(f"Extraction complete: {valid_count} valid, {invalid_count} invalid out of {total_count} total")
        return total_count, valid_count, invalid_count
        
    except Exception as e:
        logger.error(f"Error reading result file: {e}")
        raise

def process_result_files(
    result_dir: str,
    extraction_dir: str,
    batch_id: str,
    failed_images: Set[str]
) -> tuple[int, int, int]:
    """
    Process all JSONL files for a batch and extract them to individual JSON files.
    Returns: (total, valid, invalid)
    """
    logger.info(f"Processing results for batch: {batch_id}")
    
    extraction_output_dir = os.path.join(extraction_dir, batch_id)
    os.makedirs(extraction_output_dir, exist_ok=True)
    
    total = 0
    valid = 0
    invalid = 0
    
    for filename in os.listdir(result_dir):
        if filename.endswith(".jsonl") and batch_id in filename:
            result_file_path = os.path.join(result_dir, filename)
            logger.info(f"Processing file: {result_file_path}")
            
            t, v, i = extract_json_lines(result_file_path, extraction_output_dir, failed_images)
            total += t
            valid += v
            invalid += i
    
    logger.info(f"Batch {batch_id} processed: {valid}/{total} valid")
    return total, valid, invalid

def save_failed_images_list(failed_images: Set[str]) -> None:
    """Save the list of failed images for retry processing."""
    if not failed_images:
        logger.info("No failed images to save")
        return
    
    logger.info(f"Saving {len(failed_images)} failed images to: {FAILED_LIST_FILE}")
    
    try:
        with open(FAILED_LIST_FILE, "w") as f:
            for img_path in sorted(failed_images):
                f.write(img_path + "\n")
        
        logger.info(f"Failed images list saved: {FAILED_LIST_FILE}")
    except Exception as e:
        logger.error(f"Error saving failed images list: {e}")

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    """Main batch download process."""
    logger.info("=" * 80)
    logger.info("BATCH IMAGE DOWNLOAD - STARTING")
    logger.info("=" * 80)
    
    # Read API key
    try:
        api_key = read_api_key(API_KEY_FILE)
    except Exception as e:
        logger.error(f"Failed to read API key: {e}")
        return
    
    # Read batch IDs
    batch_ids = read_batch_ids(BATCH_DIR)
    if not batch_ids:
        logger.info("No pending batches found. Exiting.")
        return
    
    logger.info(f"Monitoring {len(batch_ids)} batches")
    
    # Track attempts
    attempts = {batch_id: 0 for batch_id in batch_ids}
    
    # Track failed images across all batches
    failed_images: Set[str] = set()
    
    # Track statistics
    total_batches_completed = 0
    total_images_processed = 0
    total_valid = 0
    total_invalid = 0
    
    # Poll batches
    while batch_ids:
        logger.info(f"\n--- Polling {len(batch_ids)} batches ---")
        
        for batch_id in batch_ids[:]:  # Use copy to allow modifications
            logger.info(f"\nChecking batch: {batch_id} (attempt {attempts[batch_id] + 1}/{RETRY_LIMIT})")
            
            try:
                # Get batch status
                batch = get_batch_status(api_key, batch_id)
                batch_details = serialize_batch(batch)
                
                logger.info(f"Status: {batch.status}")
                logger.info(f"Requests: {batch.request_counts.completed}/{batch.request_counts.total} completed")
                
                if batch.status == "completed":
                    logger.info(f"✓ Batch {batch_id} completed successfully")
                    
                    # Download output file
                    if batch.output_file_id:
                        try:
                            output_file_path = download_file(
                                api_key,
                                batch.output_file_id,
                                OUTPUT_DIR,
                                f"batch_{batch_id}"
                            )
                            
                            # Extract and validate JSON files
                            total, valid, invalid = process_result_files(
                                OUTPUT_DIR,
                                EXTRACTION_DIR,
                                batch_id,
                                failed_images
                            )
                            
                            total_images_processed += total
                            total_valid += valid
                            total_invalid += invalid
                            
                        except Exception as e:
                            logger.error(f"Error processing batch output: {e}")
                    
                    # Move batch file to completed directory
                    os.makedirs(COMPLETED_DIR, exist_ok=True)
                    batch_txt_file = os.path.join(BATCH_DIR, f"{batch_id}.txt")
                    completed_txt_file = os.path.join(COMPLETED_DIR, f"{batch_id}.txt")
                    
                    if os.path.exists(batch_txt_file):
                        os.rename(batch_txt_file, completed_txt_file)
                        logger.info(f"Moved batch file to: {completed_txt_file}")
                    
                    # Remove from tracking
                    batch_ids.remove(batch_id)
                    del attempts[batch_id]
                    total_batches_completed += 1
                    
                elif batch.status == "failed":
                    logger.error(f"✗ Batch {batch_id} failed")
                    logger.error(f"Errors: {batch.errors}")
                    batch_ids.remove(batch_id)
                    del attempts[batch_id]
                    
                elif batch.status == "expired":
                    logger.warning(f"⚠ Batch {batch_id} expired")
                    batch_ids.remove(batch_id)
                    del attempts[batch_id]
                    
                else:
                    # Still processing
                    logger.info(f"⋯ Batch {batch_id} is {batch.status}")
                    
                    if attempts[batch_id] >= RETRY_LIMIT:
                        logger.warning(f"Reached maximum retry limit for batch {batch_id}")
                        batch_ids.remove(batch_id)
                        del attempts[batch_id]
                    else:
                        attempts[batch_id] += 1
                
            except Exception as e:
                logger.error(f"Error checking batch {batch_id}: {e}")
                attempts[batch_id] += 1
                
                if attempts[batch_id] >= RETRY_LIMIT:
                    logger.error(f"Too many errors for batch {batch_id}, removing from queue")
                    batch_ids.remove(batch_id)
                    del attempts[batch_id]
        
        # Wait before next poll
        if batch_ids:
            logger.info(f"\nWaiting {RETRY_DELAY} seconds before next check...")
            time.sleep(RETRY_DELAY)
    
    # Save failed images list
    save_failed_images_list(failed_images)
    
    # Final summary
    logger.info("=" * 80)
    logger.info("BATCH DOWNLOAD COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Batches completed: {total_batches_completed}")
    logger.info(f"Images processed: {total_images_processed}")
    logger.info(f"Valid JSON: {total_valid} ({total_valid/total_images_processed*100:.1f}%)" if total_images_processed > 0 else "Valid JSON: 0")
    logger.info(f"Invalid JSON: {total_invalid} ({total_invalid/total_images_processed*100:.1f}%)" if total_images_processed > 0 else "Invalid JSON: 0")
    
    if failed_images:
        logger.info(f"\n⚠ {len(failed_images)} images need retry with high detail")
        logger.info(f"Failed images list saved to: {FAILED_LIST_FILE}")
        logger.info("Run batch_retry_failed.py to reprocess these images")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)