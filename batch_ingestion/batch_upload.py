"""
Batch Image Upload Script for OpenAI Vision API
Uses gpt-5-mini with "detail": "low" for cost-effective initial processing
Includes extensive logging and image preprocessing
"""

import base64
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI
from PIL import Image
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths - UPDATE THESE FOR YOUR ENVIRONMENT
API_KEY_FILE = os.path.expanduser("~/api_key.txt")  # Path to OpenAI API key
IMAGE_DIR = "./images"  # Directory containing images to process
PROMPT_FILE = "./vision_prompt.txt"  # File containing the extraction prompt
BATCH_DIR = "./batch"  # Directory for batch tracking files
LOG_DIR = os.path.join(BATCH_DIR, "logs")  # Log directory

# Processing settings
MODEL = "gpt-5-mini"  # Use the newer, more efficient model
DETAIL_LEVEL = "low"  # Start with low detail for cost efficiency
MAX_TOKENS = 4000  # Maximum tokens for response
DESCRIPTION = "Historical document OCR (low detail)"

# Image settings
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
MAX_IMAGE_SIZE_MB = 20  # OpenAI limit is 20MB per image
TARGET_MAX_DIMENSION = 2048  # Resize if larger (for efficiency)

# Batch settings
MAX_BATCH_SIZE_BYTES = 95 * 1024 * 1024  # 95 MB (under 100 MB limit)
MAX_BATCHES = 100  # Maximum number of batches to create
JSONL_FILE_BASE = "batchinput"

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging() -> logging.Logger:
    """Configure comprehensive logging."""
    os.makedirs(LOG_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"upload_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger("batch_upload")
    logger.setLevel(logging.DEBUG)
    
    # File handler (detailed)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler (less verbose)
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

def read_prompt(prompt_file: str) -> str:
    """Read the extraction prompt from a file."""
    logger.info(f"Reading prompt from: {prompt_file}")
    try:
        with open(prompt_file, "r") as file:
            prompt = file.read().strip()
        logger.debug(f"Prompt loaded ({len(prompt)} characters)")
        return prompt
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {prompt_file}")
        raise
    except Exception as e:
        logger.error(f"Error reading prompt file: {e}")
        raise

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

def get_image_size_mb(image_path: str) -> float:
    """Get image file size in megabytes."""
    size_bytes = os.path.getsize(image_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

def preprocess_image(image_path: str) -> Tuple[bool, str]:
    """
    Check and preprocess image if needed.
    Returns: (is_valid, reason_if_invalid)
    """
    try:
        # Check file size
        size_mb = get_image_size_mb(image_path)
        if size_mb > MAX_IMAGE_SIZE_MB:
            logger.warning(f"Image too large ({size_mb:.2f} MB): {image_path}")
            return False, f"File size {size_mb:.2f} MB exceeds {MAX_IMAGE_SIZE_MB} MB limit"
        
        # Check if image can be opened
        with Image.open(image_path) as img:
            width, height = img.size
            logger.debug(f"Image dimensions: {width}x{height} - {image_path}")
            
            # Check if dimensions are reasonable
            if width < 100 or height < 100:
                logger.warning(f"Image too small ({width}x{height}): {image_path}")
                return False, f"Image too small ({width}x{height})"
            
            # Note: We don't resize here because OpenAI handles it efficiently
            # with the "detail" parameter
            
        return True, ""
        
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {e}")
        return False, str(e)

def image_to_base64(image_path: str) -> Tuple[str, str]:
    """
    Convert image to base64 string.
    Returns: (base64_string, mime_type)
    """
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    
    encoded = base64.b64encode(image_data).decode("utf-8")
    
    # Determine MIME type from extension
    ext = Path(image_path).suffix.lower()
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }
    mime = mime_types.get(ext, "image/jpeg")
    
    return encoded, mime

# ============================================================================
# BATCH CREATION FUNCTIONS
# ============================================================================

def create_jsonl_entry(
    image_path: str,
    prompt: str,
    custom_id: str,
    model: str = MODEL,
    detail: str = DETAIL_LEVEL
) -> Tuple[Dict, int]:
    """
    Create a JSONL entry for batch processing using an image file.
    Returns: (jsonl_entry_dict, estimated_size_bytes)
    """
    logger.debug(f"Creating JSONL entry for: {image_path}")
    
    # Convert image to base64
    try:
        base64_image, mime_type = image_to_base64(image_path)
        image_url = f"data:{mime_type};base64,{base64_image}"
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        raise
    
    # Create request body
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "detail": detail
                        }
                    }
                ]
            }
        ],
        "max_tokens": MAX_TOKENS
    }
    
    entry = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body
    }
    
    # Estimate size (base64 is roughly file size * 1.37)
    file_size = os.path.getsize(image_path)
    estimated_size = int(file_size * 1.4) + 1000  # Add overhead for JSON structure
    
    logger.debug(f"JSONL entry created (est. {estimated_size} bytes)")
    return entry, estimated_size

def create_custom_id(image_path: str) -> str:
    """Create a custom ID by replacing problematic characters in the image path."""
    return image_path.replace("/", "|").replace("\\", "|")

def has_corresponding_json(image_path: str) -> bool:
    """Check if a corresponding JSON file exists."""
    json_path = image_path + ".json"
    exists = os.path.exists(json_path)
    if exists:
        logger.debug(f"JSON already exists: {json_path}")
    return exists

def get_image_files(image_dir: str) -> List[str]:
    """Recursively get all image files in the directory."""
    logger.info(f"Scanning for images in: {image_dir}")
    
    image_files = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if Path(file).suffix.lower() in IMAGE_EXTENSIONS:
                full_path = os.path.join(root, file)
                image_files.append(full_path)
    
    logger.info(f"Found {len(image_files)} image files")
    return image_files

def write_jsonl_file(entries: List[Dict], output_file: str) -> None:
    """Write multiple JSONL entries to a file."""
    logger.info(f"Writing {len(entries)} entries to: {output_file}")
    
    try:
        with open(output_file, "w") as jsonl_file:
            for entry in entries:
                jsonl_file.write(json.dumps(entry, separators=(',', ':')) + "\n")
        
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        logger.info(f"JSONL file written successfully ({file_size_mb:.2f} MB)")
    except Exception as e:
        logger.error(f"Error writing JSONL file: {e}")
        raise

def upload_jsonl_file(api_key: str, jsonl_file_path: str) -> object:
    """Upload the JSONL file to OpenAI for batch processing."""
    logger.info(f"Uploading JSONL file to OpenAI: {jsonl_file_path}")
    
    try:
        client = OpenAI(api_key=api_key)
        with open(jsonl_file_path, "rb") as jsonl_file:
            file_obj = client.files.create(
                file=jsonl_file,
                purpose="batch"
            )
        
        logger.info(f"File uploaded successfully. File ID: {file_obj.id}")
        return file_obj
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise

def create_batch(api_key: str, batch_input_file_id: str, description: str) -> object:
    """Create a batch using the uploaded JSONL file ID."""
    logger.info(f"Creating batch with file ID: {batch_input_file_id}")
    
    try:
        client = OpenAI(api_key=api_key)
        batch = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": description}
        )
        
        logger.info(f"Batch created successfully. Batch ID: {batch.id}")
        return batch
    except Exception as e:
        logger.error(f"Error creating batch: {e}")
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

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    """Main batch upload process."""
    logger.info("=" * 80)
    logger.info("BATCH IMAGE UPLOAD - STARTING")
    logger.info("=" * 80)
    logger.info(f"Model: {MODEL}")
    logger.info(f"Detail level: {DETAIL_LEVEL}")
    logger.info(f"Image directory: {IMAGE_DIR}")
    
    # Read API key
    try:
        api_key = read_api_key(API_KEY_FILE)
    except Exception as e:
        logger.error(f"Failed to read API key: {e}")
        return
    
    # Read prompt
    try:
        prompt = read_prompt(PROMPT_FILE)
    except Exception as e:
        logger.error(f"Failed to read prompt: {e}")
        return
    
    # Get all image files
    all_images = get_image_files(IMAGE_DIR)
    if not all_images:
        logger.warning("No image files found. Exiting.")
        return
    
    # Filter: only images without existing JSON
    logger.info("Filtering images that need processing...")
    filtered_images = [img for img in all_images if not has_corresponding_json(img)]
    logger.info(f"Images to process: {len(filtered_images)} (skipped {len(all_images) - len(filtered_images)} with existing JSON)")
    
    if not filtered_images:
        logger.info("No images need processing. Exiting.")
        return
    
    # Preprocess and validate images
    logger.info("Preprocessing and validating images...")
    valid_images = []
    invalid_images = []
    
    for img_path in tqdm(filtered_images, desc="Validating images"):
        is_valid, reason = preprocess_image(img_path)
        if is_valid:
            valid_images.append(img_path)
        else:
            invalid_images.append((img_path, reason))
            logger.warning(f"Invalid image skipped: {img_path} - {reason}")
    
    logger.info(f"Valid images: {len(valid_images)}")
    logger.info(f"Invalid images: {len(invalid_images)}")
    
    if not valid_images:
        logger.error("No valid images to process. Exiting.")
        return
    
    # Create batch directory
    os.makedirs(BATCH_DIR, exist_ok=True)
    
    # Process images in batches
    total_files_processed = 0
    batch_index = 0
    remaining_images = valid_images.copy()
    
    logger.info(f"Starting batch creation (max {MAX_BATCHES} batches)...")
    
    while batch_index < MAX_BATCHES and remaining_images:
        logger.info(f"\n--- Batch {batch_index + 1} ---")
        
        current_batch_size = 0
        batch_images = []
        batch_entries = []
        
        # Fill batch up to size limit
        for img_path in tqdm(remaining_images, desc=f"Building batch {batch_index + 1}"):
            try:
                entry, estimated_size = create_jsonl_entry(
                    img_path,
                    prompt,
                    create_custom_id(img_path),
                    model=MODEL,
                    detail=DETAIL_LEVEL
                )
                
                # Check if adding this would exceed batch size
                if current_batch_size + estimated_size > MAX_BATCH_SIZE_BYTES:
                    logger.debug(f"Batch size limit reached ({current_batch_size} bytes)")
                    break
                
                batch_images.append(img_path)
                batch_entries.append(entry)
                current_batch_size += estimated_size
                
            except Exception as e:
                logger.error(f"Error creating entry for {img_path}: {e}")
                continue
        
        if not batch_images:
            logger.warning("No images fit into batch size limit. Exiting.")
            break
        
        logger.info(f"Batch contains {len(batch_images)} images (~{current_batch_size / (1024*1024):.2f} MB)")
        
        # Write JSONL file
        jsonl_file = f"{JSONL_FILE_BASE}_batch_{batch_index + 1}.jsonl"
        try:
            write_jsonl_file(batch_entries, jsonl_file)
        except Exception as e:
            logger.error(f"Failed to write JSONL file: {e}")
            break
        
        # Upload to OpenAI
        try:
            batch_input_file = upload_jsonl_file(api_key, jsonl_file)
            batch = create_batch(api_key, batch_input_file.id, DESCRIPTION)
        except Exception as e:
            logger.error(f"Failed to create batch: {e}")
            break
        
        # Save batch details
        batch_details = serialize_batch(batch)
        batch_file_path = os.path.join(BATCH_DIR, f"{batch.id}.txt")
        
        try:
            with open(batch_file_path, "w") as batch_file:
                batch_file.write(json.dumps(batch_details, indent=2))
            logger.info(f"Batch details saved to: {batch_file_path}")
        except Exception as e:
            logger.error(f"Failed to save batch details: {e}")
        
        # Log batch summary
        logger.info(f"Batch ID: {batch.id}")
        logger.info(f"Status: {batch.status}")
        logger.info(f"Images in batch: {len(batch_images)}")
        
        # Delete JSONL file after upload
        try:
            os.remove(jsonl_file)
            logger.debug(f"Deleted temporary JSONL file: {jsonl_file}")
        except Exception as e:
            logger.warning(f"Failed to delete JSONL file: {e}")
        
        # Update progress
        total_files_processed += len(batch_images)
        remaining_images = remaining_images[len(batch_images):]
        batch_index += 1
    
    # Final summary
    logger.info("=" * 80)
    logger.info("BATCH UPLOAD COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total images found: {len(all_images)}")
    logger.info(f"Images with existing JSON: {len(all_images) - len(filtered_images)}")
    logger.info(f"Valid images to process: {len(valid_images)}")
    logger.info(f"Invalid images skipped: {len(invalid_images)}")
    logger.info(f"Images processed: {total_files_processed}")
    logger.info(f"Batches created: {batch_index}")
    logger.info(f"Images remaining: {len(remaining_images)}")
    
    if invalid_images:
        logger.info(f"\nInvalid images ({len(invalid_images)}):")
        for img_path, reason in invalid_images[:10]:  # Show first 10
            logger.info(f"  - {img_path}: {reason}")
        if len(invalid_images) > 10:
            logger.info(f"  ... and {len(invalid_images) - 10} more")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)