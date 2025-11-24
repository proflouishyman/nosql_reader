import os
import glob
import time
import json
import base64
import requests
from datetime import datetime

# Variables
MODEL = "llama3.2-vision:11b"
ROOT_DIRECTORY = "/data/lhyman6/nosql_project/nosql/archives/"
PATTERN = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
RESUME = True
TIMEOUT = 20

def image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_text_from_image(image_path):
    """Extract text from image using vision model"""
    image_base64 = image_to_base64(image_path)
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": MODEL,
                "prompt": "Extract ALL text from this image with high accuracy, preserving the original layout and formatting. Return only the text content, nothing else.",
                "images": [image_base64],
                "stream": False
            },
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error: {response.status_code}"
    except requests.exceptions.Timeout:
        return "Error: Timeout"
    except Exception as e:
        return f"Error: {str(e)}"

def process_directory(directory_path):
    """Process all images in a directory recursively"""
    
    print(f"Processing directory: {directory_path}")
    
    # Track progress
    processed = 0
    skipped = 0
    failed = 0
    start_time = time.time()
    
    # Create log file
    log_file = os.path.join(directory_path, "ocr_batch_log.txt")
    
    with open(log_file, "a") as log:
        log.write(f"\n\n=== Batch OCR Started at {datetime.now()} ===\n")
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if any(file.endswith(pattern) for pattern in PATTERN):
                    image_path = os.path.join(root, file)
                    
                    # Get output path
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    output_path = os.path.join(root, f"{base_name}_ocr.txt")
                    
                    # Skip if already processed and resume is True
                    if RESUME and os.path.exists(output_path):
                        print(f"Skipping {image_path} - already processed")
                        skipped += 1
                        continue
                    
                    # Process image
                    print(f"Processing {image_path}...", end="", flush=True)
                    
                    try:
                        start = time.time()
                        extracted_text = extract_text_from_image(image_path)
                        elapsed = time.time() - start
                        
                        if not extracted_text.startswith("Error:"):
                            # Save text
                            with open(output_path, "w", encoding="utf-8") as f:
                                f.write(extracted_text)
                            
                            processed += 1
                            print(f" ✓ ({elapsed:.1f}s)")
                            log.write(f"SUCCESS: {image_path} - {elapsed:.1f}s\n")
                        else:
                            failed += 1
                            print(f" ✗ {extracted_text}")
                            log.write(f"FAILED: {image_path} - {extracted_text}\n")
                            
                    except Exception as e:
                        failed += 1
                        print(f" ✗ Exception: {str(e)}")
                        log.write(f"ERROR: {image_path} - {str(e)}\n")
                    
                    # Show estimated time remaining
                    if processed > 0:
                        avg_time = (time.time() - start_time) / (processed + failed)
                        remaining = len([f for f in os.listdir(root) if any(f.endswith(pattern) for pattern in PATTERN)]) - 1
                        eta = remaining * avg_time
                        print(f"    ETA: {eta/60:.1f} minutes remaining")
        
        # Summary
        total_time = time.time() - start_time
        summary = f"\n=== Summary ===\nTotal: {processed + skipped + failed}\nProcessed: {processed}\nSkipped: {skipped}\nFailed: {failed}\nTime: {total_time/60:.1f} minutes\nAvg: {total_time/(processed+failed):.1f}s per image\n"
        
        print(summary)
        log.write(summary)

# Usage
if __name__ == "__main__":
    process_directory(ROOT_DIRECTORY)