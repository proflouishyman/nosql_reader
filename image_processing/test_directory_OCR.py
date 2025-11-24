# batch_ocr.py
import os
import glob
import time
import json
import base64
import requests
from datetime import datetime

def image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_text_from_image(image_path, model="llama3.2-vision:11b"):
    """Extract text from image using vision model"""
    image_base64 = image_to_base64(image_path)
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": "Extract ALL text from this image. Return only the text content, nothing else.",
            "images": [image_base64],
            "stream": False
        },
        timeout=300
    )
    
    if response.status_code == 200:
        return response.json()['response']
    else:
        return f"Error: {response.status_code}"

def process_directory(directory_path, pattern="*.jpg", resume=True):
    """Process all images in a directory"""
    
    # Get all image files
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    image_files = []
    for p in patterns:
        image_files.extend(glob.glob(os.path.join(directory_path, p)))
    
    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))
    
    print(f"Found {len(image_files)} images in {directory_path}")
    
    # Track progress
    processed = 0
    skipped = 0
    failed = 0
    start_time = time.time()
    
    # Create log file
    log_file = os.path.join(directory_path, "ocr_batch_log.txt")
    
    with open(log_file, "a") as log:
        log.write(f"\n\n=== Batch OCR Started at {datetime.now()} ===\n")
        
        for i, image_path in enumerate(image_files, 1):
            # Get output path
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(directory_path, f"{base_name}_ocr.txt")
            
            # Skip if already processed and resume is True
            if resume and os.path.exists(output_path):
                print(f"[{i}/{len(image_files)}] Skipping {base_name} - already processed")
                skipped += 1
                continue
            
            # Process image
            print(f"[{i}/{len(image_files)}] Processing {base_name}...", end="", flush=True)
            
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
                    log.write(f"SUCCESS: {base_name} - {elapsed:.1f}s\n")
                else:
                    failed += 1
                    print(f" ✗ {extracted_text}")
                    log.write(f"FAILED: {base_name} - {extracted_text}\n")
                    
            except Exception as e:
                failed += 1
                print(f" ✗ Exception: {str(e)}")
                log.write(f"ERROR: {base_name} - {str(e)}\n")
            
            # Show estimated time remaining
            if processed > 0:
                avg_time = (time.time() - start_time) / (processed + failed)
                remaining = len(image_files) - i
                eta = remaining * avg_time
                print(f"    ETA: {eta/60:.1f} minutes remaining")
        
        # Summary
        total_time = time.time() - start_time
        summary = f"\n=== Summary ===\nTotal: {len(image_files)}\nProcessed: {processed}\nSkipped: {skipped}\nFailed: {failed}\nTime: {total_time/60:.1f} minutes\nAvg: {total_time/(processed+failed):.1f}s per image\n"
        
        print(summary)
        log.write(summary)

# Usage
if __name__ == "__main__":
    directory = "/data/lhyman6/nosql_project/nosql/archives/Paper_mini"
    
    # Process all images, skipping already processed ones
    process_directory(directory, resume=True)