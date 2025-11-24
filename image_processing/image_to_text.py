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
EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
RESUME = True
TIMEOUT = 40
RETRY_ATTEMPTS = 2  # Number of retry attempts for timeouts
RETRY_DELAY = 5  # Seconds to wait between retries

def image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_text_from_image(image_path, attempt=1):
    """Extract text from image using vision model"""
    image_base64 = image_to_base64(image_path)
    
    try:
        print(f" (attempt {attempt})", end="", flush=True)
        
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
            return response.json()['response'], None
        else:
            error_detail = f"Status {response.status_code}"
            try:
                error_json = response.json()
                if 'error' in error_json:
                    error_detail += f": {error_json['error']}"
            except:
                error_detail += f": {response.text[:100]}"
            return None, error_detail
            
    except requests.exceptions.Timeout:
        if attempt < RETRY_ATTEMPTS:
            print(f" timeout, retrying in {RETRY_DELAY}s...", end="", flush=True)
            time.sleep(RETRY_DELAY)
            return extract_text_from_image(image_path, attempt + 1)
        return None, f"Timeout after {attempt} attempts"
    except requests.exceptions.ConnectionError as e:
        return None, f"Connection error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {type(e).__name__}: {str(e)}"

def check_ollama_status():
    """Check if Ollama is running and responsive"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            print(f"Ollama is running. Available models: {model_names}")
            if MODEL not in model_names:
                print(f"WARNING: Model '{MODEL}' not found in available models!")
            return True
        else:
            print(f"Ollama responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"Cannot connect to Ollama: {e}")
        return False

def process_directory(directory_path):
    """Process all images in a directory recursively"""
    
    print(f"Processing directory: {directory_path}")
    
    # Check Ollama status first
    if not check_ollama_status():
        print("ERROR: Cannot connect to Ollama. Make sure it's running.")
        return
    
    # Track progress
    processed = 0
    skipped = 0
    failed = 0
    timeouts = 0
    start_time = time.time()
    
    # Create log file
    log_file = os.path.join(directory_path, "ocr_batch_log.txt")
    error_log_file = os.path.join(directory_path, "ocr_errors.txt")
    
    with open(log_file, "a") as log, open(error_log_file, "a") as error_log:
        log.write(f"\n\n=== Batch OCR Started at {datetime.now()} ===\n")
        error_log.write(f"\n\n=== Error Log Started at {datetime.now()} ===\n")
        
        image_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext.lower()) for ext in EXTENSIONS):
                    image_path = os.path.join(root, file)
                    image_files.append(image_path)
        
        print(f"Found {len(image_files)} images in {directory_path} and subdirectories")
        
        if len(image_files) == 0:
            print("No images found. Checking directory structure...")
            for root, dirs, files in os.walk(directory_path):
                print(f"Directory: {root}")
                print(f"  Subdirs: {dirs}")
                print(f"  Files: {files[:5]}...")  # Show first 5 files
        
        for i, image_path in enumerate(image_files, 1):
            # Get output path
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(os.path.dirname(image_path), f"{base_name}_ocr.txt")
            
            # Skip if already processed and resume is True
            if RESUME and os.path.exists(output_path):
                print(f"[{i}/{len(image_files)}] Skipping {os.path.basename(image_path)} - already processed")
                skipped += 1
                continue
            
            # Process image
            print(f"[{i}/{len(image_files)}] Processing {os.path.basename(image_path)}...", end="", flush=True)
            
            try:
                # Get file size for logging
                file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
                
                start = time.time()
                extracted_text, error = extract_text_from_image(image_path)
                elapsed = time.time() - start
                
                if extracted_text and not error:
                    # Save text
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(extracted_text)
                    
                    processed += 1
                    print(f" ✓ ({elapsed:.1f}s, {file_size_mb:.1f}MB)")
                    log.write(f"SUCCESS: {image_path} - {elapsed:.1f}s - {file_size_mb:.1f}MB\n")
                else:
                    failed += 1
                    if "Timeout" in str(error):
                        timeouts += 1
                    print(f" ✗ {error}")
                    log.write(f"FAILED: {image_path} - {error}\n")
                    error_log.write(f"{datetime.now()} - {image_path} - {error} - {file_size_mb:.1f}MB\n")
                    
            except Exception as e:
                failed += 1
                print(f" ✗ Exception: {str(e)}")
                log.write(f"ERROR: {image_path} - {str(e)}\n")
                error_log.write(f"{datetime.now()} - {image_path} - Exception: {str(e)}\n")
            
            # Show estimated time remaining
            if processed + failed > 0:
                avg_time = (time.time() - start_time) / (processed + failed)
                remaining = len(image_files) - (processed + failed + skipped)
                eta = remaining * avg_time
                print(f"    ETA: {eta/60:.1f} minutes remaining")
        
        # Summary
        total_time = time.time() - start_time
        summary = f"\n=== Summary ===\n"
        summary += f"Total images: {len(image_files)}\n"
        summary += f"Processed: {processed}\n"
        summary += f"Skipped: {skipped}\n"
        summary += f"Failed: {failed} (including {timeouts} timeouts)\n"
        summary += f"Time: {total_time/60:.1f} minutes\n"
        
        if processed + failed > 0:
            avg_time = total_time / (processed + failed)
            summary += f"Avg: {avg_time:.1f}s per image\n"
        else:
            summary += "Avg: N/A\n"
        
        print(summary)
        log.write(summary)
        error_log.write(f"\n{summary}")

# Usage
if __name__ == "__main__":
    process_directory(ROOT_DIRECTORY)