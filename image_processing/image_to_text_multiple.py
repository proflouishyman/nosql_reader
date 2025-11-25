import os
import glob
import time
import json
import base64
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

# Variables
MODEL = "qwen3-vl:8b"
ROOT_DIRECTORY = "/data/lhyman6/nosql_project/nosql/archives/"
EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
RESUME = True
TIMEOUT = 60  # Increased timeout for first requests
MAX_WORKERS = 4
OLLAMA_PORTS = [11434, 11435, 11436, 11437]

# Queue to manage port allocation
port_queue = queue.Queue()
for port in OLLAMA_PORTS[:MAX_WORKERS]:
    port_queue.put(port)

def ensure_model_loaded(port):
    """Ensure model is loaded on the specified port"""
    try:
        # Check if model is already loaded
        response = requests.get(f"http://localhost:{port}/api/tags", timeout=5)
        if response.status_code == 200:
            models = [m['name'] for m in response.json().get('models', [])]
            if MODEL in models:
                return True
        
        print(f"Loading {MODEL} on port {port}...")
        # Make a dummy request to load the model
        response = requests.post(
            f"http://localhost:{port}/api/generate",
            json={
                "model": MODEL,
                "prompt": "test",
                "stream": False
            },
            timeout=300  # Long timeout for model loading
        )
        
        if response.status_code == 200:
            print(f"✓ Model loaded on port {port}")
            return True
        else:
            print(f"✗ Failed to load model on port {port}")
            return False
            
    except Exception as e:
        print(f"✗ Error loading model on port {port}: {e}")
        return False

def check_and_prepare_ollama():
    """Check all Ollama instances and ensure models are loaded"""
    print("Checking Ollama instances and loading models...")
    
    ready_ports = []
    for port in OLLAMA_PORTS[:MAX_WORKERS]:
        try:
            # First check if instance is running
            response = requests.get(f"http://localhost:{port}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✓ Ollama instance on port {port} is running")
                # Ensure model is loaded
                if ensure_model_loaded(port):
                    ready_ports.append(port)
            else:
                print(f"✗ Ollama instance on port {port} is not responding")
        except:
            print(f"✗ Cannot connect to Ollama on port {port}")
    
    return ready_ports

def extract_text_from_image_with_port(image_path, port):
    """Extract text from image using vision model on specific port"""
    image_base64 = image_to_base64(image_path)
    
    try:
        response = requests.post(
            f"http://localhost:{port}/api/generate",
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
            return None, f"Error: {response.status_code}"
            
    except requests.exceptions.Timeout:
        return None, f"Timeout on port {port}"
    except Exception as e:
        return None, f"Error on port {port}: {str(e)}"

def image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image(image_path, output_path):
    """Process a single image using an available Ollama instance"""
    # Get an available port
    port = port_queue.get()
    try:
        base_name = os.path.basename(image_path)
        print(f"Processing {base_name} on port {port}...")
        
        start_time = time.time()
        extracted_text, error = extract_text_from_image_with_port(image_path, port)
        elapsed = time.time() - start_time
        
        if extracted_text and not error:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(extracted_text)
            return True, f"SUCCESS: {base_name} - {elapsed:.1f}s on port {port}"
        else:
            return False, f"FAILED: {base_name} - {error}"
    finally:
        # Return port to queue
        port_queue.put(port)

def process_directory_parallel(directory_path):
    """Process all images in a directory recursively using parallel Ollama instances"""
    
    print(f"Processing directory: {directory_path}")
    
    # Check and prepare Ollama instances
    ready_ports = check_and_prepare_ollama()
    
    if not ready_ports:
        print("ERROR: No Ollama instances are ready. Please start them first.")
        return
    
    # Update queue with only ready ports
    while not port_queue.empty():
        port_queue.get()
    for port in ready_ports:
        port_queue.put(port)
    
    print(f"\nUsing {len(ready_ports)} Ollama instances: {ready_ports}")
    
    # Collect all image files
    image_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if any(file.lower().endswith(ext.lower()) for ext in EXTENSIONS):
                image_path = os.path.join(root, file)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_path = os.path.join(os.path.dirname(image_path), f"{base_name}_ocr.txt")
                
                if not (RESUME and os.path.exists(output_path)):
                    image_files.append((image_path, output_path))
    
    print(f"Found {len(image_files)} images to process")
    
    if len(image_files) == 0:
        print("No images to process.")
        return
    
    # Process images in parallel
    processed = 0
    failed = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=len(ready_ports)) as executor:
        # Submit all tasks
        future_to_image = {
            executor.submit(process_image, img_path, out_path): img_path 
            for img_path, out_path in image_files
        }
        
        # Process completed tasks
        for future in as_completed(future_to_image):
            image_path = future_to_image[future]
            try:
                success, message = future.result()
                if success:
                    processed += 1
                else:
                    failed += 1
                print(f"[{processed + failed}/{len(image_files)}] {message}")
                
                # Show progress
                if (processed + failed) % 5 == 0:
                    elapsed = time.time() - start_time
                    rate = (processed + failed) / elapsed
                    remaining = len(image_files) - (processed + failed)
                    eta = remaining / rate if rate > 0 else 0
                    print(f"Progress: {rate:.1f} images/sec, ETA: {eta/60:.1f} minutes")
                    
            except Exception as e:
                failed += 1
                print(f"Exception processing {image_path}: {str(e)}")
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n=== Summary ===")
    print(f"Total processed: {processed}")
    print(f"Failed: {failed}")
    print(f"Time: {total_time/60:.1f} minutes")
    if len(image_files) > 0:
        print(f"Average: {total_time/len(image_files):.1f}s per image")
        print(f"Throughput: {len(image_files)/total_time:.1f} images/sec")

# Usage
if __name__ == "__main__":
    process_directory_parallel(ROOT_DIRECTORY)