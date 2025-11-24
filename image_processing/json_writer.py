import os
import glob
import json
import requests
import time
from datetime import datetime
from pathlib import Path

# Variables
MODEL = "qwen3-coder:30b"  # Model to use (e.g. "gpt-oss:20b")
DIRECTORY = "/data/lhyman6/nosql_project/nosql/archives/Paper_mini"  # Directory containing OCR text files
OCR_SUFFIX = "_ocr.txt"  # Suffix for OCR text files
JSON_SUFFIX = "_structured.json"  # Suffix for JSON output files
PROMPT_FILE = "prompt.txt"  # File containing the prompt
TIMEOUT = 20  # Timeout for API calls in seconds

class BatchJSONConverter:
    def __init__(self, model=MODEL):
        self.model = model
        self.check_model()
        self.prompt = self.read_prompt()
    
    def read_prompt(self):
        """Read prompt from prompt.txt"""
        try:
            with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading prompt: {e}")
            return ""
    
    def check_model(self):
        """Verify model is available"""
        try:
            response = requests.get("http://localhost:11434/api/tags")
            models = [m['name'] for m in response.json().get('models', [])]
            if self.model not in models:
                print(f"Warning: {self.model} not found. Available models: {models}")
                if models:
                    self.model = models[0]
                    print(f"Using {self.model} instead")
        except Exception as e:
            print(f"Error checking models: {e}")
    
    def text_to_json(self, text_content, source_file=""):
        """Convert text to structured JSON"""
        
        prompt = f"""{self.prompt}
Text:
{text_content}
"""
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 4096  # Allow longer responses
                    }
                },
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()['response']
                
                # Clean up formatting
                result = result.strip()
                if "```json" in result:
                    result = result.split("```json")[1].split("```")[0]
                elif "```" in result:
                    result = result.split("```")[1].split("```")[0]
                
                # Validate JSON
                try:
                    parsed = json.loads(result.strip())
                    return json.dumps(parsed, indent=2), None
                except json.JSONDecodeError as e:
                    return result, f"JSON parse error: {str(e)}"
            else:
                return None, f"API error: {response.status_code}"
                
        except requests.exceptions.Timeout:
            return None, "Timeout error: API call took too long"
        except Exception as e:
            return None, f"Request error: {str(e)}"
    
    def process_directory(self, directory=DIRECTORY, ocr_suffix=OCR_SUFFIX, json_suffix=JSON_SUFFIX):
        """Process all OCR text files in directory"""
        
        # Find all OCR text files
        ocr_files = glob.glob(os.path.join(directory, f"*{ocr_suffix}"))
        
        if not ocr_files:
            print(f"No OCR files found with suffix '{ocr_suffix}' in {directory}")
            return
        
        print(f"Found {len(ocr_files)} OCR text files to process")
        print(f"Using model: {self.model}")
        print(f"Started at {datetime.now().strftime('%H:%M:%S')}\n")
        
        # Track statistics
        success_count = 0
        skip_count = 0
        error_count = 0
        errors = []
        start_time = time.time()
        
        # Process each file
        for i, ocr_file in enumerate(ocr_files, 1):
            base_name = os.path.basename(ocr_file).replace(ocr_suffix, "")
            json_file = ocr_file.replace(ocr_suffix, json_suffix)
            
            # Skip if JSON already exists
            if os.path.exists(json_file):
                print(f"[{i}/{len(ocr_files)}] ⟳ {base_name} - JSON already exists")
                skip_count += 1
                continue
            
            print(f"[{i}/{len(ocr_files)}] Processing {base_name}...", end="", flush=True)
            
            try:
                # Read OCR text
                with open(ocr_file, 'r', encoding='utf-8') as f:
                    text_content = f.read().strip()
                
                if not text_content:
                    print(" ✗ Empty file")
                    error_count += 1
                    errors.append(f"{base_name}: Empty OCR file")
                    continue
                
                # Convert to JSON
                start = time.time()
                json_content, error = self.text_to_json(text_content, base_name)
                elapsed = time.time() - start
                
                if json_content and not error:
                    # Save JSON file
                    with open(json_file, 'w', encoding='utf-8') as f:
                        f.write(json_content)
                    
                    success_count += 1
                    print(f" ✓ ({elapsed:.1f}s)")
                    
                    # Show preview of JSON structure
                    try:
                        preview = json.loads(json_content)
                        keys = list(preview.keys())[:5]  # First 5 keys
                        print(f"    Keys: {keys}")
                    except:
                        pass
                else:
                    error_count += 1
                    print(f" ✗ {error}")
                    errors.append(f"{base_name}: {error}")
                    
                    # Save raw output for debugging
                    if json_content:
                        error_file = json_file.replace('.json', '_error.json')
                        with open(error_file, 'w', encoding='utf-8') as f:
                            f.write(json_content)
                
            except Exception as e:
                error_count += 1
                print(f" ✗ Exception: {str(e)}")
                errors.append(f"{base_name}: {str(e)}")
            
            # Progress update every 10 files
            if i % 10 == 0:
                elapsed_total = time.time() - start_time
                rate = i / elapsed_total
                remaining = (len(ocr_files) - i) / rate
                print(f"\n  Progress: {i}/{len(ocr_files)} files, "
                      f"Rate: {rate:.1f} files/sec, "
                      f"ETA: {remaining/60:.1f} minutes\n")
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Completed at {datetime.now().strftime('%H:%M:%S')}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Files processed: {len(ocr_files)}")
        print(f"Success: {success_count}")
        print(f"Skipped: {skip_count}")
        print(f"Errors: {error_count}")
        
        if errors:
            print(f"\nErrors encountered:")
            for error in errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")
        
        # Save summary
        summary_file = os.path.join(directory, "json_conversion_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"JSON Conversion Summary\n")
            f.write(f"Generated at: {datetime.now()}\n")
            f.write(f"Directory: {directory}\n")
            f.write(f"Model used: {self.model}\n")
            f.write(f"Total files: {len(ocr_files)}\n")
            f.write(f"Success: {success_count}\n")
            f.write(f"Skipped: {skip_count}\n")
            f.write(f"Errors: {error_count}\n")
            f.write(f"Total time: {total_time/60:.1f} minutes\n")
            if errors:
                f.write(f"\nErrors:\n")
                for error in errors:
                    f.write(f"  - {error}\n")
        
        print(f"\nSummary saved to: {summary_file}")

# Utility function to validate all JSON files
def validate_json_files(directory=DIRECTORY, json_suffix=JSON_SUFFIX):
    """Validate all JSON files in directory"""
    json_files = glob.glob(os.path.join(directory, f"*{json_suffix}"))
    
    print(f"Validating {len(json_files)} JSON files...")
    
    valid = 0
    invalid = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json.load(f)
            valid += 1
        except Exception as e:
            invalid.append((os.path.basename(json_file), str(e)))
    
    print(f"Valid JSON files: {valid}")
    print(f"Invalid JSON files: {len(invalid)}")
    
    if invalid:
        print("\nInvalid files:")
        for filename, error in invalid[:5]:
            print(f"  - {filename}: {error}")

# Usage
if __name__ == "__main__":
    # Initialize converter
    converter = BatchJSONConverter()
    
    # Process directory
    converter.process_directory()
    
    # Optionally validate all JSON files
    print("\n" + "="*60 + "\n")
    validate_json_files()