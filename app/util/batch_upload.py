import json
import os
from openai import OpenAI
from tqdm import tqdm

# untested

# Variables
API_KEY_FILE = "G:/My Drive/2024-2025/coding/api_key.txt"
TEXT_DIR = "../broken_texts"  # Directory containing text files
PROMPT_FILE = "rolls_json_prompt.txt"  # File containing the prompt
API_URL = "https://api.openai.com/v1/chat/completions"
BATCH_DIR = "./batch"  # Directory for batch output files
DESCRIPTION = "JSON correction"
VALID_TEXT_TYPES = [".txt"]  # Only process .txt files
MAX_BATCH_SIZE_BYTES = 95 * 1024 * 1024  # slightly less than 100 MB
MAX_BATCHES = 100  # Set the maximum number of batches to process
JSONL_FILE_BASE = "batchinput"

def read_prompt(prompt_file):
    """Read the prompt from a file."""
    with open(prompt_file, "r") as file:
        return file.read().strip()

def create_jsonl_entry(text_file_path, prompt, custom_id, model="gpt-4o-mini"): # DONT CHANGE MODEL NAME
    """Create a JSONL entry for batch processing using a text file."""
    with open(text_file_path, "r") as text_file:
        text_content = text_file.read()
    
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
                        "type": "text",
                        "text": text_content
                    }
                ]
            }
        ],
        "max_tokens": 4000  # Set max_tokens to 4000 to ensure the full response is received
    }
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body
    }, len(text_content)

def get_files(text_dir):
    """Get all files in the specified directory."""
    return [os.path.join(text_dir, f) for f in os.listdir(text_dir)]

def get_primary_basename(file_path):
    """Return the primary base name of the file, cutting off at the first period."""
    return file_path.split('/')[-1].split('.', 1)[0]

def has_corresponding_json(file_path):
    """Check if a corresponding JSON file exists."""
    json_file_path = file_path.rsplit('.', 1)[0] + ".json"  # Adjusted for text-based filenames
    return os.path.exists(json_file_path)

def create_custom_id(text_path):
    """Create a custom ID by replacing problematic characters in the text file path."""
    return text_path.replace("/", "|").replace("\\", "|")

def write_jsonl_file(entries, output_file):
    """Write multiple JSONL entries to a file."""
    with open(output_file, "w") as jsonl_file:
        for entry in entries:
            jsonl_file.write(json.dumps(entry, separators=(',', ':')) + "\n")  # Compress JSON

def upload_jsonl_file(api_key, jsonl_file_path):
    """Upload the JSONL file to OpenAI for batch processing."""
    client = OpenAI(api_key=api_key)
    with open(jsonl_file_path, "rb") as jsonl_file:
        return client.files.create(
            file=jsonl_file,  # Use the file object directly
            purpose="batch"
        )

def create_batch(api_key, batch_input_file_id, description):
    """Create a batch using the uploaded JSONL file ID."""
    client = OpenAI(api_key=api_key)
    return client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": description
        }
    )

def serialize_batch(batch):
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

if __name__ == "__main__":
    print("Reading API key from file...")
    with open(API_KEY_FILE, "r") as file:
        API_KEY = file.read().strip()
    print("API key successfully read.")

    print("Reading prompt from file...")
    PROMPT = read_prompt(PROMPT_FILE)
    print(f"Prompt: {PROMPT}")

    print(f"Collecting files from directory: {TEXT_DIR}")
    all_files = get_files(TEXT_DIR)
    all_basenames = {get_primary_basename(f) for f in tqdm(all_files, desc="Processing files")}

    print(f"Found {len(all_files)} files.")

    # Filter to keep only text files without a corresponding .json file
    filtered_text_files = [
        f for f in all_files
        if f.endswith(tuple(VALID_TEXT_TYPES)) and not has_corresponding_json(f)
    ]

    print(f"Filtered text files to {len(filtered_text_files)} for processing.")

    total_files_processed = 0
    batch_index = 0

    while batch_index < MAX_BATCHES and filtered_text_files:
        current_batch_size = 0
        valid_batch_files = []
        batch_entries = []

        for file in tqdm(filtered_text_files, desc=f"Processing batch {batch_index + 1}"):
            entry, text_data_size = create_jsonl_entry(file, PROMPT, create_custom_id(file))
            if current_batch_size + text_data_size > MAX_BATCH_SIZE_BYTES:
                break
            valid_batch_files.append(file)
            batch_entries.append(entry)
            current_batch_size += text_data_size

        if not valid_batch_files:
            print("No files fit into the batch size limit. Exiting.")
            break

        jsonl_file = f"{JSONL_FILE_BASE}_batch_{batch_index + 1}.jsonl"
        write_jsonl_file(batch_entries, jsonl_file)
        print(f"JSONL file {jsonl_file} created and uploaded successfully.")

        print("Creating batch...")
        batch_input_file = upload_jsonl_file(API_KEY, jsonl_file)
        batch = create_batch(API_KEY, batch_input_file.id, DESCRIPTION)
        print("Batch created successfully. Batch details:")

        batch_details = serialize_batch(batch)
        print(json.dumps(batch_details, indent=2))

        # Save the batch details as a text file in the batch directory
        if not os.path.exists(BATCH_DIR):
            os.makedirs(BATCH_DIR)
        batch_file_path = os.path.join(BATCH_DIR, f"{batch.id}.txt")
        with open(batch_file_path, "w") as batch_file:
            batch_file.write(json.dumps(batch_details, indent=2))

        print(f"Batch details saved to {batch_file_path}")

        # Delete the JSONL file after upload
        if os.path.exists(jsonl_file):
            os.remove(jsonl_file)
            print(f"Deleted JSONL file {jsonl_file}")

        # Adjust the length of filtered_text_files to remove processed files
        total_files_processed += len(valid_batch_files)
        filtered_text_files = filtered_text_files[len(valid_batch_files):]
        batch_index += 1

    print(f"Total number of files processed: {total_files_processed}")
    print(f"Total number of batches processed: {batch_index}")
