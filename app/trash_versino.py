# data_processing.py

import os
import json
import re
import hashlib
from database_setup import (
    insert_document,
    update_field_structure,
    get_db,
    is_file_ingested,
    get_client,
)
from dotenv import load_dotenv
from pymongo import UpdateOne

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
import logging
import argparse
from collections import Counter
import pymongo

# =======================
# Logging Configuration
# =======================
logger = logging.getLogger('DataProcessingLogger')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    file_handler = logging.FileHandler('database_processing.log', mode='a')
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# =======================
# Global Variables
# =======================

root_directory = None
db = None  # Will be initialized in each process

# =======================
# Utility Functions
# =======================

def calculate_file_hash(file_path):
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        return None

def clean_json(json_text):
    """Remove control characters and extract valid JSON content."""
    # Remove control characters
    json_text = re.sub(r'[\x00-\x1F\x7F]', '', json_text)
    # Attempt to find the first and last curly braces
    start_index = json_text.find('{')
    end_index = json_text.rfind('}')
    if start_index != -1 and end_index != -1 and end_index > start_index:
        json_substring = json_text[start_index:end_index + 1]
        try:
            json.loads(json_substring)
            return json_substring
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            try:
                fixed_json = json_substring.encode('utf-8').decode('unicode_escape', 'ignore')
                json.loads(fixed_json)
                return fixed_json
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format after cleaning.")
    raise ValueError("Invalid JSON format: Unable to find valid JSON object.")

def load_and_validate_json_file(file_path):
    """Load a JSON file, validate its content, and return it as a dictionary."""
    filename = os.path.basename(file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            json_content = file.read()
        cleaned_json_content = clean_json(json_content)
        json_data = json.loads(cleaned_json_content)
        json_data.setdefault('filename', filename)
        json_data.setdefault('relative_path', os.path.relpath(file_path, start=root_directory))
        json_data['file_path'] = file_path
        json_data['file_hash'] = calculate_file_hash(file_path)
        return json_data, None
    except json.JSONDecodeError as e:
        error_msg = f"Error decoding JSON in {filename}: {str(e)}"
        return None, error_msg
    except Exception as e:
        error_msg = f"Error processing file {filename}: {str(e)}"
        return None, error_msg

# =======================
# Combined Hash Functions
# =======================

def calculate_combined_hash(file_list):
    """Calculate a combined SHA256 hash for all files in the list."""
    sha256_hash = hashlib.sha256()
    for file_path in sorted(file_list):
        file_hash = calculate_file_hash(file_path)
        if file_hash:
            sha256_hash.update(file_hash.encode())
    return sha256_hash.hexdigest()

def get_stored_combined_hash(db):
    """Retrieve the stored combined hash from the metadata collection."""
    metadata = db.metadata.find_one({"type": "combined_hash"})
    return metadata["hash"] if metadata and "hash" in metadata else None

def store_combined_hash(db, combined_hash):
    """Store the combined hash in the metadata collection."""
    db.metadata.update_one(
        {"type": "combined_hash"},
        {"$set": {"hash": combined_hash, "last_updated": time.time()}},
        upsert=True
    )

# =======================
# Processing Functions
# =======================

def init_db():
    """Initialize a new MongoDB connection for each process."""
    global db
    try:
        client = get_client()
        db = get_db(client)
        logger.debug("Database connection initialized.")
    except Exception as e:
        logger.exception("Failed to initialize database connection")
        raise e

def process_file(file_path):
    """
    Process a single file and return the result.
    """
    # Initialize database connection for this process
    if db is None:
        init_db()
    filename = os.path.basename(file_path)
    logger.debug(f"Processing file: {filename}")
    result = {'processed': [], 'failed': [], 'skipped': []}

    try:
        file_hash = calculate_file_hash(file_path)
        if is_file_ingested(db, file_hash):
            logger.debug(f"File already ingested: {filename}")
            result['skipped'].append(file_path)
            return result, None

        json_data, error = load_and_validate_json_file(file_path)
        if json_data:
            try:
                update_field_structure(db, json_data)  # Pass 'db' here
                insert_document(db, json_data)         # Pass 'db' here
                logger.debug(f"Processed and inserted document: {filename}")
                result['processed'].append(file_path)
            except Exception as e:
                logger.exception(f"Error processing {filename}")
                result['failed'].append((file_path, str(e)))
        else:
            if error:
                logger.error(error)
                result['failed'].append((file_path, error))

    except Exception as e:
        logger.exception(f"Unexpected error processing {filename}")
        result['failed'].append((file_path, str(e)))

    return result, None

def get_all_files(directory):
    """Recursively get all JSON and TXT files in the given directory and its subdirectories."""
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.json', '.txt')):
                file_path = os.path.join(root, file)
                file_list.append(file_path)
                logger.debug(f"Found file: {file_path}")
    return file_list

# =======================
# Combined Hash Processing
# =======================

def process_directory_with_combined_hash(directory_path):
    """Process all files in a directory using the combined hash optimization."""
    global root_directory
    root_directory = directory_path

    start_time = time.time()
    files = get_all_files(directory_path)
    total = len(files)

    logger.info(f"Found {total} files to process.")

    if total == 0:
        logger.warning("No files found to process. Exiting.")
        return

    # Calculate the combined hash of all files
    logger.info("Calculating combined hash of all files...")
    combined_hash = calculate_combined_hash(files)
    logger.debug(f"Combined Hash: {combined_hash}")

    # Initialize database connection
    init_db()

    # Retrieve the stored combined hash
    stored_hash = get_stored_combined_hash(db)
    logger.debug(f"Stored Combined Hash: {stored_hash}")

    if combined_hash == stored_hash:
        logger.info("No changes detected. Skipping processing.")
        print("No changes detected. Skipping processing.")
        return
    else:
        logger.info("Changes detected. Proceeding with processing.")
        print("Changes detected. Proceeding with processing.")

    # Initialize results_dict in the main process
    results_dict = {
        'processed': [],
        'failed': [],
        'skipped': []
    }

    # Setup multiprocessing pool
    num_workers = min(cpu_count(), 8)  # Limit to 8 workers to prevent excessive load
    with Pool(processes=num_workers) as pool:
        # Use tqdm for progress bar
        for result, _ in tqdm(pool.imap_unordered(process_file, files), total=total, desc="Processing files"):
            for key in ['processed', 'failed', 'skipped']:
                results_dict[key].extend(result.get(key, []))

    logger.info("\nProcessing Summary:")
    logger.info(f"Total files found: {total}")
    logger.info(f"Successfully processed: {len(results_dict['processed'])}")
    logger.info(f"Skipped (already ingested): {len(results_dict['skipped'])}")
    logger.info(f"Failed to process: {len(results_dict['failed'])}")

    if results_dict['failed']:
        logger.info("\nFailed files:")
        for file_path, error in results_dict['failed']:
            logger.error(f"- {file_path}: {error}")

    # Update the stored combined hash after successful processing
    store_combined_hash(db, combined_hash)
    logger.info("Updated the stored combined hash.")

    duration = time.time() - start_time
    logger.info(f"\nTotal processing time: {duration:.2f} seconds.")

# =======================
# Main Execution
# =======================
if __name__ == "__main__":
    print("Starting data_processing.py")
    logger.info("Starting data_processing.py")
    parser = argparse.ArgumentParser(description="Process and validate JSON and TXT files for the railroad documents database.")
    parser.add_argument("data_directory", nargs='?', default='/app/archives',
                        help="Path to the root directory containing JSON and/or text files to process (default: '/app/archives')")
    args = parser.parse_args()
    
    data_directory = args.data_directory
    print(f"Data Directory: {data_directory}")
    if not os.path.exists(data_directory):
        logger.error(f"Error: The specified directory does not exist: {data_directory}")
        logger.info(f"Creating directory: {data_directory}")
        try:
            os.makedirs(data_directory)
            logger.info(f"Directory created successfully: {data_directory}")
        except Exception as e:
            logger.exception("Failed to create directory")
            exit(1)

    logger.info(f"Processing directory: {data_directory}")
    print("Don't Forget To Turn On Your Fan!")

    # Start processing with combined hash optimization
    process_directory_with_combined_hash(data_directory)
