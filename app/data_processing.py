# data_processing.py

import os
import json
import re
import hashlib
from database_setup import insert_document, update_field_structure, get_db
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
import logging
import argparse
from collections import Counter

# =======================
# Logging Configuration
# =======================

# Create a custom logger
logger = logging.getLogger('DataProcessingLogger')
logger.setLevel(logging.INFO)  # Adjust logging level as needed

# Create handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler('data_processing.log')
file_handler.setLevel(logging.ERROR)

# Create formatters and add them to handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
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

def is_file_ingested(db, file_path, file_hash):
    """Check if a file has already been ingested based on its path and hash."""
    if not file_hash:
        return False
    try:
        documents = db['documents']
        return documents.find_one({
            'file_path': file_path,
            'file_hash': file_hash
        }) is not None
    except Exception as e:
        logger.error(f"Error checking ingestion status for {file_path}: {e}")
        return False

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

def collect_unique_terms(json_data):
    """Collect unique words and phrases from the JSON data."""
    unique_terms = {}
    for field, value in json_data.items():
        if isinstance(value, str):
            words = re.findall(r'\w+', value.lower())
            phrases = [' '.join(pair) for pair in zip(words, words[1:])]
            unique_terms.setdefault(field, {'words': Counter(), 'phrases': Counter()})
            unique_terms[field]['words'].update(words)
            unique_terms[field]['phrases'].update(phrases)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    words = re.findall(r'\w+', item.lower())
                    phrases = [' '.join(pair) for pair in zip(words, words[1:])]
                    unique_terms.setdefault(field, {'words': Counter(), 'phrases': Counter()})
                    unique_terms[field]['words'].update(words)
                    unique_terms[field]['phrases'].update(phrases)
    return unique_terms

def merge_unique_terms(main_dict, new_dict):
    """Merge two unique terms dictionaries."""
    for field, terms in new_dict.items():
        if field not in main_dict:
            main_dict[field] = {'words': Counter(), 'phrases': Counter()}
        if not isinstance(terms.get('words'), Counter) or not isinstance(terms.get('phrases'), Counter):
            logger.error(f"Expected 'words' and 'phrases' to be Counters in field '{field}', got {type(terms.get('words'))} and {type(terms.get('phrases'))}.")
            continue
        main_dict[field]['words'].update(terms['words'])
        main_dict[field]['phrases'].update(terms['phrases'])

def save_unique_terms(db, unique_terms_dict):
    """Save the unique terms to the database in a flattened structure."""
    unique_terms_collection = db['unique_terms']
    unique_terms_documents = []
    for field, terms in unique_terms_dict.items():
        for word, count in terms['words'].items():
            if count >= 2:
                unique_terms_documents.append({
                    "term": word,
                    "field": field,
                    "count": count,
                    "type": "word"
                })
        for phrase, count in terms['phrases'].items():
            if count >= 2:
                unique_terms_documents.append({
                    "term": phrase,
                    "field": field,
                    "count": count,
                    "type": "phrase"
                })
    try:
        unique_terms_collection.delete_many({})
        if unique_terms_documents:
            unique_terms_collection.insert_many(unique_terms_documents)
            unique_terms_collection.create_index([("term", 1)])
            unique_terms_collection.create_index([("field", 1)])
            unique_terms_collection.create_index([("type", 1)])
        logger.info("Unique terms updated in the database.")
    except Exception as e:
        logger.error(f"Error saving unique terms: {e}")

# Optionally, save unique terms to a file for faster loading
import pickle

def save_unique_terms_to_file(unique_terms_dict, filename='unique_terms.pkl'):
    """Serialize and save unique terms to a file."""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(unique_terms_dict, f)
        logger.info(f"Unique terms saved to file: {filename}")
    except Exception as e:
        logger.error(f"Error saving unique terms to file: {e}")

# =======================
# Processing Functions
# =======================

def init_db():
    """Initialize a new MongoDB connection for each process."""
    global db
    try:
        db = get_db()
    except Exception as e:
        logger.error(f"Failed to initialize database connection: {e}")
        raise e

def process_file(file_path):
    """Process a single file and return the result."""
    # Initialize database connection for this process
    if db is None:
        init_db()
    filename = os.path.basename(file_path)
    logger.debug(f"Processing file: {filename}")
    result = {'processed': [], 'failed': [], 'skipped': []}
    unique_terms = None

    try:
        file_hash = calculate_file_hash(file_path)
        if is_file_ingested(db, file_path, file_hash):
            logger.debug(f"File already ingested: {filename}")
            result['skipped'].append(file_path)
            return result, unique_terms

        json_data, error = load_and_validate_json_file(file_path)
        if json_data:
            try:
                update_field_structure(db, json_data)
                insert_document(db, json_data)
                logger.debug(f"Processed and inserted document: {filename}")
                result['processed'].append(file_path)
                unique_terms = collect_unique_terms(json_data)
            except Exception as e:
                error = f"Error processing {filename}: {str(e)}"
                logger.error(error)
                result['failed'].append((file_path, error))
        else:
            if error:
                logger.error(error)
                result['failed'].append((file_path, error))

    except Exception as e:
        error = f"Unexpected error processing {filename}: {str(e)}"
        logger.error(error)
        result['failed'].append((file_path, error))

    return result, unique_terms

def get_all_files(directory):
    """Recursively get all JSON and TXT files in the given directory and its subdirectories."""
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.json', '.txt')):
                file_list.append(os.path.join(root, file))
    return file_list

def process_directory(directory_path):
    """Process all files in a directory and its subdirectories using multiprocessing."""
    global root_directory
    root_directory = directory_path

    start_time = time.time()
    files = get_all_files(directory_path)
    total = len(files)
    final_unique = {}

    logger.info(f"Found {total} files to process.")

    batch_size = 1000  # Adjust based on system capabilities
    num_batches = (total // batch_size) + (1 if total % batch_size != 0 else 0)

    # Initialize results_dict in the main process
    results_dict = {
        'processed': [],
        'failed': [],
        'skipped': []
    }

    with tqdm(total=total, desc="Processing files") as pbar:
        for batch_num in range(num_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total)
            batch_files = files[start_idx:end_idx]
            logger.info(f"Processing batch {batch_num + 1}/{num_batches} with {len(batch_files)} files.")

            with Pool(processes=min(cpu_count(), 8), initializer=init_db) as pool:
                for res in pool.imap_unordered(process_file, batch_files):
                    if not isinstance(res, tuple) or len(res) != 2:
                        logger.error(f"Unexpected result format: {res}")
                        continue
                    result, unique_terms = res
                    for key in ['processed', 'failed', 'skipped']:
                        if not isinstance(result[key], list):
                            logger.error(f"Expected result[{key}] to be a list, got {type(result[key])} instead.")
                            continue
                        results_dict[key].extend(result[key])
                    if unique_terms:
                        merge_unique_terms(final_unique, unique_terms)
                    pbar.update(1)

    # Initialize database connection to save unique terms
    if db is None:
        init_db()
    save_unique_terms(db, final_unique)
    save_unique_terms_to_file(final_unique)

    logger.info("\nProcessing Summary:")
    logger.info(f"Total files found: {total}")
    logger.info(f"Successfully processed: {len(results_dict['processed'])}")
    logger.info(f"Skipped (already ingested): {len(results_dict['skipped'])}")
    logger.info(f"Failed to process: {len(results_dict['failed'])}")

    if results_dict['failed']:
        logger.info("\nFailed files:")
        for file_path, error in results_dict['failed']:
            logger.error(f"- {file_path}: {error}")

    duration = time.time() - start_time
    logger.info(f"\nTotal processing time: {duration:.2f} seconds.")

# =======================
# Main Execution
# =======================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and validate JSON and TXT files for the railroad documents database.")
    parser.add_argument("data_directory", nargs='?', default='archives',
                        help="Path to the root directory containing JSON and/or text files to process (default: './archives')")
    args = parser.parse_args()

    # Get the absolute path of the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the data directory path
    data_directory = os.path.abspath(os.path.join(script_dir, args.data_directory))

    if not os.path.exists(data_directory):
        logger.error(f"Error: The specified directory does not exist: {data_directory}")
        logger.info(f"Creating directory: {data_directory}")
        try:
            os.makedirs(data_directory)
            logger.info(f"Directory created successfully: {data_directory}")
        except Exception as e:
            logger.error(f"Failed to create directory: {e}")
            exit(1)

    logger.info(f"Processing directory: {data_directory}")
    print("Don't Forget To Turn On Your Fan!")
    process_directory(data_directory)
