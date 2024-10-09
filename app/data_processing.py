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
    save_unique_terms,
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

def collect_unique_terms(json_data):
    """Collect unique words and phrases from the JSON data categorized by field and type."""
    unique_terms = {}
    for field, value in json_data.items():
        if isinstance(value, str):
            words = re.findall(r'\w+', value.lower())
            phrases = [' '.join(pair) for pair in zip(words, words[1:])]
            unique_terms.setdefault(field, {'word': Counter(), 'phrase': Counter()})
            unique_terms[field]['word'].update(words)
            unique_terms[field]['phrase'].update(phrases)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    words = re.findall(r'\w+', item.lower())
                    phrases = [' '.join(pair) for pair in zip(words, words[1:])]
                    unique_terms.setdefault(field, {'word': Counter(), 'phrase': Counter()})
                    unique_terms[field]['word'].update(words)
                    unique_terms[field]['phrase'].update(phrases)
    return unique_terms


def merge_unique_terms(main_counter, new_counter):
    """Merge two unique terms dictionaries."""
    for field, types in new_counter.items():
        if not field:
            logger.warning("Encountered a null or empty field during merge. Skipping.") #there is a lot of redundant code to make sure no redundants slip through
            continue
        for term_type, counter in types.items():
            if not term_type:
                logger.warning("Encountered a null or empty type during merge. Skipping.")
                continue
            main_counter.setdefault(field, {'word': Counter(), 'phrase': Counter()})
            main_counter[field][term_type].update(counter)


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
    unique_terms = None

    try:
        file_hash = calculate_file_hash(file_path)
        if is_file_ingested(db, file_hash):
            logger.debug(f"File already ingested: {filename}")
            result['skipped'].append(file_path)
            return result, unique_terms

        json_data, error = load_and_validate_json_file(file_path)
        if json_data:
            try:
                update_field_structure(db, json_data)  # Pass 'db' here
                insert_document(db, json_data)         # Pass 'db' here
                logger.debug(f"Processed and inserted document: {filename}")
                result['processed'].append(file_path)
                unique_terms = collect_unique_terms(json_data)
                logger.debug(f"Collected unique terms for {filename}: {unique_terms}")
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

    return result, unique_terms



def get_all_files(directory):
    """Recursively get all JSON and TXT files in the given directory and its subdirectories."""
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.json', '.txt')):
                file_list.append(os.path.join(root, file))
    return file_list



# #multiprocessing version below
# def process_directory(directory_path):
#     """Process all files in a directory and its subdirectories using multiprocessing."""
#     global root_directory
#     root_directory = directory_path

#     start_time = time.time()
#     files = get_all_files(directory_path)
#     total = len(files)
#     final_unique = {}

#     logger.info(f"Found {total} files to process.")

#     if total == 0:
#         logger.warning("No files found to process. Exiting.")
#         return

#     batch_size = 1000  # Adjust based on system capabilities
#     num_batches = (total // batch_size) + (1 if total % batch_size != 0 else 0)

#     # Initialize results_dict in the main process
#     results_dict = {
#         'processed': [],
#         'failed': [],
#         'skipped': []
#     }

#     with tqdm(total=total, desc="Processing files") as pbar:
#         for batch_num in range(num_batches):
#             start_idx = batch_num * batch_size
#             end_idx = min(start_idx + batch_size, total)
#             batch_files = files[start_idx:end_idx]
#             logger.info(f"Processing batch {batch_num + 1}/{num_batches} with {len(batch_files)} files.")

#             with Pool(processes=min(cpu_count(), 8), initializer=init_db) as pool:
#                 for res in pool.imap_unordered(process_file, batch_files):
#                     if not isinstance(res, tuple) or len(res) != 2:
#                         logger.error(f"Unexpected result format: {res}")
#                         continue
#                     result, unique_terms = res
#                     for key in ['processed', 'failed', 'skipped']:
#                         if not isinstance(result[key], list):
#                             logger.error(f"Expected result[{key}] to be a list, got {type(result[key])} instead.")
#                             continue
#                         results_dict[key].extend(result[key])
#                     if unique_terms:
#                         merge_unique_terms(final_unique, unique_terms)
#                         logger.debug(f"Merged unique terms: {unique_terms}")
#                     pbar.update(1)

#     # Initialize database connection to save unique terms
#     if db is None:
#         init_db()
#         logger.debug("Initialized main process database connection.")

#     logger.debug(f"Final aggregated unique terms: {final_unique}")
#     save_unique_terms(db, final_unique)  # Pass 'db' here

#     logger.info("\nProcessing Summary:")
#     logger.info(f"Total files found: {total}")
#     logger.info(f"Successfully processed: {len(results_dict['processed'])}")
#     logger.info(f"Skipped (already ingested): {len(results_dict['skipped'])}")
#     logger.info(f"Failed to process: {len(results_dict['failed'])}")

#     if results_dict['failed']:
#         logger.info("\nFailed files:")
#         for file_path, error in results_dict['failed']:
#             logger.error(f"- {file_path}: {error}")

#     duration = time.time() - start_time
#     logger.info(f"\nTotal processing time: {duration:.2f} seconds.")


#sequential version below
def process_directory(directory_path):
    """Process all files in a directory and its subdirectories sequentially for debugging."""
    global root_directory
    root_directory = directory_path

    start_time = time.time()
    files = get_all_files(directory_path)
    total = len(files)
    final_unique = {}

    logger.info(f"Found {total} files to process.")

    if total == 0:
        logger.warning("No files found to process. Exiting.")
        return

    # Initialize results_dict in the main process
    results_dict = {
        'processed': [],
        'failed': [],
        'skipped': []
    }

    with tqdm(total=total, desc="Processing files") as pbar:
        for idx, file_path in enumerate(files, 1):
            result, unique_terms = process_file(file_path)
            for key in ['processed', 'failed', 'skipped']:
                results_dict[key].extend(result.get(key, []))
            if unique_terms:
                merge_unique_terms(final_unique, unique_terms)
                logger.debug(f"Merged unique terms: {unique_terms}")
            pbar.update(1)

    # Initialize database connection to save unique terms
    if db is None:
        init_db()
        logger.debug("Initialized main process database connection.")

    logger.debug(f"Final aggregated unique terms: {final_unique}")
    save_unique_terms(db, final_unique)  # Pass 'db' here

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
    print("Starting data_processing.py")
    logger.info("Starting data_processing.py")
    parser = argparse.ArgumentParser(description="Process and validate JSON and TXT files for the railroad documents database.")
    parser.add_argument("data_directory", nargs='?', default='/app/archives',
                        help="Path to the root directory containing JSON and/or text files to process (default: '/app/archives')")
    args = parser.parse_args()

    data_directory = args.data_directory

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
    process_directory(data_directory)
