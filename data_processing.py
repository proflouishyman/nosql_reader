import os
import json
import re
import hashlib
from database_setup import (
    insert_document,
    update_field_structure,
    unique_terms_collection,
    documents,
    field_structure
)
from multiprocessing import Pool, cpu_count, Manager
from tqdm import tqdm
import time
from pymongo import MongoClient
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables
root_directory = None

def init_db():
    """Initialize a new MongoDB connection for each process."""
    global documents
    client = MongoClient('mongodb://admin:secret@localhost:27017', serverSelectionTimeoutMS=1000)
    db = client['railroad_documents']
    documents = db['documents']

def calculate_file_hash(file_path):
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def is_file_ingested(file_path, file_hash):
    """Check if a file has already been ingested based on its path and hash."""
    return documents.find_one({
        'file_path': file_path,
        'file_hash': file_hash
    }) is not None

def clean_json(json_text):
    """Remove control characters and extract valid JSON content."""
    json_text = re.sub(r'[\x00-\x1F\x7F]', '', json_text)
    start_index = json_text.find('{')
    end_index = json_text.rfind('}')
    if start_index != -1 and end_index != -1:
        return json_text[start_index:end_index + 1]
    raise ValueError("Invalid JSON format: Unable to find '{' or '}'.")

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
        return None, f"Error decoding JSON in {filename}: {str(e)}"
    except Exception as e:
        return None, f"Error processing file {filename}: {str(e)}"

def collect_unique_terms(json_data):
    """Collect unique words and phrases from the JSON data."""
    unique_terms = {}
    for field, value in json_data.items():
        if isinstance(value, str):
            words = re.findall(r'\w+', value.lower())
            phrases = [' '.join(pair) for pair in zip(words, words[1:])]
            unique_terms.setdefault(field, {'words': {}, 'phrases': {}})
            for word in words:
                unique_terms[field]['words'][word] = unique_terms[field]['words'].get(word, 0) + 1
            for phrase in phrases:
                unique_terms[field]['phrases'][phrase] = unique_terms[field]['phrases'].get(phrase, 0) + 1
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    words = re.findall(r'\w+', item.lower())
                    phrases = [' '.join(pair) for pair in zip(words, words[1:])]
                    unique_terms.setdefault(field, {'words': {}, 'phrases': {}})
                    for word in words:
                        unique_terms[field]['words'][word] = unique_terms[field]['words'].get(word, 0) + 1
                    for phrase in phrases:
                        unique_terms[field]['phrases'][phrase] = unique_terms[field]['phrases'].get(phrase, 0) + 1
    return unique_terms

def merge_unique_terms(main_dict, new_dict):
    """Merge two unique terms dictionaries."""
    for field, terms in new_dict.items():
        if field not in main_dict:
            main_dict[field] = {'words': {}, 'phrases': {}}
        for word, count in terms['words'].items():
            main_dict[field]['words'][word] = main_dict[field]['words'].get(word, 0) + count
        for phrase, count in terms['phrases'].items():
            main_dict[field]['phrases'][phrase] = main_dict[field]['phrases'].get(phrase, 0) + count

def save_unique_terms(unique_terms_dict):
    """Save the unique terms dictionary to the database."""
    unique_terms_documents = []
    for field, terms in unique_terms_dict.items():
        if terms['words'] or terms['phrases']:
            unique_terms_documents.append({
                "field": field,
                "words": {word: count for word, count in terms['words'].items() if count >= 2},
                "phrases": {phrase: count for phrase, count in terms['phrases'].items() if count >= 2}
            })
    try:
        unique_terms_collection.delete_many({})
        if unique_terms_documents:
            unique_terms_collection.insert_many(unique_terms_documents)
        logging.info("Unique terms updated in the database.")
    except Exception as e:
        logging.error(f"Error saving unique terms: {e}")

def process_file(args):
    """Process a single file: load it, validate JSON, update field structure, and collect unique words and phrases."""
    file_path, results_dict = args
    filename = os.path.basename(file_path)
    logging.debug(f"Processing file: {filename}")
    
    file_hash = calculate_file_hash(file_path)
    if is_file_ingested(file_path, file_hash):
        logging.debug(f"File already ingested: {filename}")
        results_dict['skipped'].append(file_path)
        return None
    
    json_data, error = load_and_validate_json_file(file_path)
    if json_data:
        try:
            update_field_structure(json_data)
            insert_document(json_data)
            logging.debug(f"Processed and inserted document: {filename}")
            results_dict['processed'].append(file_path)
            return collect_unique_terms(json_data)
        except Exception as e:
            error = f"Error processing {filename}: {str(e)}"
    
    if error:
        logging.error(error)
        results_dict['failed'].append((file_path, error))
    
    return None

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

    logging.info(f"Found {total} files to process.")

    with Manager() as manager:
        results_dict = manager.dict()
        results_dict['processed'] = manager.list()
        results_dict['failed'] = manager.list()
        results_dict['skipped'] = manager.list()

        with Pool(processes=cpu_count(), initializer=init_db) as pool:
            for result in tqdm(pool.imap(process_file, [(f, results_dict) for f in files]), total=total, desc="Processing files"):
                if result:
                    merge_unique_terms(final_unique, result)

    save_unique_terms(final_unique)
    
    logging.info("\nProcessing Summary:")
    logging.info(f"Total files found: {total}")
    logging.info(f"Successfully processed: {len(results_dict['processed'])}")
    logging.info(f"Skipped (already ingested): {len(results_dict['skipped'])}")
    logging.info(f"Failed to process: {len(results_dict['failed'])}")
    
    if results_dict['failed']:
        logging.info("\nFailed files:")
        for file_path, error in results_dict['failed']:
            logging.error(f"- {file_path}: {error}")

    duration = time.time() - start_time
    logging.info(f"\nTotal processing time: {duration:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and validate JSON files for the railroad documents database.")
    parser.add_argument("data_directory", nargs='?', default='data',
                        help="Path to the root directory containing JSON and/or text files to process (default: './data')")
    args = parser.parse_args()
    
    # Get the absolute path of the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the data directory path
    data_directory = os.path.abspath(os.path.join(script_dir, args.data_directory))
    
    if not os.path.exists(data_directory):
        logging.error(f"Error: The specified directory does not exist: {data_directory}")
        logging.info(f"Creating directory: {data_directory}")
        try:
            os.makedirs(data_directory)
            logging.info(f"Directory created successfully: {data_directory}")
        except Exception as e:
            logging.error(f"Failed to create directory: {e}")
            exit(1)
    
    logging.info(f"Processing directory: {data_directory}")
    process_directory(data_directory)