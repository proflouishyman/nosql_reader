"""
data_processing.py - Self-contained document ingestion script

Process and validate JSON and TXT files for the railroad documents database.
Handles file ingestion with hash checking to avoid duplicate processing.

Usage:
    docker compose exec app python app/data_processing.py /data/archives/borr_data
"""

import os
import json
import re
import hashlib
import time
import logging
import argparse
from collections import Counter
from pymongo import MongoClient, ASCENDING, DESCENDING, UpdateOne
from bson import ObjectId
from tqdm import tqdm

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
# Database Functions
# =======================

def get_client():
    """Initialize and return a new MongoDB client."""
    try:
        # Try APP_MONGO_URI first (used in docker-compose), fall back to MONGO_URI
        mongo_uri = os.environ.get('APP_MONGO_URI') or os.environ.get('MONGO_URI')
        
        if not mongo_uri:
            raise ValueError("Neither APP_MONGO_URI nor MONGO_URI environment variable is set")
        
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        # Test connection
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB.")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise e

def get_db(client):
    """Return the database instance."""
    return client['railroad_documents']

def init_db():
    """Initialize a new MongoDB connection for the process."""
    global db
    try:
        client = get_client()
        db = get_db(client)
        logger.debug("Database connection initialized.")
    except Exception as e:
        logger.exception("Failed to initialize database connection")
        raise e

def insert_document(db, document):
    """Insert a document into the 'documents' collection."""
    try:
        documents = db['documents']
        documents.insert_one(document)
        logger.debug(f"Inserted document: {document.get('filename', 'unknown')}")
    except Exception as e:
        logger.error(f"Error inserting document: {e}")
        raise e

def discover_fields(document):
    """
    Recursively discover fields in a document.
    :param document: The document to analyze
    :return: A dictionary representing the field structure
    """
    structure = {}
    for key, value in document.items():
        if isinstance(value, dict):
            structure[key] = discover_fields(value)
        elif isinstance(value, list):
            if value:
                if isinstance(value[0], dict):
                    structure[key] = [discover_fields(value[0])]
                else:
                    structure[key] = [type(value[0]).__name__]
            else:
                structure[key] = []
        else:
            structure[key] = type(value).__name__
    return structure

def merge_structures(existing, new):
    """
    Merge two field structures.
    :param existing: The existing field structure
    :param new: The new field structure to merge
    :return: The merged field structure
    """
    for key, value in new.items():
        if key not in existing:
            existing[key] = value
        elif isinstance(value, dict) and isinstance(existing[key], dict):
            merge_structures(existing[key], value)
        elif isinstance(value, list) and isinstance(existing[key], list):
            if value and existing[key]:
                if isinstance(value[0], dict) and isinstance(existing[key][0], dict):
                    merge_structures(existing[key][0], value[0])
    return existing

def update_field_structure(db, document):
    """
    Update the field structure based on a new document.
    :param db: Database instance
    :param document: The new document to analyze
    """
    field_structure_collection = db['field_structure']
    new_structure = discover_fields(document)
    merged_structure = {}

    # Attempt to retrieve the existing structure
    existing_structure = field_structure_collection.find_one({"_id": "current_structure"})

    if existing_structure:
        # Merge the new structure with the existing one
        merged_structure = merge_structures(existing_structure['structure'], new_structure)
    else:
        # If no existing structure, use the new structure
        merged_structure = new_structure

    # Perform an upsert operation to update or insert the structure
    field_structure_collection.update_one(
        {"_id": "current_structure"},
        {"$set": {"structure": merged_structure}},
        upsert=True
    )

def is_file_ingested(db, file_hash):
    """Check if a file has already been ingested based on its hash."""
    if not file_hash:
        return False
    try:
        documents = db['documents']
        ingested = documents.find_one({'file_hash': file_hash}) is not None
        logger.debug(f"File ingestion check for hash {file_hash}: {ingested}")
        return ingested
    except Exception as e:
        logger.error(f"Error checking ingestion status for hash {file_hash}: {e}")
        return False

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

def total_hash_check(directory_path):
    """Check if the archive directory has changed since last processing."""
    files = get_all_files(directory_path)
    total = len(files)

    logger.info(f"Found {total} files to process.")
    print(f"Found {total} files to process.")

    if total == 0:
        logger.warning("No files found to process. Exiting.")
        print("No .json or .txt files found to process.")
        return False

    # Calculate the combined hash of all files
    logger.info("Calculating combined hash of all files...")
    print("Calculating combined hash of all files...")
    combined_hash = calculate_combined_hash(files)
    logger.debug(f"Combined Hash: {combined_hash}")

    # Initialize database connection
    if db is None:
        init_db()

    # Retrieve the stored combined hash
    stored_hash = get_stored_combined_hash(db)
    logger.debug(f"Stored Combined Hash: {stored_hash}")

    if combined_hash == stored_hash:
        logger.info("No changes detected. Skipping processing.")
        print("No changes detected. Skipping processing.")
        return False
    else:
        logger.info("Changes detected. Proceeding with processing.")
        print("Changes detected. Proceeding with processing.")
        return True

# =======================
# Processing Functions
# =======================

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
                update_field_structure(db, json_data)
                insert_document(db, json_data)
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
                file_list.append(os.path.join(root, file))
                logger.debug(f"Found file: {os.path.join(root, file)}")
    return file_list

# =======================
# Sequential Processing
# =======================
def process_directory(directory_path):
    """Process all files in a directory and its subdirectories sequentially."""
    global root_directory
    root_directory = directory_path

    start_time = time.time()
    files = get_all_files(directory_path)
    total = len(files)

    logger.info(f"Found {total} files to process.")

    if total == 0:
        logger.warning("No files found to process. Exiting.")
        print("No .json or .txt files found.")
        return

    # Initialize results_dict in the main process
    results_dict = {
        'processed': [],
        'failed': [],
        'skipped': []
    }

    print(f"\nProcessing {total} files...")
    with tqdm(total=total, desc="Processing files") as pbar:
        for idx, file_path in enumerate(files, 1):
            result, _ = process_file(file_path)
            for key in ['processed', 'failed', 'skipped']:
                results_dict[key].extend(result.get(key, []))
            pbar.update(1)

    # Initialize database connection (if not already)
    if db is None:
        init_db()
        logger.debug("Initialized main process database connection.")

    print("\n" + "="*60)
    print("Processing Summary:")
    print("="*60)
    print(f"Total files found: {total}")
    print(f"Successfully processed: {len(results_dict['processed'])}")
    print(f"Skipped (already ingested): {len(results_dict['skipped'])}")
    print(f"Failed to process: {len(results_dict['failed'])}")

    logger.info("\nProcessing Summary:")
    logger.info(f"Total files found: {total}")
    logger.info(f"Successfully processed: {len(results_dict['processed'])}")
    logger.info(f"Skipped (already ingested): {len(results_dict['skipped'])}")
    logger.info(f"Failed to process: {len(results_dict['failed'])}")

    if results_dict['failed']:
        print("\nFailed files:")
        logger.info("\nFailed files:")
        for file_path, error in results_dict['failed']:
            print(f"- {file_path}: {error}")
            logger.error(f"- {file_path}: {error}")

    duration = time.time() - start_time
    print(f"\nTotal processing time: {duration:.2f} seconds.")
    logger.info(f"\nTotal processing time: {duration:.2f} seconds.")

# =======================
# Main Execution
# =======================
if __name__ == "__main__":
    print("="*60)
    print("Data Processing Script")
    print("="*60)
    logger.info("Starting data_processing.py")
    
    parser = argparse.ArgumentParser(
        description="Process and validate JSON and TXT files for the railroad documents database."
    )
    parser.add_argument(
        "data_directory", 
        nargs='?', 
        default='/data/archives',
        help="Path to the root directory containing JSON and/or text files to process (default: '/data/archives')"
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Force reprocessing even if no changes detected (clears hash)"
    )
    args = parser.parse_args()
    
    data_directory = args.data_directory
    print(f"Target directory: {data_directory}")
    
    if not os.path.exists(data_directory):
        logger.error(f"Error: The specified directory does not exist: {data_directory}")
        print(f"Error: Directory does not exist: {data_directory}")
        print(f"Creating directory: {data_directory}")
        try:
            os.makedirs(data_directory)
            logger.info(f"Directory created successfully: {data_directory}")
            print(f"Directory created successfully.")
        except Exception as e:
            logger.exception("Failed to create directory")
            print(f"Failed to create directory: {e}")
            exit(1)

    # Force option - clear hash
    if args.force:
        print("\n--force flag detected. Clearing stored hash...")
        logger.info("Force flag set. Clearing stored hash.")
        init_db()
        db.metadata.delete_one({'type': 'combined_hash'})
        print("Hash cleared. Will reprocess all files.\n")

    # Hash check for archive file change
    logger.info("Checking to see if archive has changed")
    print("\nChecking for changes in archive...")
    archives_file_change = total_hash_check(data_directory)

    if archives_file_change:
        logger.info(f"Processing directory: {data_directory}")
        process_directory(data_directory)

        # After processing, calculate and store the new hash
        new_combined_hash = calculate_combined_hash(get_all_files(data_directory))
        store_combined_hash(db, new_combined_hash)
        logger.info("Updated stored hash after processing.")
        print("\nHash updated. Future runs will skip unchanged files.")
    else:
        logger.info("No change in archives")
        print("\nNo processing needed.")
        print("Use --force flag to reprocess all files.")