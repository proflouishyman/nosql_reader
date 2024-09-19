
import os
import json
from database_setup import (
    insert_document,
    update_field_structure,
    unique_terms_collection,
    documents
)
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
from pymongo import MongoClient
import re
import logging

# Function to initialize a new MongoDB connection for each process
def init_db():
    global documents
    client = MongoClient('mongodb://localhost:27017/')
    db = client['railroad_documents']
    documents = db['documents']

def load_json_file(file_path):
    """
    Load a JSON file and return its content as a dictionary.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {file_path}: {str(e)}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None

def process_file(file_path):
    """
    Process a single JSON file: load it, add filename, update field structure,
    and collect unique words and phrases.
    """
    filename = os.path.basename(file_path)
    logging.debug(f"Processing file: {filename}")
    json_data = load_json_file(file_path)
    if json_data:
        # Add filename as a field if it's not already present
        json_data.setdefault('filename', filename)
        try:
            update_field_structure(json_data)  # Update field structure for each new document
            logging.debug(f"Updated field structure for {filename}")
        except Exception as e:
            logging.error(f"Error updating field structure for {filename}: {e}")
            return None

        # Insert document into the database
        try:
            insert_document(json_data)
            logging.debug(f"Inserted document: {filename}")
        except Exception as e:
            logging.error(f"Error inserting document {filename}: {e}")
            return None

        # Collect unique words and phrases by tokenizing
        unique_terms = {}
        for field, value in json_data.items():
            if isinstance(value, str):
                words = re.findall(r'\w+', value.lower())
                phrases = [' '.join(pair) for pair in zip(words, words[1:])]
                if words:
                    unique_terms.setdefault(field, {'words': {}, 'phrases': {}})
                    # Count words
                    for word in words:
                        unique_terms[field]['words'][word] = unique_terms[field]['words'].get(word, 0) + 1
                    # Count phrases
                    for phrase in phrases:
                        unique_terms[field]['phrases'][phrase] = unique_terms[field]['phrases'].get(phrase, 0) + 1
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        words = re.findall(r'\w+', item.lower())
                        phrases = [' '.join(pair) for pair in zip(words, words[1:])]
                        if words:
                            unique_terms.setdefault(field, {'words': {}, 'phrases': {}})
                            # Count words
                            for word in words:
                                unique_terms[field]['words'][word] = unique_terms[field]['words'].get(word, 0) + 1
                            # Count phrases
                            for phrase in phrases:
                                unique_terms[field]['phrases'][phrase] = unique_terms[field]['phrases'].get(phrase, 0) + 1
            # Add more conditions if fields can have other types

        # Filter out words and phrases with count < 2
        for field in unique_terms:
            # Filter words
            original_word_count = len(unique_terms[field]['words'])
            unique_terms[field]['words'] = {word: count for word, count in unique_terms[field]['words'].items() if count >= 2}
            filtered_word_count = len(unique_terms[field]['words'])
            logging.debug(f"Filtered words in '{field}': {original_word_count} -> {filtered_word_count}")

            # Filter phrases
            original_phrase_count = len(unique_terms[field]['phrases'])
            unique_terms[field]['phrases'] = {phrase: count for phrase, count in unique_terms[field]['phrases'].items() if count >= 2}
            filtered_phrase_count = len(unique_terms[field]['phrases'])
            logging.debug(f"Filtered phrases in '{field}': {original_phrase_count} -> {filtered_phrase_count}")

        logging.debug(f"Collected unique terms for {filename}")
        return unique_terms

def merge_unique_terms(main_dict, new_dict):
    """
    Merge two unique terms dictionaries.
    Only includes words and phrases with counts >= 2.
    """
    for field, terms in new_dict.items():
        if field not in main_dict:
            main_dict[field] = {'words': {}, 'phrases': {}}
        # Merge words
        for word, count in terms['words'].items():
            main_dict[field]['words'][word] = main_dict[field]['words'].get(word, 0) + count
        # Merge phrases
        for phrase, count in terms['phrases'].items():
            main_dict[field]['phrases'][phrase] = main_dict[field]['phrases'].get(phrase, 0) + count


def save_unique_terms(unique_terms_dict):
    """
    Save the unique terms dictionary to the database.
    Only includes words and phrases with counts >= 2.
    """
    unique_terms_documents = []
    for field, terms in unique_terms_dict.items():
        # Ensure that words and phrases are not empty after filtering
        if terms['words'] or terms['phrases']:
            unique_terms_documents.append({
                "field": field,
                "words": terms['words'],
                "phrases": terms['phrases']
            })
    try:
        unique_terms_collection.delete_many({})
        if unique_terms_documents:
            unique_terms_collection.insert_many(unique_terms_documents)
        logging.debug("Unique terms updated in the database.")
    except Exception as e:
        logging.error(f"Error saving unique terms: {e}")


def process_directory(directory_path):
    """
    Process all JSON files in a directory using multiprocessing:
    delete existing documents, insert new ones, update field structure,
    and compute unique terms.
    """
    # Clear existing data
    print("Clearing existing data...")
    try:
        documents.delete_many({})
        unique_terms_collection.delete_many({})
        from database_setup import field_structure
        field_structure.delete_many({})
        print("Cleared data.")
    except Exception as e:
        print(f"Error clearing data: {e}")

    start_time = time.time()
    files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.json')]
    total = len(files)
    final_unique = {}

    with Pool(processes=cpu_count(), initializer=init_db) as pool:
        for result in tqdm(pool.imap(process_file, files), total=total, desc="Processing files"):
            if result:
                merge_unique_terms(final_unique, result)

    # Save unique terms
    save_unique_terms(final_unique)
    duration = time.time() - start_time
    print(f"Processed {total} files in {duration:.2f} seconds.")

if __name__ == "__main__":
    data_directory = r'G:\My Drive\2024-2025\coding\rolls_txt\scratch4\lhyman6\OCR\data\borr\rolls'  # Update as needed
    process_directory(data_directory)