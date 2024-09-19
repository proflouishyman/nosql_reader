# File: data_processing.py

import os
import json
from database_setup import (
    documents,  # Added import for documents
    insert_document,
    update_field_structure,
    unique_terms_collection
)
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
from pymongo import MongoClient
import re

def init_db():
    global documents
    client = MongoClient('mongodb://localhost:27017/')
    db = client['railroad_documents']
    documents = db['documents']

def load_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def process_file(file_path):
    filename = os.path.basename(file_path)
    json_data = load_json_file(file_path)
    if json_data:
        json_data.setdefault('filename', filename)
        update_field_structure(json_data)
        try:
            insert_document(json_data)
        except Exception as e:
            print(f"Insert error for {filename}: {e}")
            return None

        unique_terms = {}
        for field, value in json_data.items():
            if isinstance(value, str):
                words = re.findall(r'\w+', value.lower())
                unique_terms.setdefault(field, set()).update(words)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        words = re.findall(r'\w+', item.lower())
                        unique_terms.setdefault(field, set()).update(words)
        return unique_terms
    return None

def merge_unique_terms(main_dict, new_dict):
    for field, terms in new_dict.items():
        main_dict.setdefault(field, set()).update(terms)

def save_unique_terms(unique_terms_dict):
    unique_terms_documents = []
    for field, terms in unique_terms_dict.items():
        unique_terms_documents.append({
            "field": field,
            "terms": list(terms)
        })
    try:
        unique_terms_collection.delete_many({})
        if unique_terms_documents:
            unique_terms_collection.insert_many(unique_terms_documents)
        print("Unique terms updated.")
    except Exception as e:
        print(f"Error saving unique terms: {e}")

def process_directory(directory_path):
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

    with Pool(cpu_count(), initializer=init_db) as pool:
        for result in tqdm(pool.imap(process_file, files), total=total):
            if result:
                merge_unique_terms(final_unique, result)

    save_unique_terms(final_unique)
    duration = time.time() - start_time
    print(f"Processed {total} files in {duration:.2f} seconds.")

if __name__ == "__main__":
    data_directory = r'G:\My Drive\2024-2025\coding\rolls_txt\scratch4\lhyman6\OCR\data\borr\rolls'
    process_directory(data_directory)
