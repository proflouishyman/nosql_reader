# File: data_processing.py
# Path: railroad_documents_project/data_processing.py

import os
import json
from database_setup import documents, insert_document, update_field_structure, unique_terms_collection
from tqdm import tqdm
import time

def load_json_file(file_path):
    """
    Load a JSON file and return its content as a dictionary.

    :param file_path: The path to the JSON file
    :return: The JSON content as a dictionary, or None if an error occurs
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

def process_file(file_path, unique_terms_dict):
    """
    Process a single JSON file: load it, add filename, update field structure, and collect unique terms.

    :param file_path: The path to the JSON file
    :param unique_terms_dict: Dictionary to collect unique terms per field
    :return: The loaded JSON data with filename added, or None if an error occurs
    """
    filename = os.path.basename(file_path)
    json_data = load_json_file(file_path)
    if json_data:
        # Add filename as a field if it's not already present
        if 'filename' not in json_data:
            json_data['filename'] = filename
        update_field_structure(json_data)  # Update field structure for each new document

        # Collect unique terms
        for field, value in json_data.items():
            if isinstance(value, str):
                term = value.strip().lower()  # Normalize the term
                if field not in unique_terms_dict:
                    unique_terms_dict[field] = {}
                unique_terms_dict[field][term] = unique_terms_dict[field].get(term, 0) + 1
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        term = item.strip().lower()
                        if field not in unique_terms_dict:
                            unique_terms_dict[field] = {}
                        unique_terms_dict[field][term] = unique_terms_dict[field].get(term, 0) + 1
            # Add more conditions if fields can have other types

        return json_data
    return None

def save_unique_terms(unique_terms_dict):
    """
    Save the unique terms dictionary to the database.

    :param unique_terms_dict: Dictionary containing unique terms per field with their counts
    """
    # Convert the dictionary into a list of documents for easier querying
    unique_terms_documents = []
    for field, terms in unique_terms_dict.items():
        unique_terms_documents.append({
            "field": field,
            "terms": [{"term": term, "count": count} for term, count in terms.items()]
        })
    
    # Clear existing unique terms
    unique_terms_collection.delete_many({})
    
    # Insert new unique terms
    if unique_terms_documents:
        unique_terms_collection.insert_many(unique_terms_documents)
    print("Unique terms have been updated in the database.")

def process_directory(directory_path):
    """
    Process all JSON files in a directory: delete existing documents, insert new ones, update field structure, and compute unique terms.

    :param directory_path: The path to the directory containing JSON files
    """
    # Delete all documents from the documents collection
    print("Deleting all existing documents from the database...")
    result = documents.delete_many({})
    print(f"Deleted {result.deleted_count} documents from the database.")

    # Clear the existing field structure
    print("Clearing existing field structure...")
    from database_setup import field_structure
    field_structure.delete_many({})
    print("Field structure cleared.")

    # Clear existing unique terms
    print("Clearing existing unique terms...")
    unique_terms_collection.delete_many({})
    print("Unique terms cleared.")

    start_time = time.time()
    files_to_process = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    total_files = len(files_to_process)
    
    processed_files = 0
    error_files = 0

    unique_terms_dict = {}  # Dictionary to hold unique terms per field

    progress_bar = tqdm(files_to_process, desc="Processing files", unit="file")

    for filename in progress_bar:
        file_path = os.path.join(directory_path, filename)
        try:
            document = process_file(file_path, unique_terms_dict)
            if document:
                inserted_id = insert_document(document)
                if inserted_id:
                    processed_files += 1
                else:
                    error_files += 1
            else:
                error_files += 1
        except Exception as e:
            error_files += 1
            print(f"\nError processing {filename}: {str(e)}")
        
        progress_bar.set_postfix({
            "Processed": processed_files,
            "Errors": error_files
        })

    # After processing all files, save unique terms
    save_unique_terms(unique_terms_dict)

    end_time = time.time()
    duration = end_time - start_time

    print("\nProcessing complete!")
    print(f"Total files: {total_files}")
    print(f"Successfully processed: {processed_files}")
    print(f"Errors: {error_files}")
    print(f"Time taken: {duration:.2f} seconds")

if __name__ == "__main__":
    data_directory = r'G:\My Drive\2024-2025\coding\rolls_txt\scratch4\lhyman6\OCR\data\borr\rolls'  # Change this to the path of your JSON files directory
    process_directory(data_directory)
