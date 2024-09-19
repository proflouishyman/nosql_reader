# File: data_processing.py
# Path: railroad_documents_project/data_processing.py

import os
import json
from database_setup import documents, insert_document, update_field_structure
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

def process_file(file_path):
    """
    Process a single JSON file: load it, add filename, update field structure.

    :param file_path: The path to the JSON file
    :return: The loaded JSON data with filename added, or None if an error occurs
    """
    filename = os.path.basename(file_path)
    json_data = load_json_file(file_path)
    if json_data:
        # Add filename as a field if it's not already present
        if 'filename' not in json_data:
            json_data['filename'] = filename
        update_field_structure(json_data)  # Update field structure for each new document
        return json_data
    return None

def process_directory(directory_path):
    """
    Process all JSON files in a directory: delete existing documents, insert new ones, update field structure.

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

    start_time = time.time()
    files_to_process = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    total_files = len(files_to_process)
    
    processed_files = 0
    error_files = 0

    progress_bar = tqdm(files_to_process, desc="Processing files", unit="file")

    for filename in progress_bar:
        file_path = os.path.join(directory_path, filename)
        try:
            document = process_file(file_path)
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

    end_time = time.time()
    duration = end_time - start_time

    print("\nProcessing complete!")
    print(f"Total files: {total_files}")
    print(f"Successfully processed: {processed_files}")
    print(f"Errors: {error_files}")
    print(f"Time taken: {duration:.2f} seconds")

if __name__ == "__main__":
    data_directory = r'G:\My Drive\2024-2025\coding\borr\json_files'  # Change this to the path of your JSON files directory
    process_directory(data_directory)
