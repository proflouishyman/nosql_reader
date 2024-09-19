# 2024-08-20
# Script to clean JSON, validate, and replace valid files with .json extension.
# Invalid files remain as .txt. It tracks how many files were cleaned and replaced.

import os
import json
import re
from tqdm import tqdm

def clean_json(json_text):
    # Remove all control characters
    json_text = re.sub(r'[\x00-\x1F\x7F]', '', json_text)

    # Find the index of the first '{' and the last '}'
    start_index = json_text.find('{')
    end_index = json_text.rfind('}')

    # Extract the clean JSON string
    if start_index != -1 and end_index != -1:
        clean_json_text = json_text[start_index:end_index + 1]
        return clean_json_text
    else:
        raise ValueError("Invalid JSON format: Unable to find '{' or '}'.")

def validate_json_file(file_path):
    filename = os.path.basename(file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            json_content = file.read()

        # Clean the JSON content before validation
        cleaned_json_content = clean_json(json_content)
        cleaned = cleaned_json_content != json_content

        # Validate the cleaned JSON content
        json_data = json.loads(cleaned_json_content)

        # Define the new file path with .json extension
        base_name, _ = os.path.splitext(file_path)
        new_file_path = f"{base_name}.json"

        # Write the validated and cleaned JSON to the new file
        with open(new_file_path, 'w', encoding='utf-8') as file:
            json.dump(json_data, file, indent=4)

        # Remove the original .txt file
        os.remove(file_path)

        return (filename, True, cleaned)
    except Exception as e:
        return (filename, False, str(e))

def validate_and_replace_json_files(source_dir):
    file_paths = [
        os.path.join(source_dir, f)
        for f in os.listdir(source_dir)
        if os.path.isfile(os.path.join(source_dir, f)) and f.lower().endswith('.txt')
    ]
    total_files = len(file_paths)

    cleaned_count = 0
    replaced_count = 0
    invalid_count = 0

    print(f"Processing {total_files} .txt files...")

    results = []
    for file_path in tqdm(file_paths, desc="Validating JSON files"):
        result = validate_json_file(file_path)
        results.append(result)

        filename, is_valid, info = result
        if is_valid:
            if info:
                cleaned_count += 1
            replaced_count += 1
        else:
            invalid_count += 1

    print(f"\nProcessing complete:")
    print(f"Valid JSON files replaced with .json: {replaced_count}")
    print(f"Files cleaned: {cleaned_count}")
    print(f"Invalid or unreadable files remain as .txt: {invalid_count}")

    print("\nDetailed results:")
    for filename, is_valid, info in results:
        if is_valid:
            if info:
                print(f"{filename} was cleaned and replaced with a .json file.")
            else:
                print(f"{filename} is valid and replaced with a .json file.")
        else:
            print(f"{filename} is invalid or unreadable. Remains as .txt. Error: {info}")

if __name__ == "__main__":
    source_directory = r"G:\My Drive\2024-2025\coding\rolls_txt"
    validate_and_replace_json_files(source_directory)
