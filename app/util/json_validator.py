import os
import json
import re
import multiprocessing
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

def validate_and_replace_json_files(source_dir, num_workers):
    # Collect all .txt file paths recursively
    file_paths = []
    for root, dirs, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith('.txt'):
                file_paths.append(os.path.join(root, f))
    total_files = len(file_paths)

    print(f"Processing {total_files} .txt files with {num_workers} worker processes...")

    # Initialize counters
    cleaned_count = 0
    replaced_count = 0
    invalid_count = 0

    # Use multiprocessing Pool to process files in parallel
    with multiprocessing.Pool(num_workers) as pool:
        # Use imap_unordered for better performance and integrate with tqdm
        results = []
        for result in tqdm(pool.imap_unordered(validate_json_file, file_paths), total=total_files, desc="Validating JSON files"):
            results.append(result)

    # Process the results
    for filename, is_valid, info in results:
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
    # Specify the source directory
    source_directory = r"G:\My Drive\2024-2025\coding\rolls_txt"

    # Calculate the number of worker processes (3/4 of available CPUs)
    num_cpus = multiprocessing.cpu_count()
    num_workers = max(1, int(num_cpus * 0.75))

    # Start the validation and replacement process
    validate_and_replace_json_files(source_directory, num_workers)
