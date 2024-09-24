import os
import json
import re
import multiprocessing
from tqdm import tqdm

def clean_json(json_text):
    """
    Cleans the JSON text by removing control characters and extracting the valid JSON portion.
    """
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
    """
    Validates and cleans a single JSON file.
    Converts valid .txt files to .json and renames invalid ones to .bad.
    """
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
        # Rename the original .txt file by appending .bad
        bad_file_path = f"{file_path}.bad"
        try:
            os.rename(file_path, bad_file_path)
        except Exception as rename_error:
            return (filename, False, f"Failed to rename to .bad. Original error: {e}; Rename error: {rename_error}")
        return (filename, False, f"File renamed to .bad due to invalid JSON. Error: {e}")

def validate_and_replace_json_files(source_dir, num_workers):
    """
    Traverses the source directory recursively to find and process all .txt files.
    Utilizes multiprocessing for parallel processing.
    """
    # Collect all .txt file paths recursively
    file_paths = []
    print(f"🔍 Starting to walk through the source directory: {source_dir}\n")
    for root, dirs, files in os.walk(source_dir, followlinks=True):
        print(f"📂 Accessing directory: {root}")
        for f in files:
            if f.lower().endswith('.txt'):
                full_path = os.path.join(root, f)
                file_paths.append(full_path)
                print(f"📄 Found .txt file: {full_path}")
    total_files = len(file_paths)

    print(f"\n📊 Processing {total_files} .txt files with {num_workers} worker processes...\n")

    # Initialize counters
    cleaned_count = 0
    replaced_count = 0
    invalid_count = 0

    if total_files == 0:
        print("🚫 No .txt files found. Exiting the script.")
        return

    # Use multiprocessing Pool to process files in parallel
    with multiprocessing.Pool(num_workers) as pool:
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

    print(f"\n✅ Processing complete:")
    print(f"✔️  Valid JSON files replaced with .json: {replaced_count}")
    print(f"🧹 Files cleaned: {cleaned_count}")
    print(f"❌ Invalid or unreadable files renamed to .bad: {invalid_count}")

    print("\n📋 Detailed results:")
    for filename, is_valid, info in results:
        if is_valid:
            if info:
                print(f"✅ {filename} was cleaned and replaced with a .json file.")
            else:
                print(f"✅ {filename} is valid and replaced with a .json file.")
        else:
            print(f"❌ {filename} was invalid and renamed to .bad. Error: {info}")

if __name__ == "__main__":
    # Specify the source directory
    source_directory = "/home/lhyman/coding/nosql_reader/archives"

    # Calculate the number of worker processes (3/4 of available CPUs)
    num_cpus = multiprocessing.cpu_count()
    num_workers = max(1, int(num_cpus * 0.75))

    # Start the validation and replacement process
    validate_and_replace_json_files(source_directory, num_workers)
