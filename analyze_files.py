import json
import os
from collections import defaultdict, Counter
from pprint import pprint

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # The content is wrapped in ```json ```, so we need to extract the JSON part
            json_content = content.split('```json')[1].split('```')[0]
            return json.loads(json_content), None
    except Exception as e:
        return None, str(e)

def analyze_json_structure(data, prefix=''):
    structure = defaultdict(lambda: {"types": set(), "count": 0})
    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            structure[new_prefix]["count"] += 1
            if isinstance(value, (dict, list)):
                sub_structure = analyze_json_structure(value, new_prefix)
                for sub_key, sub_value in sub_structure.items():
                    structure[sub_key]["types"].update(sub_value["types"])
                    structure[sub_key]["count"] += sub_value["count"]
            else:
                structure[new_prefix]["types"].add(type(value).__name__)
    elif isinstance(data, list):
        for item in data:
            sub_structure = analyze_json_structure(item, prefix)
            for sub_key, sub_value in sub_structure.items():
                structure[sub_key]["types"].update(sub_value["types"])
                structure[sub_key]["count"] += sub_value["count"]
    return structure

def analyze_json_documents(directory):
    file_count = 0
    error_count = 0
    structures = defaultdict(lambda: {"types": set(), "count": 0})
    error_types = Counter()
    
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            content, error = read_json_file(file_path)
            
            if content is not None:
                file_count += 1
                file_structure = analyze_json_structure(content)
                for key, info in file_structure.items():
                    structures[key]["types"].update(info["types"])
                    structures[key]["count"] += 1
            else:
                error_count += 1
                error_type = type(eval(error)).__name__
                error_types[error_type] += 1
                print(f"Error reading file {file_path}: {error}")

    return structures, file_count, error_count, error_types

# Specify the directory containing your JSON files
directory = r'G:\My Drive\2024-2025\coding\borr\rolls_txt\scratch4\lhyman6\OCR\data\borr\rolls'

results, successful_files, error_files, error_types = analyze_json_documents(directory)

print(f"\nAnalysis Complete")
print(f"Successfully parsed files: {successful_files}")
print(f"Files with errors: {error_files}")
print("\nError types encountered:")
for error_type, count in error_types.items():
    print(f"  {error_type}: {count}")

print("\nJSON Structure Analysis:")
for key, info in sorted(results.items()):
    print(f"{key}:")
    print(f"  Types: {', '.join(info['types'])}")
    print(f"  Present in {info['count']} out of {successful_files} successfully parsed files")
    print(f"  Consistency: {info['count'] / successful_files * 100:.2f}%")
    print()