import os
import datetime
from collections import Counter

# Configuration variables
ROOT_DIRECTORY = r"G:\My Drive\2024-2025\coding\nosql_reader" 
OUTPUT_FILE = r"G:\My Drive\2024-2025\coding\nosql_reader/combined_project_files.txt"
FILE_TYPES = [".js", ".html", ".py", ".md", ".me", ".css", ".schema"]
SEPARATOR = "*" * 80

# Exclusion variables
EXCLUDED_DIRECTORIES = [".git", "util", "logs", "flask_session", "__pycache__"]
EXCLUDED_FILE_TYPES = [".ini"]

def get_file_statistics(root_dir):
    total_files = 0
    file_type_counts = Counter()

    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRECTORIES]
        for file in files:
            if not any(file.endswith(ext) for ext in EXCLUDED_FILE_TYPES):
                total_files += 1
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in FILE_TYPES:
                    file_type_counts[file_ext] += 1

    return total_files, file_type_counts

def get_file_structure(root_dir):
    structure = []
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRECTORIES]
        level = root.replace(root_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        structure.append(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for file in files:
            if not any(file.endswith(ext) for ext in EXCLUDED_FILE_TYPES):
                structure.append(f'{subindent}{file}')
    return '\n'.join(structure)

def combine_files(root_dir, output_file, file_types):
    total_files, file_type_counts = get_file_statistics(root_dir)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Write header information
        outfile.write(f"Project File Combination\n")
        outfile.write(f"Generated on: {datetime.datetime.now()}\n\n")
        outfile.write(f"Total files: {total_files}\n")
        outfile.write("File type counts:\n")
        for file_type, count in file_type_counts.items():
            outfile.write(f"  {file_type}: {count}\n")
        outfile.write("\nFile Structure:\n")
        outfile.write(get_file_structure(root_dir))
        outfile.write(f"\n\n{SEPARATOR}\n\n")

        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRECTORIES]
            for file in files:
                if any(file.endswith(ext) for ext in file_types) and not any(file.endswith(ext) for ext in EXCLUDED_FILE_TYPES):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, root_dir)
                    
                    outfile.write(f"File: {relative_path}\n")
                    outfile.write(f"{SEPARATOR}\n\n")
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                    except Exception as e:
                        outfile.write(f"Error reading file: {str(e)}\n")
                    
                    outfile.write(f"\n\n{SEPARATOR}\n\n")

if __name__ == "__main__":
    combine_files(ROOT_DIRECTORY, OUTPUT_FILE, FILE_TYPES)
    print(f"Combined files have been written to {OUTPUT_FILE}")