# requirements_generator.py
# Date: 2024-10-10
# Purpose: This script scans all Python files in a specified directory to find imported packages and generates a requirements.txt file.

import os
import re

def find_imports(directory):
    packages = set()
    # Regular expression to match import statements
    import_pattern = re.compile(r'^\s*(?:import|from)\s+([a-zA-Z0-9_]+)')

    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.py'):
                with open(os.path.join(dirpath, filename), 'r') as file:
                    for line in file:
                        match = import_pattern.match(line)
                        if match:
                            packages.add(match.group(1))

    return packages

def write_requirements(packages, output_file='requirements.txt'):
    with open(output_file, 'w') as file:
        for package in sorted(packages):
            file.write(f"{package}\n")

if __name__ == "__main__":
    directory_to_scan = '/home/lhyman6/coding/nosql_reader/setup'
    packages = find_imports(directory_to_scan)
    write_requirements(packages)
    print(f"Requirements written to requirements.txt with {len(packages)} packages.")
