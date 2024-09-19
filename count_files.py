# Script created on 2024-09-19 at 10:05 AM
# This script recursively counts all files in a specified subdirectory and shows progress indicators.

import os

def count_files(directory):
    total_files = 0
    for root, dirs, files in os.walk(directory):
        print(f"Traversing directory: {root}")
        total_files += len(files)
    return total_files

if __name__ == "__main__":
    directory_path = input("Enter the path of the directory: ")
    file_count = count_files(directory_path)
    print(f"Total number of files: {file_count}")
