# Script to randomly select and copy 50 files from one directory to another
# Date: 2024-08-20

import os
import shutil
import random

def copy_random_files(src_dir, dest_dir, num_files):
    """
    Randomly selects and copies a specified number of files from the source directory to the destination directory.

    Parameters:
    - src_dir (str): Path to the source directory
    - dest_dir (str): Path to the destination directory
    - num_files (int): Number of files to copy
    """
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # List all files in the source directory
    all_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

    # Check if there are enough files to select from
    if len(all_files) < num_files:
        raise ValueError(f"Not enough files in the source directory. Found {len(all_files)}, but need {num_files}.")

    # Randomly select files
    selected_files = random.sample(all_files, num_files)

    # Copy selected files to the destination directory
    for file_name in selected_files:
        src_file = os.path.join(src_dir, file_name)
        dest_file = os.path.join(dest_dir, file_name)
        shutil.copy(src_file, dest_file)
        print(f"Copied: {file_name}")

# Example usage
source_directory = "../texts"
destination_directory = "../random_texts"
number_of_files_to_copy = 50

copy_random_files(source_directory, destination_directory, number_of_files_to_copy)
