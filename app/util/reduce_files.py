import os
import random

def reduce_files_recursively(directory, reduction_percentage=95):
    # Check if the directory exists
    if not os.path.isdir(directory):
        print(f"The directory {directory} does not exist.")
        return

    # Collect all files in the directory and its subdirectories
    all_files = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            all_files.append(os.path.join(dirpath, filename))

    total_files = len(all_files)

    if total_files == 0:
        print("No files found in the directory.")
        return

    # Calculate the number of files to delete
    files_to_delete_count = int(total_files * (reduction_percentage / 100))

    # Randomly select files to delete
    files_to_delete = random.sample(all_files, min(files_to_delete_count, total_files))

    # Delete the selected files
    for file in files_to_delete:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting file {file}: {e}")

    remaining_files = total_files - len(files_to_delete)
    print(f"Reduced the number of files by {len(files_to_delete)}. Remaining files: {remaining_files}")

if __name__ == "__main__":
    # Specify the directory you want to reduce files from
    target_directory = "/home/lhyman6/coding/nosql_reader/archives/paper"  # Change this to your target directory
    reduce_files_recursively(target_directory)
    target_directory = "/home/lhyman6/coding/nosql_reader/archives/rolls"  # Change this to your target directory
    reduce_files_recursively(target_directory)
