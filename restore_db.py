# restore_db.py
# Date created: 2024-10-19
# Purpose: This script restores the MongoDB database 'railroad_documents' from the most recent backup found in the ../db_backup/ directory.

import os
import subprocess
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Get MongoDB credentials from environment variables
username = os.getenv("MONGO_INITDB_ROOT_USERNAME")
password = os.getenv("MONGO_INITDB_ROOT_PASSWORD")
database_name = "railroad_documents"  # Change this if needed
backup_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db_backup")

# Find the most recent backup directory
def get_latest_backup_dir(backup_root):
    backup_dirs = [d for d in os.listdir(backup_root) if os.path.isdir(os.path.join(backup_root, d))]
    if not backup_dirs:
        print("No backup directories found.")
        return None
    latest_backup = max(backup_dirs, key=lambda d: os.path.getmtime(os.path.join(backup_root, d)))
    return os.path.join(backup_root, latest_backup)

# Get the latest backup directory
latest_backup_dir = get_latest_backup_dir(backup_root)

if latest_backup_dir:
    # Create the mongorestore command
    command = [
        "mongorestore",
        f"--uri=mongodb://{username}:{password}@localhost:27017/{database_name}?authSource=admin",
        latest_backup_dir
    ]

    # Execute the restore command
    try:
        subprocess.run(command, check=True)
        print(f"Database '{database_name}' restored successfully from {latest_backup_dir}.")
    except subprocess.CalledProcessError as e:
        print(f"Restore failed: {e}")
else:
    print("No backup to restore from.")
