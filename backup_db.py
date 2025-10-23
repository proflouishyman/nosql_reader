# backup_db.py
# Date created: 2024-10-19
# Purpose: This script creates a backup of the MongoDB database 'railroad_documents' using credentials from a .env file and reports the backup size.

import os
import subprocess
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Get MongoDB credentials from environment variables
# Use the canonical MONGO_ROOT_* names so the script matches the updated .env file.
username = os.getenv("MONGO_ROOT_USERNAME")
password = os.getenv("MONGO_ROOT_PASSWORD")
database_name = "railroad_documents"  # Change this if needed
backup_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db_backup")

# Create a timestamped directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_dir = os.path.join(backup_root, f"backup_{timestamp}")

# Create the backup directory if it doesn't exist
os.makedirs(backup_dir, exist_ok=True)

# Create the mongodump command
command = [
    "mongodump",
    f"--uri=mongodb://{username}:{password}@localhost:27017/{database_name}?authSource=admin",
    f"--out={backup_dir}"
]

# Execute the backup command
try:
    subprocess.run(command, check=True)
    print(f"Backup successful! Files are stored in: {backup_dir}")

    # Calculate the size of the backup directory
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(backup_dir):
        for file in filenames:
            fp = os.path.join(dirpath, file)
            total_size += os.path.getsize(fp)

    # Convert size to MB
    total_size_mb = total_size / (1024 * 1024)
    print(f"Total backup size: {total_size_mb:.2f} MB")
except subprocess.CalledProcessError as e:
    print(f"Backup failed: {e}")
