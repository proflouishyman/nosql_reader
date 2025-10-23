# backup_db.py
# Date created: 2024-10-19
# Purpose: This script creates a backup of the MongoDB database 'railroad_documents' using credentials from a .env file and reports the backup size.

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from app.util.mongo_env import build_admin_auth_uri

# Load environment variables from the .env file
load_dotenv()

# Get MongoDB credentials from environment variables
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
    f"--uri={build_admin_auth_uri(host=os.getenv('MONGO_BACKUP_HOST', 'localhost'))}?authSource=admin",
    f"--db={database_name}",
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
