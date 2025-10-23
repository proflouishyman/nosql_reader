# restore_db.py
# Date created: 2024-10-19
# Purpose: This script restores the MongoDB database 'railroad_documents' from the most recent backup found in the ../db_backup/ directory.

import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv
import getpass

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from app.util.mongo_env import build_admin_auth_uri, resolve_credentials

# Load environment variables from the .env file
load_dotenv()

# Get MongoDB credentials from environment variables
creds = resolve_credentials()
username = creds.username or "admin"
password = creds.password
database_name = "railroad_documents"  # Change this if needed
backup_root = "../db_backup"

# Function to securely get the password if not set in .env
def get_password():
    if not password:
        return getpass.getpass(prompt='Enter MongoDB password: ')
    return password

# Find the most recent backup directory
def get_latest_backup_dir(backup_root):
    try:
        backup_dirs = [d for d in os.listdir(backup_root) if os.path.isdir(os.path.join(backup_root, d))]
        if not backup_dirs:
            print("No backup directories found.")
            return None
        # Sort directories by modification time
        latest_backup = max(backup_dirs, key=lambda d: os.path.getmtime(os.path.join(backup_root, d)))
        return os.path.join(backup_root, latest_backup)
    except Exception as e:
        print(f"Error accessing backup directories: {e}")
        return None

# Get the latest backup directory
latest_backup_dir = get_latest_backup_dir(backup_root)
print(f"Latest backup directory: {latest_backup_dir}")

if latest_backup_dir:
    # Securely get the password
    password = get_password()

    # Execute the restore command
    try:
        uri_with_password = build_admin_auth_uri(
            host=os.getenv('MONGO_BACKUP_HOST', 'localhost'),
            username=username,
            password=password,
        )
        command = [
            "mongorestore",
            f"--uri={uri_with_password}?authSource={creds.auth_db}",
            f"--nsInclude={database_name}.*",
            latest_backup_dir
        ]

        subprocess.run(command, check=True)
        print(f"Database '{database_name}' restored successfully from {latest_backup_dir}.")
    except subprocess.CalledProcessError as e:
        print(f"Restore failed: {e}")
else:
    print("No backup to restore from.")
