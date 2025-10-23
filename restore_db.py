# restore_db.py
# Date created: 2024-10-19
# Purpose: This script restores the MongoDB database 'railroad_documents' from the most recent backup found in the ../db_backup/ directory.

import os
import subprocess
from datetime import datetime
from dotenv import load_dotenv
import getpass

# Load environment variables from the .env file
load_dotenv()

# Get MongoDB credentials from environment variables
# Use the canonical MONGO_ROOT_* names so credential loading matches .env.
username = os.getenv("MONGO_ROOT_USERNAME")
password = os.getenv("MONGO_ROOT_PASSWORD")
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

    # Construct the MongoDB URI without the database name
    uri = f"mongodb://{username}@localhost:27017/?authSource=admin"

    # Option 1: Use --nsInclude to specify the database
    command = [
        "mongorestore",
        f"--uri={uri}",
        f"--nsInclude={database_name}.*",
        latest_backup_dir
    ]

    # Option 2: Point directly to the specific database directory
    # Uncomment the following lines if you prefer this method
    """
    backup_dir = os.path.join(latest_backup_dir, database_name)
    if not os.path.exists(backup_dir):
        print(f"Backup directory for database '{database_name}' does not exist: {backup_dir}")
    else:
        command = [
            "mongorestore",
            f"--uri={uri}",
            backup_dir
        ]
    """

    # Execute the restore command
    try:
        # If password is required via URI, you might need to include it securely
        # Alternatively, use a config file or prompt
        # Here, we'll include it directly for simplicity, but be cautious
        # Update the URI to include the password
        uri_with_password = f"mongodb://{username}:{password}@localhost:27017/?authSource=admin"
        command_with_password = [
            "mongorestore",
            f"--uri={uri_with_password}",
            f"--nsInclude={database_name}.*",
            latest_backup_dir
        ]

        subprocess.run(command_with_password, check=True)
        print(f"Database '{database_name}' restored successfully from {latest_backup_dir}.")
    except subprocess.CalledProcessError as e:
        print(f"Restore failed: {e}")
else:
    print("No backup to restore from.")
