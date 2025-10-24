# drop_db.py
"""Utility for dropping the MongoDB database using the canonical helpers."""

import sys
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so we can import database_setup when
# this script is executed directly (python app/util/delete_db.py).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Import the shared connection helpers so we honour the naming conventions.
# ---------------------------------------------------------------------------
from app.database_setup import get_client, get_db

# Load environment variables from .env file
load_dotenv()

def drop_database():
    """Drop the specified MongoDB database."""
    client = None
    # Track the name for logging even if connection fails.
    database_name = "unknown"
    try:
        print("Connecting to db")
        # Reuse the centralised helper so we always read MONGO_URI correctly.
        client = get_client()

        # Use the helper again to obtain the configured database handle.
        db = get_db(client)

        # Derive the database name from the handle to avoid hard-coded strings.
        database_name = db.name

        # Drop the database using the canonical client connection.
        client.drop_database(database_name)
        print(f"Database '{database_name}' dropped successfully.")
    except Exception as e:
        print(f"Error dropping database '{database_name}': {e}")
    finally:
        if client is not None:
            # Close the client so the connection pool is released cleanly.
            client.close()

if __name__ == "__main__":
    drop_database()
