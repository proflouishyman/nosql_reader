# drop_db.py
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get MongoDB connection details from environment variables
MONGODB_URI = os.getenv("MONGODB_URI")  # Ensure you have this set in your .env file
DATABASE_NAME = "railroad_documents"      # Replace with the name of the database you want to delete

def drop_database():
    """Drop the specified MongoDB database."""
    try:
        print("Connecting to db")
        # Connect to the MongoDB server
        client = MongoClient(MONGODB_URI)
        # Drop the database
        client.drop_database(DATABASE_NAME)
        print(f"Database '{DATABASE_NAME}' dropped successfully.")
    except Exception as e:
        print(f"Error dropping database '{DATABASE_NAME}': {e}")

if __name__ == "__main__":
    drop_database()
