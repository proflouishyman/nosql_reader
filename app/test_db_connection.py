# test_db_connection.py

import os
import logging
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TestDBLogger')

def get_client():
    """Initialize and return a new MongoDB client."""
    try:
        mongo_uri = os.environ.get('MONGO_URI')
        if not mongo_uri:
            raise ValueError("MONGO_URI environment variable not set")
        print(mongo_uri)
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        # Test connection
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB.")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise e

def get_db(client):
    """Return the database instance."""
    return client['railroad_documents']  # Adjust if your database name is different

def main():
    try:
        
        client = get_client()
        db = get_db(client)
        unique_terms_collection = db['unique_terms']

        # Check the total number of documents
        total_docs = unique_terms_collection.count_documents({})
        print(f"Total documents in unique_terms: {total_docs}")

        # Retrieve a sample document
        sample_doc = unique_terms_collection.find_one()
        print("\nSample document from unique_terms collection:")
        print(sample_doc)

        # Retrieve distinct 'type' field values
        type_values = unique_terms_collection.distinct('type')
        print("\nDistinct 'type' values in unique_terms collection:")
        print(type_values)

        # Count documents for each 'type' value
        print("\nDocument count for each 'type' value:")
        for type_value in type_values:
            count = unique_terms_collection.count_documents({'type': type_value})
            print(f"Type '{type_value}': {count} documents")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
