# database_setup.py

from pymongo import MongoClient
import logging

# =======================
# Logging Configuration
# =======================

# Create a custom logger
logger = logging.getLogger('DatabaseSetupLogger')
logger.setLevel(logging.INFO)  # Set to INFO to capture all levels

# Create handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler('database_setup.log')
file_handler.setLevel(logging.DEBUG)

# Create formatters and add them to handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# =======================
# Database Functions
# =======================

def get_db():
    """Initialize and return a new MongoDB connection."""
    try:
        client = MongoClient('mongodb://admin:secret@localhost:27017', serverSelectionTimeoutMS=1000)
        db = client['railroad_documents']
        logger.info("Successfully connected to MongoDB.")
        return db
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise e

def get_collections(db):
    """Return the collections used in the application."""
    documents = db['documents']
    unique_terms_collection = db['unique_terms']
    field_structure = db['field_structure']
    logger.info("Accessed collections from the database.")
    return documents, unique_terms_collection, field_structure

def insert_document(db, document):
    """Insert a document into the 'documents' collection."""
    try:
        documents = db['documents']
        documents.insert_one(document)
        logger.info("Inserted document into 'documents' collection.")
    except Exception as e:
        logger.error(f"Error inserting document: {e}")
        raise e

def update_field_structure(db, json_data):
    """Update the field structure in the 'field_structure' collection based on json_data."""
    try:
        field_structure = db['field_structure']
        # Flatten the JSON data to get all fields
        def flatten_json(y):
            out = {}

            def flatten(x, name=''):
                if isinstance(x, dict):
                    for a in x:
                        flatten(x[a], f'{name}{a}.')
                elif isinstance(x, list):
                    i = 0
                    for a in x:
                        flatten(a, f'{name}{i}.')
                        i += 1
                else:
                    out[name[:-1]] = type(x).__name__
            flatten(y)
            return out

        flat_json = flatten_json(json_data)

        # Update the field_structure collection
        for field, field_type in flat_json.items():
            field_structure.update_one(
                {'field': field},
                {'$addToSet': {'types': field_type}},
                upsert=True
            )
        logger.info("Updated field structure based on JSON data.")
    except Exception as e:
        logger.error(f"Error updating field structure: {e}")
        raise e

def is_file_ingested(db, file_path, file_hash):
    """Check if a file has already been ingested based on its path and hash."""
    if not file_hash:
        return False
    try:
        documents = db['documents']
        ingested = documents.find_one({
            'file_path': file_path,
            'file_hash': file_hash
        }) is not None
        logger.debug(f"File ingestion check for {file_path}: {ingested}")
        return ingested
    except Exception as e:
        logger.error(f"Error checking ingestion status for {file_path}: {e}")
        return False

def save_unique_terms(db, unique_terms_dict):
    """Save the unique terms to the database in a flattened structure."""
    unique_terms_collection = db['unique_terms']
    unique_terms_documents = []
    for field, terms in unique_terms_dict.items():
        for word, count in terms['words'].items():
            if count >= 2:
                unique_terms_documents.append({
                    "term": word,
                    "field": field,
                    "count": count,
                    "type": "word"
                })
        for phrase, count in terms['phrases'].items():
            if count >= 2:
                unique_terms_documents.append({
                    "term": phrase,
                    "field": field,
                    "count": count,
                    "type": "phrase"
                })
    try:
        unique_terms_collection.delete_many({})
        if unique_terms_documents:
            unique_terms_collection.insert_many(unique_terms_documents)
            unique_terms_collection.create_index([("term", 1)])
            unique_terms_collection.create_index([("field", 1)])
            unique_terms_collection.create_index([("type", 1)])
        logger.info("Saved unique terms to the database.")
    except Exception as e:
        logger.error(f"Error saving unique terms: {e}")

# =======================
# Main Execution (Optional)
# =======================

if __name__ == "__main__":
    # If you want to run this file directly for testing purposes
    db = get_db()
    documents, unique_terms_collection, field_structure = get_collections(db)
    logger.info("Database setup module executed directly.")
