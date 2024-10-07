# database_setup.py

from pymongo import MongoClient
from bson import ObjectId
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =======================
# Logging Configuration
# =======================
# Create a logger
logger = logging.getLogger('DatabaseSetupLogger')
logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs

# Create handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)  # Show only warnings and above in console

file_handler = logging.FileHandler('database_setup.log')
file_handler.setLevel(logging.DEBUG)  # Capture all debug and higher level logs in file

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Set formatter for handlers
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# =======================
# Database Functions
# =======================

def get_client():
    """Initialize and return a new MongoDB client."""
    try:
        print("this is the main sequence")
        mongo_uri = os.environ.get('MONGO_URI') # this SHOULD read from .env file but it doesnt not.
        #mongo_uri= "mongodb://admin:secret@mongodb:27017/admin" # this is correct and needs the /admin for the administrative db. REMOVE before prduction
        print(mongo_uri)

        if not mongo_uri:
            raise ValueError("MONGO_URI environment variable not set")
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=1000)
        # Test connection
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB.")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise e

def get_db(client):
    """Return the database instance."""
    return client['railroad_documents']

def initialize_database(client):
    db = get_db(client)
    # Create collections if they don't exist
    for collection_name in ['documents', 'unique_terms', 'field_structure']:
        if collection_name not in db.list_collection_names():
            db.create_collection(collection_name)
            logger.info(f"Created collection: {collection_name}")
    
    # Create necessary indexes
    documents = db['documents']
    existing_indexes = documents.index_information()
    if 'file_hash_1' not in existing_indexes:
        documents.create_index([("file_hash", 1)], unique=True)
    
    unique_terms = db['unique_terms']
    unique_terms.create_index([("term", 1)])
    unique_terms.create_index([("field", 1)])
    unique_terms.create_index([("type", 1)])
    
    field_structure = db['field_structure']
    field_structure.create_index([("field", 1)], unique=True)
    
    logger.info("Database initialized with required collections and indexes.")

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

def is_file_ingested(db, file_hash):
    """Check if a file has already been ingested based on its hash."""
    if not file_hash:
        return False
    try:
        documents = db['documents']
        ingested = documents.find_one({'file_hash': file_hash}) is not None
        logger.debug(f"File ingestion check for hash {file_hash}: {ingested}")
        return ingested
    except Exception as e:
        logger.error(f"Error checking ingestion status for hash {file_hash}: {e}")
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
        raise e
# In database_setup.py

def update_document(db, document_id, update_data):
    """
    Update a document's information.
    :param db: Database instance
    :param document_id: The ObjectId of the document to update
    :param update_data: A dictionary containing the fields to update
    :return: The number of documents modified
    """
    try:
        documents = db['documents']
        result = documents.update_one({"_id": ObjectId(document_id)}, {"$set": update_data})
        if result.matched_count == 0:
            logger.warning(f"No document found with _id: {document_id}")
        else:
            logger.info(f"Updated document with _id: {document_id}")
        return result.modified_count
    except Exception as e:
        logger.error(f"Error updating document {document_id}: {e}")
        raise e

def delete_document(db, document_id):
    """
    Delete a document from the database.
    :param db: Database instance
    :param document_id: The ObjectId of the document to delete
    :return: The number of documents deleted
    """
    try:
        documents = db['documents']
        result = documents.delete_one({"_id": ObjectId(document_id)})
        if result.deleted_count == 0:
            logger.warning(f"No document found with _id: {document_id}")
        else:
            logger.info(f"Deleted document with _id: {document_id}")
        return result.deleted_count
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise e
def get_field_structure(db):
    """
    Get the current field structure.
    :param db: Database instance
    :return: The current field structure as a dictionary
    """
    try:
        field_structure_collection = db['field_structure']
        field_structure_doc = field_structure_collection.find_one({})
        if field_structure_doc:
            # Remove '_id' if present
            field_structure_doc.pop('_id', None)
            logger.info("Retrieved field structure from the database.")
            return field_structure_doc
        else:
            logger.warning("No field structure document found.")
            return {}
    except Exception as e:
        logger.error(f"Error retrieving field structure: {e}")
        raise e

def find_document_by_id(db, document_id):
    """
    Find a document by its ObjectId.
    :param db: Database instance
    :param document_id: The ObjectId of the document
    :return: The document, or None if not found
    """
    try:
        documents = db['documents']
        document = documents.find_one({"_id": ObjectId(document_id)})
        if document:
            logger.info(f"Found document with _id: {document_id}")
            return document
        else:
            logger.warning(f"No document found with _id: {document_id}")
            return None
    except Exception as e:
        logger.error(f"Error finding document {document_id}: {e}")
        raise e

def get_collections(db):
    """Retrieve and return references to the required collections."""
    try:
        documents = db['documents']
        unique_terms_collection = db['unique_terms']
        field_structure_collection = db['field_structure']
        return documents, unique_terms_collection, field_structure_collection
    except Exception as e:
        logger.error(f"Error getting collections: {e}")
        raise

# =======================
# Main Execution (Optional)
# =======================
if __name__ == "__main__":
    client = get_client()  # Get the MongoDB client
    initialize_database(client)  # Initialize the database
    logger.info("Database setup module executed directly.")
