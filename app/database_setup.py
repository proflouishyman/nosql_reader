from pymongo import MongoClient
from bson import ObjectId
import logging
import os

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


def get_client(): #changed for containerization
    """Initialize and return a new MongoDB client."""
    try:
        mongo_uri = os.environ.get('MONGO_URI')
        if not mongo_uri:
            raise ValueError("MONGO_URI environment variable not set")
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=1000)
        logger.info("Successfully connected to MongoDB.")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise e

def initialize_database(client):
    db = get_db(client)
    # Create collections if they don't exist
    for collection_name in ['documents', 'unique_terms', 'field_structure']:
        if collection_name not in db.list_collection_names():
            db.create_collection(collection_name)
            logger.info(f"Created collection: {collection_name}")
    
    # Create necessary indexes
    documents = db['documents']
    documents.create_index([("file_path", 1), ("file_hash", 1)], unique=True)
    
    unique_terms = db['unique_terms']
    unique_terms.create_index([("term", 1)])
    unique_terms.create_index([("field", 1)])
    unique_terms.create_index([("type", 1)])
    
    field_structure = db['field_structure']
    field_structure.create_index([("field", 1)], unique=True)
    
    logger.info("Database initialized with required collections and indexes.")



def get_db(client):
    """Return the database instance."""
    return client['railroad_documents']

def get_collections(db):
    """Return the collections used in the application."""
    documents = db['documents']
    unique_terms_collection = db['unique_terms']
    field_structure_collection = db['field_structure']
    logger.info("Accessed collections from the database.")
    return documents, unique_terms_collection, field_structure_collection

def insert_document(client, document):
    """Insert a document into the 'documents' collection."""
    db = get_db(client)
    try:
        documents = db['documents']
        documents.insert_one(document)
        logger.info("Inserted document into 'documents' collection.")
    except Exception as e:
        logger.error(f"Error inserting document: {e}")
        raise e

def update_document(client, document_id, update_data):
    """
    Update a document's information.
    :param client: MongoDB client
    :param document_id: The ObjectId of the document to update
    :param update_data: A dictionary containing the fields to update
    :return: The result of the update operation
    """
    db = get_db(client)
    documents = db['documents']
    result = documents.update_one({"_id": ObjectId(document_id)}, {"$set": update_data})
    return result.modified_count

def delete_document(client, document_id):
    """
    Delete a document from the database.
    :param client: MongoDB client
    :param document_id: The ObjectId of the document to delete
    :return: The result of the delete operation
    """
    db = get_db(client)
    documents = db['documents']
    result = documents.delete_one({"_id": ObjectId(document_id)})
    return result.deleted_count

def update_field_structure(client, json_data):
    """Update the field structure in the 'field_structure' collection based on json_data."""
    db = get_db(client)
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

def is_file_ingested(client, file_path, file_hash):
    """Check if a file has already been ingested based on its path and hash."""
    if not file_hash:
        return False
    db = get_db(client)
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

def save_unique_terms(client, unique_terms_dict):
    """Save the unique terms to the database in a flattened structure."""
    db = get_db(client)
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

def find_document_by_id(client, document_id):
    """
    Find a document by its ObjectId.
    :param client: MongoDB client
    :param document_id: The ObjectId of the document
    :return: The document, or None if not found
    """
    db = get_db(client)
    documents = db['documents']
    try:
        return documents.find_one({"_id": ObjectId(document_id)})
    except Exception as e:
        logger.error(f"Error finding document by ID: {e}")
        return None

def get_field_structure(client):
    """
    Get the current field structure.
    :param client: MongoDB client
    :return: The current field structure
    """
    db = get_db(client)
    field_structure_collection = db['field_structure']
    structure = field_structure_collection.find_one({"_id": "current_structure"})
    return structure['structure'] if structure else {}


# =======================
# Main Execution (Optional)
# =======================
if __name__ == "__main__":
    client = get_client()  # Get the MongoDB client
    initialize_database(client) #recent suggestion. lets see how it shifts db behavior
    logger.info("Database setup module executed directly.")
    db = get_db(client)    # Get the database
    documents, unique_terms_collection, field_structure_collection = get_collections(db)
    logger.info("Database setup module executed directly.")
