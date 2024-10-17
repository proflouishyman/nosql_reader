# database_setup.py

from pymongo import MongoClient, ASCENDING, DESCENDING, UpdateOne
from bson import ObjectId
import logging
import os
from dotenv import load_dotenv
import pymongo
from pymongo import UpdateOne

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
        
        mongo_uri = os.environ.get('MONGO_URI') # this SHOULD read from .env file but sometimes does not.
        #mongo_uri= "mongodb://admin:secret@mongodb:27017/admin" # this is correct and needs the /admin for the administrative db. REMOVE before prduction
        #print(mongo_uri)

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


def clean_unique_terms_collection(db):
    """Remove documents with null term, field, or type."""
    result = db['unique_terms'].delete_many({
        "$or": [
            {"term": None},
            {"field": None},
            {"type": None}
        ]
    })
    logger.info(f"Removed {result.deleted_count} documents with null term, field, or type.")



def initialize_database(client):
    db = get_db(client)
    # Create collections if they don't exist
    for collection_name in ['documents', 'unique_terms', 'field_structure']:
        if collection_name not in db.list_collection_names():
            db.create_collection(collection_name)
            logger.info(f"Created collection: {collection_name}")
    
  
    # Clean unique_terms collection
    clean_unique_terms_collection(db)


    # Create necessary indexes
    documents = db['documents']
    existing_indexes = documents.index_information()
    if 'file_hash_1' not in existing_indexes:
        documents.create_index([("file_hash", 1)], unique=True)
        logger.info("Created unique index on 'file_hash' in 'documents' collection.")
    
    unique_terms = db['unique_terms']
    # Unique compound index on term, field, and type
    unique_terms.create_index(
        [("term", ASCENDING), ("field", ASCENDING), ("type", ASCENDING)],
        unique=True,
        name="unique_term_field_type"
    )
    logger.info("Created unique compound index on 'term', 'field', and 'type' in 'unique_terms' collection.")
    
    # Compound index on field, type, and frequency for efficient sorting
    unique_terms.create_index(
        [("field", ASCENDING), ("type", ASCENDING), ("frequency", DESCENDING)],
        name="field_type_frequency_idx"
    )
    logger.info("Created compound index on 'field', 'type', and 'frequency' in 'unique_terms' collection.")
    
    field_structure = db['field_structure']
    field_structure.create_index([("field", 1)], unique=True)
    logger.info("Created unique index on 'field' in 'field_structure' collection.")


    # Create unique index on 'term' in 'linked_entities' collection
    linked_entities = db['linked_entities']
    existing_le_indexes = linked_entities.index_information()
    if 'term_1' not in existing_le_indexes:
        linked_entities.create_index([("term", ASCENDING)], unique=True)
        logger.info("Created unique index on 'term' in 'linked_entities' collection.")
    
    
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
    
    
def discover_fields(document):
    """
    Recursively discover fields in a document.
    :param document: The document to analyze
    :return: A dictionary representing the field structure
    """
    structure = {}
    for key, value in document.items():
        if isinstance(value, dict):
            structure[key] = discover_fields(value)
        elif isinstance(value, list):
            if value:
                if isinstance(value[0], dict):
                    structure[key] = [discover_fields(value[0])]
                else:
                    structure[key] = [type(value[0]).__name__]
            else:
                structure[key] = []
        else:
            structure[key] = type(value).__name__
    return structure

def merge_structures(existing, new):
    """
    Merge two field structures.
    :param existing: The existing field structure
    :param new: The new field structure to merge
    :return: The merged field structure
    """
    for key, value in new.items():
        if key not in existing:
            existing[key] = value
        elif isinstance(value, dict) and isinstance(existing[key], dict):
            merge_structures(existing[key], value)
        elif isinstance(value, list) and isinstance(existing[key], list):
            if value and existing[key]:
                if isinstance(value[0], dict) and isinstance(existing[key][0], dict):
                    merge_structures(existing[key][0], value[0])
    return existing

def update_field_structure(db, document):
    """
    Update the field structure based on a new document.
    :param db: Database instance
    :param document: The new document to analyze
    """
    field_structure_collection = db['field_structure']
    new_structure = discover_fields(document)
    merged_structure = {}

    # Attempt to retrieve the existing structure
    existing_structure = field_structure_collection.find_one({"_id": "current_structure"})

    if existing_structure:
        # Merge the new structure with the existing one
        merged_structure = merge_structures(existing_structure['structure'], new_structure)
    else:
        # If no existing structure, use the new structure
        merged_structure = new_structure

    # Perform an upsert operation to update or insert the structure
    field_structure_collection.update_one(
        {"_id": "current_structure"},
        {"$set": {"structure": merged_structure}},
        upsert=True
    )


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
def save_unique_terms(db, unique_terms_counter):
    """
    Save the unique terms counter to the database as individual documents.
    :param db: Database instance
    :param unique_terms_counter: Nested dictionary {field: {type: Counter()}}
    """
    unique_terms_collection = db['unique_terms']
    
    # Log the unique_terms_counter content
    logger.debug(f"Unique terms counter before saving: {unique_terms_counter}")
    
    # Prepare bulk operations
    operations = []
    for field, types in unique_terms_counter.items():
        if not field:
            logger.warning("Encountered a null or empty field. Skipping.")
            continue
        for term_type, counter in types.items():
            if not term_type:
                logger.warning("Encountered a null or empty type. Skipping.")
                continue
            for term, freq in counter.items():
                if not term:
                    logger.warning("Encountered a null or empty term. Skipping.")
                    continue
                operations.append(
                    pymongo.UpdateOne(
                        {"term": term, "field": field, "type": term_type},
                        {"$set": {"frequency": freq}},
                        upsert=True
                    )
                )
    
    logger.debug(f"Prepared {len(operations)} bulk operations for unique_terms.")
    
    if operations:
        try:
            result = unique_terms_collection.bulk_write(operations, ordered=False)
            logger.info(f"Bulk upserted {result.upserted_count + result.modified_count} unique terms.")
        except Exception as e:
            logger.error(f"Error bulk upserting unique terms: {e}")
            raise e
    else:
        logger.warning("No unique terms to upsert.")






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
    :return: The current field structure
    """
    field_structure_collection = db['field_structure']
    structure = field_structure_collection.find_one({"_id": "current_structure"})
    return structure['structure'] if structure else {}


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
    db = get_db(client)
    initialize_database(client)  # Initialize the database
    logger.info("Database setup module executed directly.")
