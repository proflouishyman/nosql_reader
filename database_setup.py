# File: database_setup.py
# Path: railroad_documents_project/database_setup.py

import sys
import logging
from pymongo import MongoClient, ASCENDING, DESCENDING
from bson import ObjectId

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def is_mongodb_running():
    try:
        # Connect using authenticated credentials
        client = MongoClient('mongodb://admin:secret@localhost:27017', serverSelectionTimeoutMS=1000)
        client.server_info()
        return True
    except Exception:
        return False

def check_mongodb():
    if not is_mongodb_running():
        logging.error("MongoDB is not running. Please start it before running this script.")
        print("\nTo start MongoDB:")
        if sys.platform.startswith('win'):
            print("1. Open a new Command Prompt as Administrator")
            print("2. Run the following command:")
            print("   net start MongoDB")
        elif sys.platform.startswith('darwin'):
            print("1. Open a Terminal")
            print("2. Run the following command:")
            print("   brew services start mongodb-community")
        elif sys.platform.startswith('linux'):
            print("1. Open a Terminal")
            print("2. Run the following command to start the MongoDB Docker container:")
            print("   sudo docker compose up -d")
        else:
            print("Please refer to MongoDB documentation for instructions on starting MongoDB on your operating system.")
        sys.exit(1)
    else:
        logging.info("MongoDB is running.")

class DatabaseSetup:
    def __init__(self):
        check_mongodb()
        # Connect to MongoDB with authentication
        self.client = MongoClient('mongodb://admin:secret@localhost:27017/')
        self.db = self.client['railroad_documents']
        self.documents = self.db['documents']
        self.field_structure = self.db['field_structure']
        self.unique_terms_collection = self.db['unique_terms']

    def discover_fields(self, document):
        """
        Recursively discover fields in a document.
        
        :param document: The document to analyze
        :return: A dictionary representing the field structure
        """
        structure = {}
        for key, value in document.items():
            if isinstance(value, dict):
                structure[key] = self.discover_fields(value)
            elif isinstance(value, list):
                if value:
                    if isinstance(value[0], dict):
                        structure[key] = [self.discover_fields(value[0])]
                    else:
                        structure[key] = [type(value[0]).__name__]
                else:
                    structure[key] = []
            else:
                structure[key] = type(value).__name__
        return structure

    def merge_structures(self, existing, new):
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
                self.merge_structures(existing[key], value)
            elif isinstance(value, list) and isinstance(existing[key], list):
                if value and existing[key]:
                    if isinstance(value[0], dict) and isinstance(existing[key][0], dict):
                        self.merge_structures(existing[key][0], value[0])
        return existing

    def update_field_structure(self, document):
        """
        Update the field structure based on a new document.
        Performs an upsert to avoid duplicate key errors.
        
        :param document: The new document to analyze
        """
        new_structure = self.discover_fields(document)
        merged_structure = {}

        # Attempt to retrieve the existing structure
        existing_structure = self.field_structure.find_one({"_id": "current_structure"})
        
        if existing_structure:
            # Merge the new structure with the existing one
            merged_structure = self.merge_structures(existing_structure['structure'], new_structure)
        else:
            # If no existing structure, use the new structure
            merged_structure = new_structure

        # Perform an upsert operation to update or insert the structure
        self.field_structure.update_one(
            {"_id": "current_structure"},
            {"$set": {"structure": merged_structure}},
            upsert=True
        )

    def get_field_structure(self):
        """
        Get the current field structure.
        
        :return: The current field structure
        """
        structure = self.field_structure.find_one({"_id": "current_structure"})
        return structure['structure'] if structure else {}

    def insert_document(self, document_data):
        """
        Insert a new document into the database.
        
        :param document_data: A dictionary containing the document's information
        :return: The ObjectId of the inserted document
        """
        result = self.documents.insert_one(document_data)
        return result.inserted_id

    def find_document_by_id(self, document_id):
        """
        Find a document by its ObjectId.
        
        :param document_id: The ObjectId of the document
        :return: The document, or None if not found
        """
        return self.documents.find_one({"_id": ObjectId(document_id)})

    def find_documents(self, query, limit=10):
        """
        Find documents based on a query.
        
        :param query: A dictionary containing the search criteria
        :param limit: Maximum number of results to return (default 10)
        :return: A cursor containing the matching documents
        """
        return self.documents.find(query).limit(limit)

    def update_document(self, document_id, update_data):
        """
        Update a document's information.
        
        :param document_id: The ObjectId of the document to update
        :param update_data: A dictionary containing the fields to update
        :return: The result of the update operation
        """
        result = self.documents.update_one({"_id": ObjectId(document_id)}, {"$set": update_data})
        return result.modified_count

    def delete_document(self, document_id):
        """
        Delete a document from the database.
        
        :param document_id: The ObjectId of the document to delete
        :return: The result of the delete operation
        """
        result = self.documents.delete_one({"_id": ObjectId(document_id)})
        return result.deleted_count

    def create_indexes(self):
        """
        Create necessary indexes to optimize query performance.
        """
        # Text index for full-text search on all string fields
        self.documents.create_index([("$**", "text")])

        # Index for unique_terms_collection to optimize retrieval by field
        self.unique_terms_collection.create_index([("field", ASCENDING)])

        # Additional indexes can be created here as needed
        # Example:
        # self.documents.create_index([("specific_field", ASCENDING)])

# Global instance of DatabaseSetup
db_setup = DatabaseSetup()

# Expose necessary variables and methods for import in app.py and other scripts
client = db_setup.client
db = db_setup.db
documents = db_setup.documents
field_structure = db_setup.field_structure
unique_terms_collection = db_setup.unique_terms_collection

# Expose methods for other scripts
insert_document = db_setup.insert_document
find_document_by_id = db_setup.find_document_by_id
find_documents = db_setup.find_documents
update_field_structure = db_setup.update_field_structure
get_field_structure = db_setup.get_field_structure
update_document = db_setup.update_document
delete_document = db_setup.delete_document
create_indexes = db_setup.create_indexes

def init_database():
    """
    Initialize the database structure and indexes.
    """
    logging.info("Dropping existing database...")
    db_setup.client.drop_database('railroad_documents')  # Drop the entire database
    logging.info("Existing database dropped.")

    logging.info("Initializing field structure...")
    db_setup.field_structure.delete_many({})  # Clear existing field structure
    all_documents = db_setup.documents.find()
    for doc in all_documents:
        db_setup.update_field_structure(doc)
    logging.info("Field structure initialized.")

    logging.info("Creating indexes...")
    db_setup.create_indexes()
    logging.info("Indexes created.")

if __name__ == "__main__":
    init_database()
