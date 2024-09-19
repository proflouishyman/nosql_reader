# File: database_setup.py
# Path: railroad_documents_project/database_setup.py

from pymongo import MongoClient, ASCENDING, DESCENDING
from bson import ObjectId

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['railroad_documents']
documents = db['documents']
field_structure = db['field_structure']
unique_terms_collection = db['unique_terms']  # New collection for unique terms

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

def update_field_structure(document):
    """
    Update the field structure based on a new document.
    
    :param document: The new document to analyze
    """
    new_structure = discover_fields(document)
    existing_structure = field_structure.find_one({"_id": "current_structure"})
    
    if existing_structure:
        merged_structure = merge_structures(existing_structure['structure'], new_structure)
        field_structure.update_one(
            {"_id": "current_structure"},
            {"$set": {"structure": merged_structure}},
            upsert=True
        )
    else:
        field_structure.insert_one({"_id": "current_structure", "structure": new_structure})

def get_field_structure():
    """
    Get the current field structure.
    
    :return: The current field structure
    """
    structure = field_structure.find_one({"_id": "current_structure"})
    return structure['structure'] if structure else {}

def insert_document(document_data):
    """
    Insert a new document into the database.
    
    :param document_data: A dictionary containing the document's information
    :return: The ObjectId of the inserted document
    """
    result = documents.insert_one(document_data)
    return result.inserted_id

def find_document_by_id(document_id):
    """
    Find a document by its ObjectId.
    
    :param document_id: The ObjectId of the document
    :return: The document, or None if not found
    """
    return documents.find_one({"_id": ObjectId(document_id)})

def find_documents(query, limit=10):
    """
    Find documents based on a query.
    
    :param query: A dictionary containing the search criteria
    :param limit: Maximum number of results to return (default 10)
    :return: A cursor containing the matching documents
    """
    return documents.find(query).limit(limit)

def update_document(document_id, update_data):
    """
    Update a document's information.
    
    :param document_id: The ObjectId of the document to update
    :param update_data: A dictionary containing the fields to update
    :return: The result of the update operation
    """
    result = documents.update_one({"_id": ObjectId(document_id)}, {"$set": update_data})
    return result.modified_count

def delete_document(document_id):
    """
    Delete a document from the database.
    
    :param document_id: The ObjectId of the document to delete
    :return: The result of the delete operation
    """
    result = documents.delete_one({"_id": ObjectId(document_id)})
    return result.deleted_count

# Create indexes for performance optimization
def create_indexes():
    """
    Create necessary indexes to optimize query performance.
    """
    # Text index for full-text search on all string fields
    documents.create_index([("$**", "text")])

    # Index for unique_terms_collection to optimize retrieval by field
    unique_terms_collection.create_index([("field", ASCENDING)])

    # Additional indexes can be created here as needed
    # Example:
    # documents.create_index([("specific_field", ASCENDING)])

if __name__ == "__main__":
    # Recalculate the field structure based on existing documents
    print("Initializing field structure...")
    field_structure.delete_many({})  # Clear existing field structure
    all_documents = documents.find()
    for doc in all_documents:
        update_field_structure(doc)
    print("Field structure initialized.")

    # Create necessary indexes
    print("Creating indexes...")
    create_indexes()
    print("Indexes created.")
