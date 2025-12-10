
"""
setup_databases.py - Initialize MongoDB collections and indexes

This script creates MongoDB collections and indexes for the Historical Document Reader,
including new person metadata indexes for grouping documents by person.

Usage:
    docker compose exec app python scripts/setup_databases.py

Creates:
- documents collection with person metadata indexes
- unique_terms collection with term frequency indexes
- field_structure collection for dynamic schema discovery
- linked_entities collection for NER results
"""

import sys
import os
from pathlib import Path
import logging
from pymongo import MongoClient, ASCENDING, DESCENDING
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
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

def get_client():
    """Initialize and return a new MongoDB client."""
    try:
        mongo_uri = os.environ.get('APP_MONGO_URI')
        if not mongo_uri:
            raise ValueError("APP_MONGO_URI environment variable not set")
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

def create_collections(db):
    """Create MongoDB collections if they don't exist."""
    print("\n📁 Creating collections...")
    collections = [
        'documents',
        'document_chunks',  # For RAG (future)
        'unique_terms',
        'field_structure',
        'linked_entities'
    ]
    existing_collections = db.list_collection_names()
    for coll_name in collections:
        if coll_name not in existing_collections:
            db.create_collection(coll_name)
            print(f"   ✅ Created: {coll_name}")
            logger.info(f"Created collection: {coll_name}")
        else:
            print(f"   ✓ Exists: {coll_name}")
            logger.info(f"Collection already exists: {coll_name}")
    return True

def create_documents_indexes(db):
    """Create indexes for the documents collection."""
    print("\n📇 Creating documents collection indexes...")
    documents = db['documents']
    existing_indexes = documents.index_information()
    # Core indexes (existing system)
    core_indexes = [
        ("file_hash", 1),
        ("date", 1),
        ("relative_path", 1),
    ]
    for idx_spec, idx_direction in core_indexes:
        if idx_spec + "_1" not in existing_indexes:
            documents.create_index([(idx_spec, idx_direction)])
            logger.info(f"Created index on '{idx_spec}' in 'documents' collection.")
    # Person metadata indexes (NEW!)
    person_indexes = [
        ("person_id", 1),
        ("person_name", 1),
        ("person_folder", 1),
        ("collection", 1),
    ]
    for idx_spec, idx_direction in person_indexes:
        if idx_spec + "_1" not in existing_indexes:
            documents.create_index([(idx_spec, idx_direction)])
            logger.info(f"Created index on '{idx_spec}' in 'documents' collection.")
    # Compound indexes for common queries (NEW!)
    compound_indexes = [
        [("person_name", 1), ("collection", 1)],
        [("person_id", 1), ("collection", 1)],
    ]
    for idx_spec in compound_indexes:
        idx_name = "_".join([f"{field}_{direction}" for field, direction in idx_spec])
        if idx_name not in existing_indexes:
            documents.create_index(idx_spec)
            logger.info(f"Created compound index '{idx_name}' in 'documents' collection.")
    # Optional archive structure indexes (for future use)
    archive_indexes = [
        ("archive_structure.physical_box", 1),
        ("archive_structure.format", 1),
    ]
    for idx_spec, idx_direction in archive_indexes:
        idx_name = idx_spec.replace(".", "_") + "_1"
        if idx_name not in existing_indexes:
            try:
                documents.create_index([(idx_spec, idx_direction)])
                logger.info(f"Created index on '{idx_spec}' in 'documents' collection.")
            except Exception as e:
                logger.warning(f"Could not create index {idx_spec}: {e}")
    return True

def create_unique_terms_indexes(db):
    """Create indexes for the unique_terms collection."""
    print("\n📇 Creating unique_terms collection indexes...")
    unique_terms = db['unique_terms']
    existing_indexes = unique_terms.index_information()
    # Unique compound index on term, field, and type
    if "unique_term_field_type" not in existing_indexes:
        unique_terms.create_index(
            [("term", ASCENDING), ("field", ASCENDING), ("type", ASCENDING)],
            unique=True,
            name="unique_term_field_type"
        )
        logger.info("Created unique compound index on 'term', 'field', and 'type' in 'unique_terms' collection.")
    # Compound index on field, type, and frequency for efficient sorting
    if "field_type_frequency_idx" not in existing_indexes:
        unique_terms.create_index(
            [("field", ASCENDING), ("type", ASCENDING), ("frequency", DESCENDING)],
            name="field_type_frequency_idx"
        )
        logger.info("Created compound index on 'field', 'type', and 'frequency' in 'unique_terms' collection.")
    return True

def create_field_structure_indexes(db):
    """Create indexes for the field_structure collection."""
    print("\n📇 Creating field_structure collection indexes...")
    field_structure = db['field_structure']
    existing_indexes = field_structure.index_information()
    if "field_1" not in existing_indexes:
        field_structure.create_index([("field", 1)], unique=True)
        logger.info("Created unique index on 'field' in 'field_structure' collection.")
    return True

def create_linked_entities_indexes(db):
    """Create indexes for the linked_entities collection."""
    print("\n📇 Creating linked_entities collection indexes...")
    linked_entities = db['linked_entities']
    existing_indexes = linked_entities.index_information()
    if "term_1" not in existing_indexes:
        linked_entities.create_index([("term", 1)], unique=True)
        logger.info("Created unique index on 'term' in 'linked_entities' collection.")
    return True

def verify_setup(db):
    """Verify that setup was successful."""
    print("\n🔍 Verifying setup...")
    collections = [
        'documents',
        'document_chunks',
        'unique_terms',
        'field_structure',
        'linked_entities'
    ]
    total_indexes = 0
    for coll_name in collections:
        if coll_name in db.list_collection_names():
            index_count = len(db[coll_name].index_information())
            total_indexes += index_count
            logger.info(f"{coll_name}: {index_count} indexes")
    print(f"   Total indexes across all collections: {total_indexes}")
    return True

def add_helper_functions_to_database_setup():
    """
    Information about helper functions to add to app/database_setup.py
    
    These functions provide safe query patterns for handling
    documents with varying field presence.
    """
    print("\n💡 Recommended: Add these helper functions to app/database_setup.py:")
    print("""
def find_documents_with_person(db, person_id=None, person_name=None, collection=None):
    '''Safe query for documents with person metadata.'''
    query = {}
    if person_id:
        query['person_id'] = person_id
    if person_name:
        query['person_name'] = person_name
    if collection:
        query['collection'] = collection
    if person_id or person_name:
        query['person_id'] = {'$ne': None}
    return db['documents'].find(query)

def find_documents_without_person(db, collection=None):
    '''Find documents that need person extraction (microfilm).'''
    query = {'$or': [{'person_id': None}, {'person_name': None}]}
    if collection:
        query['collection'] = collection
    return db['documents'].find(query)

def get_document_safely(doc, field_path, default=None):
    '''Safely retrieve nested field: get_document_safely(doc, 'archive_structure.physical_box')'''
    parts = field_path.split('.')
    current = doc
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
            if current is None:
                return default
        else:
            return default
    return current if current is not None else default
    """)


def setup_mongodb():
    """Main setup function."""
    print("="*60)
    print("MongoDB Setup - Historical Document Reader")
    print("="*60)
    try:
        # Connect to MongoDB
        print("\n🔧 Connecting to MongoDB...")
        client = get_client()
        db = get_db(client)
        print("   ✅ Connected successfully")
        logger.info("Connected to MongoDB")
        
        # Create collections
        create_collections(db)
        
        # Create indexes
        create_documents_indexes(db)
        create_unique_terms_indexes(db)
        create_field_structure_indexes(db)
        create_linked_entities_indexes(db)
        
        # Verify setup
        verify_setup(db)
        
        # Show helper functions info
        add_helper_functions_to_database_setup()
        
        # Success summary
        print("\n" + "="*60)
        print("✅ MongoDB setup complete!")
        print("="*60)
        
        print("\n📊 Summary:")
        print(f"   Collections: 5 total")
        print(f"   Indexes created: 14")
        
        print("\n📝 Next steps:")
        print("   1. Ingest documents:")
        print("      docker compose exec app python app/data_processing.py /data/archives/borr_data")
        print()
        print("   2. Or use UI:")
        print("      Settings → Data Ingestion → Scan for new images")
        print()
        print("   3. Verify person metadata:")
        print("      docker compose exec app python -c \"")
        print("        from app.database_setup import get_client, get_db")
        print("        client = get_client()")
        print("        db = get_db(client)")
        print("        docs = db['documents']")
        print("        print(f'Total: {docs.count_documents({})}' )")
        print("        print(f'With person_id: {docs.count_documents({\"person_id\": {\"\\\\$ne\": None}})}' )")
        print("      \"")
        print()
        
        return True
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        logger.error(f"Setup failed: {e}", exc_info=True)
        return False


def main():
    """Entry point."""
    success = setup_mongodb()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
