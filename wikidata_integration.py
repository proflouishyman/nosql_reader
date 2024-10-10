# wikidata_integration.py

import os
import logging
from pymongo import UpdateOne, MongoClient
from dotenv import load_dotenv
import requests
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# =======================
# Logging Configuration
# =======================
logger = logging.getLogger('WikidataIntegrationLogger')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    file_handler = logging.FileHandler('wikidata_integration.log', mode='a')
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# =======================
# Database Functions
# =======================

def get_client():
    """Initialize and return a new MongoDB client."""
    try:
        mongo_uri = os.environ.get('MONGO_URI')
        if not mongo_uri:
            raise ValueError("MONGO_URI environment variable not set")
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
    return client['railroad_documents']

def get_collections(db):
    """Retrieve and return references to the required collections."""
    try:
        unique_terms_collection = db['unique_terms']
        linked_entities_collection = db['linked_entities']
        return unique_terms_collection, linked_entities_collection
    except Exception as e:
        logger.error(f"Error getting collections: {e}")
        raise e

# =======================
# Utility Functions
# =======================

def fetch_wikidata_entity(term):
    """
    Fetch Wikidata entity ID for a given term using the Wikidata API.
    Returns the entity ID if found, else None.
    """
    try:
        url = "https://www.wikidata.org/w/api.php"
        params = {
            'action': 'wbsearchentities',
            'search': term,
            'language': 'en',
            'format': 'json'
        }
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        if 'search' in data and len(data['search']) > 0:
            # Return the first matching entity ID
            return data['search'][0]['id']
        else:
            return None
    except Exception as e:
        logger.error(f"Error fetching Wikidata entity for term '{term}': {e}")
        return None

# =======================
# Main Processing Function
# =======================

def link_entities(db):
    """
    Link entities from the unique_terms collection to Wikidata.
    Populate the linked_entities collection with the results.
    """
    unique_terms_collection, linked_entities_collection = get_collections(db)
    
    # Fetch unique terms that are likely entities (filter by type)
    # Adjust the filter based on your specific schema and requirements
    entity_types = ['PERSON', 'ORG', 'GPE', 'LOC', 'EVENT', 'WORK_OF_ART']  # Example types
    cursor = unique_terms_collection.find(
        {"type": {"$in": entity_types}},
        {"_id": 0, "term": 1, "field":1, "type":1, "frequency":1}
    )
    
    aggregated_entities = defaultdict(int)
    processed_count = 0
    linked_count = 0
    
    logger.info("Starting Wikidata entity linking process.")
    
    for term_doc in tqdm(cursor, desc="Linking entities"):
        term = term_doc.get('term')
        field = term_doc.get('field')
        ent_type = term_doc.get('type')
        frequency = term_doc.get('frequency', 1)
        
        if not term:
            logger.warning("Encountered a term with no value. Skipping.")
            continue
        
        # Fetch Wikidata ID using the Wikidata API
        wikidata_id = fetch_wikidata_entity(term)
        
        if wikidata_id:
            key = (term.lower(), field, ent_type, wikidata_id)
            aggregated_entities[key] += frequency
            linked_count += 1
        processed_count += 1
    
    logger.debug(f"Aggregated linked entities: {aggregated_entities}")
    logger.info(f"Processed {processed_count} terms.")
    logger.info(f"Successfully linked {linked_count} entities.")
    
    # Prepare bulk operations for linked_entities
    operations = []
    for (term, field, ent_type, wikidata_id), freq in aggregated_entities.items():
        operations.append(
            UpdateOne(
                {"term": term, "field": field, "type": ent_type, "kb_id": wikidata_id},
                {"$inc": {"frequency": freq}},
                upsert=True
            )
        )
    
    logger.debug(f"Prepared {len(operations)} bulk operations for linked_entities.")
    
    if operations:
        try:
            result = linked_entities_collection.bulk_write(operations, ordered=False)
            logger.info(f"Bulk upserted {result.upserted_count + result.modified_count} linked entities.")
        except Exception as e:
            logger.error(f"Error bulk upserting linked entities: {e}")
            raise e
    else:
        logger.warning("No linked entities to upsert.")
    
    logger.info("Wikidata entity linking process completed.")

# =======================
# Main Execution
# =======================

if __name__ == "__main__":
    try:
        client = get_client()
        db = get_db(client)
        logger.info("Connected to MongoDB.")
        
        # Perform Wikidata entity linking
        link_entities(db)
        
    except Exception as e:
        logger.error(f"An error occurred during Wikidata entity linking: {e}", exc_info=True)
