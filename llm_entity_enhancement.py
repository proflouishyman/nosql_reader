# llm_entity_enhancement.py

import os
import logging
from pymongo import UpdateOne, MongoClient
from dotenv import load_dotenv
import openai
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# =======================
# Logging Configuration
# =======================
logger = logging.getLogger('LLMEnhancementLogger')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    file_handler = logging.FileHandler('llm_enhancement.log', mode='a')
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

def link_entity_with_llm(term, context):
    """
    Use LLM (e.g., OpenAI's GPT-4) to disambiguate and link an entity.
    Returns the Wikidata ID if successful, else None.
    """
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not set in environment variables.")
        return None

    openai.api_key = openai_api_key

    try:
        prompt = f"""
        Given the following context, disambiguate the entity "{term}" and provide its Wikidata ID.

        Context: {context}

        Only provide the Wikidata ID (e.g., Q42). If not found, respond with "Not Found".
        """
        response = openai.Completion.create(
            engine="text-davinci-003",  # Use the appropriate engine
            prompt=prompt,
            max_tokens=10,
            n=1,
            stop=None,
            temperature=0.3,
        )
        wikidata_id = response.choices[0].text.strip()
        if wikidata_id.lower() == "not found":
            return None
        return wikidata_id
    except Exception as e:
        logger.error(f"Error linking entity with LLM for term '{term}': {e}")
        return None

# =======================
# Main Processing Function
# =======================

def enhance_entities_with_llm(db):
    """
    Enhance entity linking using LLM for terms that haven't been linked yet.
    """
    unique_terms_collection, linked_entities_collection = get_collections(db)
    
    # Fetch unique terms that are entities and have not been linked yet
    entity_types = ['PERSON', 'ORG', 'GPE', 'LOC', 'EVENT', 'WORK_OF_ART']  # Example types
    cursor = unique_terms_collection.find(
        {"type": {"$in": entity_types}},
        {"_id": 1, "term": 1, "field":1, "type":1}
    )
    
    operations = []
    processed = 0
    enhanced = 0
    
    logger.info("Starting LLM-based entity enhancement.")
    
    for term_doc in tqdm(cursor, desc="Enhancing entities"):
        term_id = term_doc.get('_id')
        term = term_doc.get('term')
        field = term_doc.get('field')
        ent_type = term_doc.get('type')
        
        if not term:
            logger.warning(f"Document {_id} has no term. Skipping.")
            continue
        
        # Check if this term has already been linked
        existing_link = linked_entities_collection.find_one({"term": term.lower(), "field": field, "type": ent_type})
        if existing_link:
            continue  # Already linked
        
        # Prepare context (can be customized based on your data)
        context = f"Field: {field}, Term: {term}"
        
        # Use LLM to get Wikidata ID
        wikidata_id = link_entity_with_llm(term, context)
        
        if wikidata_id:
            operations.append(
                UpdateOne(
                    {"term": term.lower(), "field": field, "type": ent_type, "kb_id": wikidata_id},
                    {"$inc": {"frequency": 1}},
                    upsert=True
                )
            )
            enhanced += 1
            logger.debug(f"Enhanced term '{term}' with Wikidata ID '{wikidata_id}'.")
        else:
            logger.debug(f"Could not enhance term '{term}' with LLM.")
        
        processed += 1
    
    logger.info(f"Processed {processed} terms for LLM enhancement.")
    logger.info(f"Successfully enhanced {enhanced} entities with LLM.")
    
    # Execute bulk operations
    if operations:
        try:
            result = linked_entities_collection.bulk_write(operations, ordered=False)
            logger.info(f"Bulk upserted {result.upserted_count + result.modified_count} linked entities via LLM.")
        except Exception as e:
            logger.error(f"Error bulk upserting linked entities via LLM: {e}")
            raise e
    else:
        logger.warning("No entities to enhance with LLM.")
    
    logger.info("LLM-based entity enhancement completed.")

# =======================
# Main Execution
# =======================

if __name__ == "__main__":
    try:
        client = get_client()
        db = get_db(client)
        logger.info("Connected to MongoDB.")
        
        # Check if LLM is enabled
        enable_llm = os.getenv('ENABLE_LLM', 'False').lower() in ('true', '1', 't')
        if not enable_llm:
            logger.info("LLM integration is DISABLED. Exiting.")
            exit(0)
        else:
            logger.info("LLM integration is ENABLED. Proceeding with entity enhancement.")
        
        # Perform LLM-based entity enhancement
        enhance_entities_with_llm(db)
        
    except Exception as e:
        logger.error(f"An error occurred during LLM entity enhancement: {e}", exc_info=True)
