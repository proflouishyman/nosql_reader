# entity_linking.py

import os
import logging
from collections import defaultdict
from pymongo import UpdateOne, MongoClient
from dotenv import load_dotenv
import requests
import openai
from tqdm import tqdm
import spacy
from rapidfuzz import process, fuzz

# Load environment variables from .env file
load_dotenv()

# =======================
# Logging Configuration
# =======================
logger = logging.getLogger('EntityLinkingLogger')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    file_handler = logging.FileHandler('entity_linking.log', mode='a')
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
# spaCy Initialization
# =======================
# Initialize spaCy
def initialize_spacy():
    """Load spaCy model based on GPU availability."""
    try:
        if torch.cuda.is_available():
            logger.info("GPU detected. Loading transformer-based spaCy model.")
            nlp = spacy.load("en_core_web_trf")  # Transformer-based model
        else:
            logger.info("No GPU detected. Loading large spaCy model.")
            nlp = spacy.load("en_core_web_lg")   # Large model without transformer
        return nlp
    except Exception as e:
        logger.error(f"Error loading spaCy model: {e}")
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

def fuzzy_match(term, reference_terms, threshold=90):
    """
    Perform fuzzy matching to find the best match for a term in reference_terms.
    Returns the matched term if similarity exceeds the threshold, else None.
    """
    try:
        match, score = process.extractOne(term, reference_terms, scorer=fuzz.token_sort_ratio)
        if score >= threshold:
            return match
        return None
    except Exception as e:
        logger.error(f"Error during fuzzy matching for term '{term}': {e}")
        return None

# =======================
# Main Processing Function
# =======================
# Updated link_entities function with extended entity types
def link_entities(db, nlp, enable_llm=False, fuzzy_threshold=90, batch_size=1000):
    """
    Link entities from the unique_terms collection to Wikidata.
    Incorporates spaCy for enhanced NER and RapidFuzz for fuzzy matching.
    Populate the linked_entities collection with the results.
    """
    unique_terms_collection, linked_entities_collection = get_collections(db)
    
    # Extended entity types based on sample JSON
    entity_types = [
        'PERSON',
        'ORG',
        'GPE',
        'LOC',
        'EVENT',
        'WORK_OF_ART',
        'DIVISION',
        'LOCATION',
        'DATE',
        'MEDICAL'
    ]
    
    cursor = unique_terms_collection.find(
        {"type": {"$in": entity_types}},
        {"_id": 0, "term": 1, "field":1, "type":1, "frequency":1}
    )
    
    # Prepare a list of already linked terms for fuzzy matching
    existing_linked_terms = linked_entities_collection.distinct("term")
    
    aggregated_entities = defaultdict(int)
    processed_count = 0
    linked_count = 0
    fuzzy_matched = 0

    # Batch processing
    batch = []
    for term_doc in tqdm(cursor, desc="Linking entities"):
        term = term_doc.get('term')
        field = term_doc.get('field')
        ent_type = term_doc.get('type')
        frequency = term_doc.get('frequency', 1)
        
        if not term:
            logger.warning("Encountered a term with no value. Skipping.")
            continue
        
        batch.append((term, field, ent_type, frequency))
        
        if len(batch) >= batch_size:
            results = process_batch(batch, existing_linked_terms, nlp, enable_llm, fuzzy_threshold, linked_entities_collection)
            for result in results:
                if result['wikidata_id']:
                    key = (result['term'].lower(), result['field'], result['type'], result['wikidata_id'])
                    aggregated_entities[key] += result['frequency']
                    linked_count += 1
                elif result['fuzzy_match']:
                    key = (result['term'].lower(), result['field'], result['type'], result['fuzzy_matched_id'])
                    aggregated_entities[key] += result['frequency']
                    fuzzy_matched += 1
            processed_count += len(batch)
            batch = []
    
    # Process remaining batch
    if batch:
        results = process_batch(batch, existing_linked_terms, nlp, enable_llm, fuzzy_threshold, linked_entities_collection)
        for result in results:
            if result['wikidata_id']:
                key = (result['term'].lower(), result['field'], result['type'], result['wikidata_id'])
                aggregated_entities[key] += result['frequency']
                linked_count += 1
            elif result['fuzzy_match']:
                key = (result['term'].lower(), result['field'], result['type'], result['fuzzy_matched_id'])
                aggregated_entities[key] += result['frequency']
                fuzzy_matched += 1
        processed_count += len(batch)
    
    logger.debug(f"Aggregated linked entities: {aggregated_entities}")
    logger.info(f"Processed {processed_count} terms.")
    logger.info(f"Successfully linked {linked_count} entities via API/LLM.")
    logger.info(f"Successfully linked {fuzzy_matched} entities via Fuzzy Matching.")
    
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
    
    logger.info("Entity linking process completed.")

# process_batch function updated to include entity validation with spaCy
def process_batch(batch, existing_linked_terms, nlp, enable_llm, fuzzy_threshold, linked_entities_collection):
    """
    Process a batch of terms for entity linking.
    Returns a list of results with linked Wikidata IDs or fuzzy matched IDs.
    """
    results = []
    for term, field, ent_type, frequency in batch:
        # Use spaCy to validate the entity
        doc = nlp(term)
        valid_entity = False
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'EVENT', 'WORK_OF_ART', 'DIVISION', 'LOCATION', 'MEDICAL']:
                valid_entity = True
                break
        if not valid_entity:
            logger.debug(f"Term '{term}' does not match any valid entity types. Skipping.")
            continue
        
        wikidata_id = fetch_wikidata_entity(term)
        
        # If not found and LLM is enabled, attempt to use LLM for disambiguation
        if not wikidata_id and enable_llm:
            context = f"Term: {term}, Field: {field}"
            wikidata_id = link_entity_with_llm(term, context)
        
        # If still not found, perform fuzzy matching
        if not wikidata_id:
            fuzzy_match_term = fuzzy_match(term, existing_linked_terms, threshold=fuzzy_threshold)
            if fuzzy_match_term:
                # Fetch the corresponding Wikidata ID for the matched term
                matched_entity = linked_entities_collection.find_one({"term": fuzzy_match_term.lower()})
                if matched_entity:
                    wikidata_id = matched_entity.get('kb_id')
                    results.append({
                        'term': term,
                        'field': field,
                        'type': ent_type,
                        'frequency': frequency,
                        'wikidata_id': None,
                        'fuzzy_match': True,
                        'fuzzy_matched_id': wikidata_id
                    })
                    continue
        
        results.append({
            'term': term,
            'field': field,
            'type': ent_type,
            'frequency': frequency,
            'wikidata_id': wikidata_id,
            'fuzzy_match': False,
            'fuzzy_matched_id': None
        })
    return results

# =======================
# Main Execution
# =======================

# Sample usage within the main execution block
if __name__ == "__main__":
    try:
        client = get_client()
        db = get_db(client)
        logger.info("Connected to MongoDB.")
        
        unique_terms_collection, linked_entities_collection = get_collections(db)
        
        # Initialize spaCy
        nlp = initialize_spacy()
        
        # Determine if LLM integration is enabled
        enable_llm = os.getenv('ENABLE_LLM', 'False').lower() in ('true', '1', 't')
        if enable_llm:
            logger.info("LLM integration is ENABLED.")
        else:
            logger.info("LLM integration is DISABLED.")
        
        # Fetch configuration for fuzzy matching
        fuzzy_threshold = int(os.getenv('FUZZY_MATCH_THRESHOLD', 90))
        batch_size = int(os.getenv('BATCH_SIZE', 1000))
        
        # Perform entity linking with spaCy and fuzzy matching
        link_entities(db, nlp, enable_llm=enable_llm, fuzzy_threshold=fuzzy_threshold, batch_size=batch_size)
        
    except Exception as e:
        logger.error(f"An error occurred during entity linking: {e}", exc_info=True)
