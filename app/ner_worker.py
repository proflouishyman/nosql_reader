# ner_worker.py

import os
import logging
from pymongo import UpdateOne, MongoClient
from dotenv import load_dotenv
import spacy
from collections import defaultdict
from rapidfuzz import process, fuzz
import requests

# Load environment variables from .env file
load_dotenv()

# =======================
# Logging Configuration
# =======================
logger = logging.getLogger('EntityProcessingWorker')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('entity_processing_worker.log', mode='a')
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
        documents_collection = db['documents']
        linked_entities_collection = db['linked_entities']
        return documents_collection, linked_entities_collection
    except Exception as e:
        logger.error(f"Error getting collections: {e}")
        raise e

# =======================
# spaCy Initialization
# =======================
def initialize_spacy():
    """Load spaCy model."""
    try:
        logger.info("Loading spaCy model with optimized settings.")
        # Load spaCy model with only the NER component
        nlp = spacy.load("en_core_web_lg", disable=["tagger", "parser", "lemmatizer"])
        return nlp
    except Exception as e:
        logger.error(f"Error loading spaCy model: {e}")
        raise e

# Global variable for spaCy model in worker processes
nlp = None

def initialize_worker():
    """Initialize spaCy model in worker processes."""
    global nlp
    nlp = initialize_spacy()

# =======================
# Utility Functions
# =======================

def entity_default():
    """Default factory function for defaultdict."""
    return {'frequency': 0, 'document_ids': set(), 'type': ''}

def fuzzy_match(term, reference_terms, threshold=90):
    """
    Perform fuzzy matching to find the best match for a term in reference_terms.
    Returns the matched term if similarity exceeds the threshold, else None.
    """
    try:
        result = process.extractOne(term, reference_terms, scorer=fuzz.token_sort_ratio)
        if result is None:
            logger.debug(f"No fuzzy match found for term '{term}'.")
            return None
        match, score = result
        logger.debug(f"Fuzzy match for term '{term}': '{match}' with score {score}.")
        if score >= threshold:
            return match
        return None
    except Exception as e:
        logger.error(f"Exception during fuzzy matching for term '{term}': {e}")
        return None

# In-memory cache for Wikidata entities
wikidata_cache = {}

def fetch_wikidata_entity(term):
    """
    Fetch Wikidata entity ID for a given term using the Wikidata API.
    Returns the entity ID if found, else None.
    """
    if term in wikidata_cache:
        return wikidata_cache[term]
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
            wikidata_id = data['search'][0]['id']
            wikidata_cache[term] = wikidata_id
            logger.debug(f"Fetched Wikidata ID '{wikidata_id}' for term '{term}'.")
            return wikidata_id
        else:
            wikidata_cache[term] = None
            logger.debug(f"No Wikidata ID found for term '{term}'.")
            return None
    except Exception as e:
        logger.error(f"Error fetching Wikidata entity for term '{term}': {e}")
        return None

# =======================
# Main Processing Function
# =======================

def process_documents_batch(args):
    """
    Process a batch of documents to extract entities.
    This function is intended to be run in a worker process.
    """
    batch_docs, fields_to_process, valid_entity_labels, existing_linked_terms, link_wikidata, fuzzy_threshold = args
    global nlp
    if nlp is None:
        initialize_worker()
    processed_doc_ids = []
    aggregated_entities = defaultdict(entity_default)

    # Initialize database connection in worker
    try:
        client = get_client()
        db = get_db(client)
        documents_collection, _ = get_collections(db)
    except Exception as e:
        logger.error(f"Worker failed to connect to MongoDB: {e}")
        return processed_doc_ids, aggregated_entities

    texts = []
    doc_id_mapping = []
    for doc in batch_docs:
        doc_id = doc['_id']
        combined_text = ''
        for field in fields_to_process:
            text = doc.get(field, '')
            if not text:
                continue
            if isinstance(text, list):
                text = ' '.join(text)
            elif not isinstance(text, str):
                continue
            combined_text += ' ' + text
        if combined_text.strip():
            texts.append(combined_text)
            doc_id_mapping.append(doc_id)
        else:
            logger.warning(f"Document '{doc_id}' has no text in specified fields.")

    if not texts:
        logger.warning("No texts found in the current batch to process.")
        return processed_doc_ids, aggregated_entities

    logger.debug(f"Processing batch of {len(texts)} documents.")
    # Process texts in batch using nlp.pipe()
    try:
        for doc_id, spacy_doc in zip(doc_id_mapping, nlp.pipe(texts, batch_size=100, n_process=1)):
            entities = []
            for ent in spacy_doc.ents:
                if ent.label_ in valid_entity_labels:
                    term = ent.text
                    ent_type = ent.label_
                    term_lower = term.lower()
                    entities.append({'text': term, 'type': ent_type})
                    logger.debug(f"Found entity '{term}' of type '{ent_type}' in document '{doc_id}'.")

                    # Aggregate entities for linking
                    aggregated_entities[(term_lower, ent_type)]['frequency'] += 1
                    aggregated_entities[(term_lower, ent_type)]['document_ids'].add(doc_id)
                    aggregated_entities[(term_lower, ent_type)]['type'] = ent_type

            if entities:
                # Update the document with extracted entities and mark as processed
                try:
                    result = documents_collection.update_one(
                        {'_id': doc_id, 'entities_processed': {'$ne': True}},
                        {'$set': {
                            'extracted_entities': entities,
                            'entities_processed': True  # Mark as processed here
                        }}
                    )
                    if result.modified_count > 0:
                        processed_doc_ids.append(doc_id)
                        logger.debug(f"Updated document '{doc_id}' with extracted entities and marked as processed.")
                    else:
                        logger.debug(f"Document '{doc_id}' was already processed by another worker.")
                except Exception as e:
                    logger.error(f"Failed to update document '{doc_id}': {e}")
            else:
                logger.info(f"No valid entities found in document '{doc_id}'. Marking as processed.")
                # Even if no entities found, mark as processed to avoid reprocessing
                try:
                    result = documents_collection.update_one(
                        {'_id': doc_id, 'entities_processed': {'$ne': True}},
                        {'$set': {'entities_processed': True}}
                    )
                    if result.modified_count > 0:
                        processed_doc_ids.append(doc_id)
                        logger.debug(f"Marked document '{doc_id}' as processed with no entities found.")
                    else:
                        logger.debug(f"Document '{doc_id}' was already processed by another worker.")
                except Exception as e:
                    logger.error(f"Failed to mark document '{doc_id}' as processed: {e}")
    except Exception as e:
        logger.error(f"Error during processing batch: {e}")

    return processed_doc_ids, aggregated_entities
