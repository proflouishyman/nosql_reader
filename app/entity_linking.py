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
import torch
import json  # For parsing JSON strings from .env
import multiprocessing

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

def chunkify(iterable, chunk_size):
    """
    Split an iterable into chunks of a specified size.
    """
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

# =======================
# Main Processing Functions
# =======================

def process_documents_batch(args):
    """
    Process a batch of documents to extract entities.
    """
    batch_docs, fields_to_process, valid_entity_labels = args
    nlp = initialize_spacy()
    batch_aggregated_entities = defaultdict(lambda: {'frequency': 0, 'document_ids': set()})
    processed_doc_ids = []

    for doc in batch_docs:
        doc_id = doc['_id']
        entities_found = False
        for field in fields_to_process:
            text = doc.get(field, '')
            if not text:
                continue

            # **Add this check to handle text being a list**
            if isinstance(text, list):
                text = ' '.join(text)
            elif not isinstance(text, str):
                # Skip if text is neither a string nor a list
                continue

            spacy_doc = nlp(text)

            for ent in spacy_doc.ents:
                if ent.label_ in valid_entity_labels:
                    term = ent.text
                    ent_type = ent.label_
                    term_lower = term.lower()

                    # Update frequency and document IDs
                    batch_aggregated_entities[(term_lower, ent_type)]['frequency'] += 1
                    batch_aggregated_entities[(term_lower, ent_type)]['document_ids'].add(doc_id)
                    entities_found = True

        if entities_found:
            processed_doc_ids.append(doc_id)

    return batch_aggregated_entities, processed_doc_ids


def link_entities(db, fields_to_process, enable_llm=False, fuzzy_threshold=90, batch_size=1000, use_multiprocessing=False):
    """
    Link entities from the documents collection to Wikidata.
    Incorporates spaCy for NER and RapidFuzz for fuzzy matching.
    Populates the linked_entities collection with the results.
    """
    documents_collection, linked_entities_collection = get_collections(db)

    # Extended entity types to consider
    valid_entity_labels = [
        'PERSON', 'ORG', 'GPE', 'LOC', 'EVENT', 'WORK_OF_ART',
        'FAC', 'PRODUCT', 'LAW', 'LANGUAGE', 'DATE', 'TIME',
        'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'
    ]

    # Prepare a list of already linked terms for fuzzy matching
    existing_linked_terms = linked_entities_collection.distinct("term")

    aggregated_entities = defaultdict(lambda: {'frequency': 0, 'document_ids': set()})
    processed_count = 0
    linked_count = 0

    # Modify the query to fetch only unprocessed documents
    cursor = documents_collection.find(
        {"$or": [{"entities_processed": {"$exists": False}}, {"entities_processed": False}]},
        {'_id': 1, **{field: 1 for field in fields_to_process}}
    )
    total_documents = cursor.count()
    logger.info(f"Total unprocessed documents to process: {total_documents}")

    if total_documents == 0:
        logger.info("No unprocessed documents found. Exiting.")
        return

    if use_multiprocessing:
        logger.info("Multiprocessing is ENABLED.")
        # Prepare batches of documents
        batches = list(chunkify(cursor, batch_size))
        total_batches = len(batches)
        logger.info(f"Total batches to process: {total_batches}")

        # Prepare data for multiprocessing
        batch_args = [(batch, fields_to_process, valid_entity_labels) for batch in batches]

        with multiprocessing.Pool() as pool:
            results = tqdm(pool.imap_unordered(process_documents_batch, batch_args), total=total_batches, desc="Processing batches")
            for batch_result in results:
                batch_aggregated_entities, processed_doc_ids = batch_result
                # Merge batch results into the main aggregated_entities
                for key, value in batch_aggregated_entities.items():
                    aggregated_entities[key]['frequency'] += value['frequency']
                    aggregated_entities[key]['document_ids'].update(value['document_ids'])
                # Update documents as processed
                if processed_doc_ids:
                    documents_collection.update_many(
                        {"_id": {"$in": processed_doc_ids}},
                        {"$set": {"entities_processed": True}}
                    )
                processed_count += len(processed_doc_ids)
    else:
        logger.info("Multiprocessing is DISABLED.")
        nlp = initialize_spacy()
        batches = chunkify(cursor, batch_size)
        for batch_docs in tqdm(batches, desc="Processing batches"):
            batch_aggregated_entities, processed_doc_ids = process_documents_batch((batch_docs, fields_to_process, valid_entity_labels))
            # Merge batch results into the main aggregated_entities
            for key, value in batch_aggregated_entities.items():
                aggregated_entities[key]['frequency'] += value['frequency']
                aggregated_entities[key]['document_ids'].update(value['document_ids'])
            # Update documents as processed
            if processed_doc_ids:
                documents_collection.update_many(
                    {"_id": {"$in": processed_doc_ids}},
                    {"$set": {"entities_processed": True}}
                )
            processed_count += len(processed_doc_ids)

    # Process aggregated entities
    batch = []
    for (term_lower, ent_type), data in aggregated_entities.items():
        frequency = data['frequency']
        document_ids = list(data['document_ids'])
        batch.append((term_lower, ent_type, frequency, document_ids))

        if len(batch) >= batch_size:
            results = process_entity_batch(batch, existing_linked_terms, enable_llm, fuzzy_threshold, linked_entities_collection)
            insert_linked_entities(results, linked_entities_collection)
            linked_count += len(results)
            batch = []

    # Process remaining batch
    if batch:
        results = process_entity_batch(batch, existing_linked_terms, enable_llm, fuzzy_threshold, linked_entities_collection)
        insert_linked_entities(results, linked_entities_collection)
        linked_count += len(results)

    logger.info(f"Processed {processed_count} documents.")
    logger.info(f"Successfully linked {linked_count} entities.")

def process_entity_batch(batch, existing_linked_terms, enable_llm, fuzzy_threshold, linked_entities_collection):
    """
    Process a batch of entities for linking.
    Returns a list of results with linked Wikidata IDs.
    """
    results = []
    for term, ent_type, frequency, document_ids in batch:
        wikidata_id = fetch_wikidata_entity(term)

        # If not found and LLM is enabled, attempt to use LLM for disambiguation
        if not wikidata_id and enable_llm:
            context = f"Entity: {term}, Type: {ent_type}"
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
            'type': ent_type,
            'frequency': frequency,
            'document_ids': document_ids,
            'wikidata_id': wikidata_id
        })
    return results

def insert_linked_entities(results, linked_entities_collection):
    """
    Insert or update linked entities in the database.
    """
    operations = []
    for result in results:
        term = result['term']
        ent_type = result['type']
        frequency = result['frequency']
        document_ids = result['document_ids']
        wikidata_id = result['wikidata_id']

        update = {
            "$inc": {"frequency": frequency},
            "$addToSet": {"document_ids": {"$each": document_ids}},
            "$set": {"type": ent_type, "kb_id": wikidata_id}
        }

        operations.append(
            UpdateOne(
                {"term": term.lower()},
                update,
                upsert=True
            )
        )

    if operations:
        try:
            result = linked_entities_collection.bulk_write(operations, ordered=False)
            logger.info(f"Bulk upserted {result.upserted_count + result.modified_count} linked entities.")
        except Exception as e:
            logger.error(f"Error bulk upserting linked entities: {e}")
            raise e
    else:
        logger.warning("No linked entities to upsert.")

# =======================
# Main Execution
# =======================
if __name__ == "__main__":
    try:
        client = get_client()
        db = get_db(client)
        logger.info("Connected to MongoDB.")

        documents_collection, linked_entities_collection = get_collections(db)

        # Determine if LLM integration is enabled
        enable_llm = os.getenv('ENABLE_LLM', 'False').lower() in ('true', '1', 't')
        if enable_llm:
            logger.info("LLM integration is ENABLED.")
        else:
            logger.info("LLM integration is DISABLED.")

        # Determine if multiprocessing is enabled
        use_multiprocessing = os.getenv('ENABLE_MULTIPROCESSING', 'False').lower() in ('true', '1', 't')
        if use_multiprocessing:
            logger.info("Multiprocessing is ENABLED.")
        else:
            logger.info("Multiprocessing is DISABLED.")

        # Fetch configuration for fuzzy matching
        fuzzy_threshold = int(os.getenv('FUZZY_MATCH_THRESHOLD', 90))
        batch_size = int(os.getenv('BATCH_SIZE', 1000))  # Set default batch_size to 1000

        # Get list of fields to process from .env or default to ['ocr_text']
        fields_env = os.getenv('FIELDS_TO_PROCESS', '["ocr_text"]')
        try:
            fields_to_process = json.loads(fields_env)
            if not isinstance(fields_to_process, list):
                raise ValueError
        except ValueError:
            logger.error("Invalid FIELDS_TO_PROCESS format. It should be a JSON array of field names.")
            raise

        logger.info(f"Fields to process: {fields_to_process}")

        # Perform entity linking
        link_entities(db, fields_to_process, enable_llm=enable_llm, fuzzy_threshold=fuzzy_threshold,
                      batch_size=batch_size, use_multiprocessing=use_multiprocessing)

    except Exception as e:
        logger.error(f"An error occurred during entity linking: {e}", exc_info=True)
