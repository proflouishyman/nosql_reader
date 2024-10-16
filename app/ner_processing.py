import os
import logging
from pymongo import UpdateOne, MongoClient
from dotenv import load_dotenv
import spacy
from tqdm import tqdm
import multiprocessing
import json
from collections import defaultdict
from rapidfuzz import process, fuzz

# Load environment variables from .env file
load_dotenv()

# =======================
# Logging Configuration
# =======================
logger = logging.getLogger('EntityProcessingLogger')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('entity_processing.log', mode='a')
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

def update_documents_as_processed(documents_collection, processed_doc_ids):
    """
    Batch update documents as processed.
    """
    operations = [
        UpdateOne({"_id": doc_id}, {"$set": {"entities_processed": True}})
        for doc_id in processed_doc_ids
    ]
    if operations:
        documents_collection.bulk_write(operations, ordered=False)
        logger.debug(f"Updated {len(operations)} documents as processed.")

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
# Main Processing Functions
# =======================

def process_documents_batch(args):
    """
    Process a batch of documents to extract entities.
    """
    batch_docs, fields_to_process, valid_entity_labels, existing_linked_terms, link_wikidata, fuzzy_threshold = args
    global nlp
    if nlp is None:
        nlp = initialize_spacy()
    processed_doc_ids = []
    aggregated_entities = defaultdict(entity_default)

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

    logger.debug(f"Processing batch of {len(texts)} documents.")
    # Process texts in batch using nlp.pipe()
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
            # Update the document with extracted entities
            documents_collection.update_one(
                {'_id': doc_id},
                {'$set': {'extracted_entities': entities}}
            )
            processed_doc_ids.append(doc_id)
    return processed_doc_ids, aggregated_entities

def extract_and_link_entities(documents_collection, linked_entities_collection, fields_to_process, link_wikidata, fuzzy_threshold=90, batch_size=1000, use_multiprocessing=False):
    """
    Extract entities from documents, perform entity linking, and store results.
    """
    # Extended entity types to consider
    valid_entity_labels = [
        'PERSON', 'ORG', 'GPE', 'LOC', 'EVENT',
        'FAC', 'PRODUCT', 'LANGUAGE', 'DATE', 'TIME',
        'MONEY'
    ]

    processed_count = 0

    # Prepare a list of already linked terms for fuzzy matching
    existing_linked_terms = linked_entities_collection.distinct("term")
    logger.debug(f"Fetched {len(existing_linked_terms)} existing linked terms for fuzzy matching.")

    # Query to fetch only unprocessed documents
    query = {"$or": [{"entities_processed": {"$exists": False}}, {"entities_processed": False}]}
    cursor = documents_collection.find(
        query,
        {'_id': 1, **{field: 1 for field in fields_to_process}}
    )

    total_documents = documents_collection.count_documents(query)
    logger.info(f"Total unprocessed documents to process: {total_documents}")

    if total_documents == 0:
        logger.info("No unprocessed documents found. Exiting.")
        return

    aggregated_entities = defaultdict(entity_default)

    if use_multiprocessing:
        logger.info("Multiprocessing is ENABLED.")
        # Prepare batches of documents
        batches = list(chunkify(cursor, batch_size))
        total_batches = len(batches)
        logger.info(f"Total batches to process: {total_batches}")

        # Prepare data for multiprocessing
        batch_args = [(batch, fields_to_process, valid_entity_labels, existing_linked_terms, link_wikidata, fuzzy_threshold) for batch in batches]

        num_processes = multiprocessing.cpu_count() - 1 or 1  # Reserve one core
        logger.info(f"Using {num_processes} worker processes for multiprocessing.")

        with multiprocessing.Pool(processes=num_processes, initializer=initialize_worker) as pool:
            results = tqdm(pool.imap_unordered(process_documents_batch, batch_args), total=total_batches, desc="Processing batches")
            for processed_doc_ids, batch_aggregated_entities in results:
                # Batch update documents as processed
                if processed_doc_ids:
                    update_documents_as_processed(documents_collection, processed_doc_ids)
                processed_count += len(processed_doc_ids)
                logger.debug(f"Processed {processed_count}/{total_documents} documents so far.")

                # Merge batch results into the main aggregated_entities
                for key, value in batch_aggregated_entities.items():
                    aggregated_entities[key]['frequency'] += value['frequency']
                    aggregated_entities[key]['document_ids'].update(value['document_ids'])
                    aggregated_entities[key]['type'] = value['type']
    else:
        logger.info("Multiprocessing is DISABLED.")
        global nlp
        nlp = initialize_spacy()  # Initialize spaCy once here
        batches = chunkify(cursor, batch_size)
        for batch_docs in tqdm(batches, desc="Processing batches"):
            processed_doc_ids, batch_aggregated_entities = process_documents_batch((batch_docs, fields_to_process, valid_entity_labels, existing_linked_terms, link_wikidata, fuzzy_threshold))
            # Batch update documents as processed
            if processed_doc_ids:
                update_documents_as_processed(documents_collection, processed_doc_ids)
            processed_count += len(processed_doc_ids)
            logger.debug(f"Processed {processed_count}/{total_documents} documents so far.")

            # Merge batch results into the main aggregated_entities
            for key, value in batch_aggregated_entities.items():
                aggregated_entities[key]['frequency'] += value['frequency']
                aggregated_entities[key]['document_ids'].update(value['document_ids'])
                aggregated_entities[key]['type'] = value['type']

    logger.info(f"Processed {processed_count} documents.")

    # Now, perform entity linking
    link_entities(aggregated_entities, linked_entities_collection, existing_linked_terms, link_wikidata, fuzzy_threshold, batch_size)

def link_entities(aggregated_entities, linked_entities_collection, existing_linked_terms, link_wikidata, fuzzy_threshold, batch_size):
    """
    Link aggregated entities and update the linked_entities collection.
    """
    linked_count = 0
    processed_entities = 0
    total_entities = len(aggregated_entities)
    logger.info(f"Total unique entities to process: {total_entities}")

    operations = []

    for (term_lower, ent_type), data in aggregated_entities.items():
        frequency = data['frequency']
        document_ids = list(data['document_ids'])
        wikidata_id = None

        if link_wikidata:
            wikidata_id = fetch_wikidata_entity(term_lower)

        # If Wikidata linking is disabled or Wikidata ID not found, perform fuzzy matching
        if not wikidata_id:
            fuzzy_match_term = fuzzy_match(term_lower, existing_linked_terms, threshold=fuzzy_threshold)
            if fuzzy_match_term:
                # Fetch the corresponding Wikidata ID for the matched term
                matched_entity = linked_entities_collection.find_one({"term": fuzzy_match_term.lower()})
                if matched_entity:
                    wikidata_id = matched_entity.get('kb_id')
                    logger.debug(f"Fuzzy matched term '{term_lower}' to '{fuzzy_match_term}' with Wikidata ID '{wikidata_id}'.")
            else:
                logger.debug(f"No match found for term '{term_lower}'.")
        else:
            logger.debug(f"Linked term '{term_lower}' to Wikidata ID '{wikidata_id}'.")

        update = {
            "$inc": {"frequency": frequency},
            "$addToSet": {"document_ids": {"$each": document_ids}},
            "$set": {"type": ent_type, "kb_id": wikidata_id}
        }

        operations.append(
            UpdateOne(
                {"term": term_lower},
                update,
                upsert=True
            )
        )

        linked_count += 1
        processed_entities += 1

        # Log progress every 1000 entities
        if processed_entities % 1000 == 0:
            logger.info(f"Processed {processed_entities}/{total_entities} entities.")

        # Execute operations in batches
        if len(operations) >= batch_size:
            logger.info(f"Writing batch of {len(operations)} entities to the database.")
            try:
                result = linked_entities_collection.bulk_write(operations, ordered=False)
                logger.info(f"Bulk upserted {result.upserted_count + result.modified_count} linked entities.")
            except Exception as e:
                logger.error(f"Error bulk upserting linked entities: {e}")
                raise e
            operations = []

    # Execute remaining operations
    if operations:
        logger.info(f"Writing final batch of {len(operations)} entities to the database.")
        try:
            result = linked_entities_collection.bulk_write(operations, ordered=False)
            logger.info(f"Bulk upserted {result.upserted_count + result.modified_count} linked entities.")
        except Exception as e:
            logger.error(f"Error bulk upserting linked entities: {e}")
            raise e

    logger.info(f"Successfully linked {linked_count} entities.")

# =======================
# Main Execution
# =======================
if __name__ == "__main__":
    try:
        client = get_client()
        db = get_db(client)
        logger.info("Connected to MongoDB.")

        documents_collection, linked_entities_collection = get_collections(db)

        # Determine if multiprocessing is enabled
        use_multiprocessing = os.getenv('ENABLE_MULTIPROCESSING', 'False').lower() in ('true', '1', 't')
        if use_multiprocessing:
            logger.info("Multiprocessing is ENABLED.")
        else:
            logger.info("Multiprocessing is DISABLED.")

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

        # Determine if Wikidata linking is enabled
        link_wikidata = os.getenv('LINK_WIKIDATA', 'False').lower() in ('true', '1', 't')
        if link_wikidata:
            logger.info("Wikidata linking is ENABLED.")
        else:
            logger.info("Wikidata linking is DISABLED. Focusing on fuzzy matching.")

        # Fetch configuration for fuzzy matching
        fuzzy_threshold = int(os.getenv('FUZZY_MATCH_THRESHOLD', 90))

        # Extract entities and perform linking
        extract_and_link_entities(
            documents_collection,
            linked_entities_collection,
            fields_to_process,
            link_wikidata,
            fuzzy_threshold=fuzzy_threshold,
            batch_size=batch_size,
            use_multiprocessing=use_multiprocessing
        )

    except Exception as e:
        logger.error(f"An error occurred during entity processing: {e}", exc_info=True)
