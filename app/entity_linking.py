import os
import logging
from collections import defaultdict
from pymongo import UpdateOne, MongoClient
from dotenv import load_dotenv
import requests
import openai
from tqdm import tqdm
from rapidfuzz import process, fuzz
import uuid
import pymongo

# Load environment variables from .env file
load_dotenv()

# =======================
# Logging Configuration
# =======================
logger = logging.getLogger('EntityLinkingLogger')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

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
# Utility Functions
# =======================

def generate_custom_kb_id():
    """Generate a unique custom kb_id."""
    return f"CUSTOM_{uuid.uuid4()}"

def entity_default():
    """Default factory function for defaultdict."""
    return {'frequency': 0, 'document_ids': set(), 'type': ''}

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
            logger.debug(f"LLM did not find a Wikidata ID for term '{term}'.")
            return None
        logger.debug(f"LLM linked term '{term}' to Wikidata ID '{wikidata_id}'.")
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
        result = process.extractOne(term, reference_terms, scorer=fuzz.token_sort_ratio)
        if result is None:
            logger.debug(f"No fuzzy match found for term '{term}'.")
            return None

        # Unpack only the first two elements: match and score
        match, score, *_ = result  # The *_ captures any additional elements

        logger.debug(f"Fuzzy match for term '{term}': '{match}' with score {score}.")

        if score >= threshold:
            return match
        return None
    except Exception as e:
        logger.error(f"Exception during fuzzy matching for term '{term}': {e}")
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

def aggregate_entities(documents_collection, batch_size=1000):
    """
    Aggregate entities from the 'extracted_entities' field of documents.
    """
    # Query to fetch documents that have 'extracted_entities' and have not been processed
    query = {'extracted_entities': {'$exists': True, '$ne': []}, 'entity_linking_processed': {'$ne': True}}

    cursor = documents_collection.find(query, {'_id': 1, 'extracted_entities': 1})

    total_documents = documents_collection.count_documents(query)
    logger.info(f"Total documents with extracted entities to process: {total_documents}")

    aggregated_entities = defaultdict(entity_default)

    batches = chunkify(cursor, batch_size)
    for batch_docs in tqdm(batches, desc="Aggregating entities"):
        for doc in batch_docs:
            doc_id = doc['_id']
            entities = doc.get('extracted_entities', [])
            for entity in entities:
                term = entity['text']
                ent_type = entity['type']
                term_lower = term.lower()

                aggregated_entities[(term_lower, ent_type)]['frequency'] += 1
                aggregated_entities[(term_lower, ent_type)]['document_ids'].add(doc_id)
                aggregated_entities[(term_lower, ent_type)]['type'] = ent_type

    return aggregated_entities

def link_entities(aggregated_entities, linked_entities_collection, existing_linked_terms, link_wikidata, fuzzy_threshold, batch_size, documents_collection):
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
                logger.debug(f"No fuzzy match found for term '{term_lower}'.")

        # If still no wikidata_id, automate custom entry creation
        if not wikidata_id:
            # Generate a unique kb_id for the custom entry
            custom_kb_id = generate_custom_kb_id()
            try:
                # Insert the new linked entity
                linked_entities_collection.insert_one({
                    "term": term_lower,
                    "kb_id": custom_kb_id,
                    "frequency": frequency,
                    "document_ids": document_ids,
                    "type": ent_type
                })
                wikidata_id = custom_kb_id
                logger.debug(f"Inserted custom linked entity for term '{term_lower}' with kb_id '{custom_kb_id}'.")
            except pymongo.errors.DuplicateKeyError:
                # If the term was inserted by another process/thread, update it instead
                linked_entities_collection.update_one(
                    {"term": term_lower},
                    {
                        "$inc": {"frequency": frequency},
                        "$addToSet": {"document_ids": {"$each": document_ids}},
                        "$set": {"type": ent_type}
                    }
                )
                matched_entity = linked_entities_collection.find_one({"term": term_lower})
                wikidata_id = matched_entity.get('kb_id')
                logger.debug(f"Updated existing custom linked entity for term '{term_lower}' with kb_id '{wikidata_id}'.")

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

    # Mark documents as processed
    document_ids_to_mark = set()
    for (term_lower, ent_type), data in aggregated_entities.items():
        document_ids_to_mark.update(data['document_ids'])

    if document_ids_to_mark:
        try:
            documents_collection.update_many(
                {"_id": {"$in": list(document_ids_to_mark)}},
                {"$set": {"entity_linking_processed": True}}
            )
            logger.info(f"Marked {len(document_ids_to_mark)} documents as processed.")
        except Exception as e:
            logger.error(f"Error marking documents as processed: {e}")
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

        # Determine if LLM integration is enabled
        enable_llm = os.getenv('ENABLE_LLM', 'False').lower() in ('true', '1', 't')
        if enable_llm:
            logger.info("LLM integration is ENABLED.")
        else:
            logger.info("LLM integration is DISABLED.")

        fuzzy_threshold = int(os.getenv('FUZZY_MATCH_THRESHOLD', 90))
        batch_size = int(os.getenv('BATCH_SIZE', 1000))  # Set default batch_size to 1000

        # Aggregate entities from unprocessed documents
        aggregated_entities = aggregate_entities(documents_collection, batch_size=batch_size)

        if not aggregated_entities:
            logger.info("No new entities to process.")
            exit(0)

        # Fetch existing linked terms for fuzzy matching
        existing_linked_terms = [doc['term'] for doc in linked_entities_collection.find({}, {'term': 1})]

        # Perform entity linking
        link_entities(
            aggregated_entities,
            linked_entities_collection,
            existing_linked_terms,
            link_wikidata=enable_llm,
            fuzzy_threshold=fuzzy_threshold,
            batch_size=batch_size,
            documents_collection=documents_collection
        )

    except Exception as e:
        logger.error(f"An error occurred during entity linking: {e}", exc_info=True)
