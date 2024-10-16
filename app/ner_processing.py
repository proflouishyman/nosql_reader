import os
import logging
from pymongo import UpdateOne, MongoClient
from dotenv import load_dotenv
import spacy
from tqdm import tqdm
import multiprocessing
import json

# Load environment variables from .env file
load_dotenv()

# =======================
# Logging Configuration
# =======================
logger = logging.getLogger('NERProcessingLogger')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler('ner_processing.log', mode='a')
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

def get_documents_collection(db):
    """Retrieve and return reference to the documents collection."""
    try:
        documents_collection = db['documents']
        return documents_collection
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

# =======================
# Main Processing Functions
# =======================

def process_documents_batch(args):
    """
    Process a batch of documents to extract entities.
    """
    batch_docs, fields_to_process, valid_entity_labels = args
    global nlp
    if nlp is None:
        nlp = initialize_spacy()
    processed_doc_ids = []

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
                entities.append({'text': term, 'type': ent_type})
                logger.debug(f"Found entity '{term}' of type '{ent_type}' in document '{doc_id}'.")
        if entities:
            # Update the document with extracted entities
            documents_collection.update_one(
                {'_id': doc_id},
                {'$set': {'extracted_entities': entities}}
            )
            processed_doc_ids.append(doc_id)
    return processed_doc_ids

def extract_entities(documents_collection, fields_to_process, batch_size=1000, use_multiprocessing=False):
    """
    Extract entities from documents and store them in the 'extracted_entities' field of the document.
    """
    # Extended entity types to consider
    valid_entity_labels = [
        'PERSON', 'ORG', 'GPE', 'LOC', 'EVENT', 'WORK_OF_ART',
        'FAC', 'PRODUCT', 'LAW', 'LANGUAGE', 'DATE', 'TIME',
        'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'
    ]

    processed_count = 0

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

    if use_multiprocessing:
        logger.info("Multiprocessing is ENABLED.")
        # Prepare batches of documents
        batches = list(chunkify(cursor, batch_size))
        total_batches = len(batches)
        logger.info(f"Total batches to process: {total_batches}")

        # Prepare data for multiprocessing
        batch_args = [(batch, fields_to_process, valid_entity_labels) for batch in batches]

        num_processes = multiprocessing.cpu_count() - 1 or 1  # Reserve one core
        logger.info(f"Using {num_processes} worker processes for multiprocessing.")

        with multiprocessing.Pool(processes=num_processes, initializer=initialize_worker) as pool:
            results = tqdm(pool.imap_unordered(process_documents_batch, batch_args), total=total_batches, desc="Processing batches")
            for processed_doc_ids in results:
                # Batch update documents as processed
                if processed_doc_ids:
                    update_documents_as_processed(documents_collection, processed_doc_ids)
                processed_count += len(processed_doc_ids)
                logger.debug(f"Processed {processed_count}/{total_documents} documents so far.")
    else:
        logger.info("Multiprocessing is DISABLED.")
        global nlp
        nlp = initialize_spacy()  # Initialize spaCy once here
        batches = chunkify(cursor, batch_size)
        for batch_docs in tqdm(batches, desc="Processing batches"):
            processed_doc_ids = process_documents_batch((batch_docs, fields_to_process, valid_entity_labels))
            # Batch update documents as processed
            if processed_doc_ids:
                update_documents_as_processed(documents_collection, processed_doc_ids)
            processed_count += len(processed_doc_ids)
            logger.debug(f"Processed {processed_count}/{total_documents} documents so far.")

    logger.info(f"Processed {processed_count} documents.")

# =======================
# Main Execution
# =======================
if __name__ == "__main__":
    try:
        client = get_client()
        db = get_db(client)
        logger.info("Connected to MongoDB.")

        documents_collection = get_documents_collection(db)

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

        # Extract entities
        extract_entities(documents_collection, fields_to_process, batch_size=batch_size, use_multiprocessing=use_multiprocessing)

    except Exception as e:
        logger.error(f"An error occurred during NER processing: {e}", exc_info=True)
