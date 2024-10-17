# generate_unique_terms.py

import os
import re
import logging
from collections import Counter, defaultdict
from tqdm import tqdm

from pymongo import UpdateOne, MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =======================
# Logging Configuration
# =======================
logger = logging.getLogger('GenerateUniqueTermsLogger')
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    file_handler = logging.FileHandler('generate_unique_terms.log', mode='a')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# =======================
# Database Configuration
# =======================

def get_client():
    mongo_uri = os.getenv('MONGO_URI', 'mongodb://admin:secret@mongodbt:27017/admin')
    return MongoClient(mongo_uri)

def get_db(client):
    return client['railroad_documents']

def get_collections(db):
    documents = db['documents']
    unique_terms_collection = db['unique_terms']
    return documents, unique_terms_collection

# =======================
# Utility Functions
# =======================

def collect_unique_terms_from_text(text):
    """Collect unique words and phrases from a text string."""
    unique_terms = {'word': Counter(), 'phrase': Counter()}
    if isinstance(text, str):
        words = re.findall(r'\w+', text.lower())
        phrases = [' '.join(pair) for pair in zip(words, words[1:])]
        unique_terms['word'].update(words)
        unique_terms['phrase'].update(phrases)
    return unique_terms

def merge_counters(main_counter, new_counter):
    """Merge two unique terms dictionaries."""
    for term_type in ['word', 'phrase']:
        main_counter[term_type].update(new_counter.get(term_type, Counter()))

def is_text_field(value):
    """Check if the field value is a string or a list of strings."""
    if isinstance(value, str):
        return True
    elif isinstance(value, list):
        # Check if all elements in the list are strings
        return all(isinstance(item, str) for item in value)
    return False

def chunkify(iterable, chunk_size):
    """Split an iterable into chunks of a specified size."""
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

def process_documents_batch(batch_docs):
    """Process a batch of documents to extract unique terms."""
    unique_terms = defaultdict(lambda: {'word': Counter(), 'phrase': Counter()})
    processed_doc_ids = []

    for doc in batch_docs:
        doc_id = doc['_id']
        doc_unique = process_document(doc)
        for field, terms in doc_unique.items():
            merge_counters(unique_terms[field], terms)
        processed_doc_ids.append(doc_id)

    # Establish a new MongoDB client for updating
    client = get_client()
    db = get_db(client)
    documents_collection, _ = get_collections(db)

    # Mark documents as processed
    if processed_doc_ids:
        documents_collection.update_many(
            {"_id": {"$in": processed_doc_ids}},
            {"$set": {"unique_terms_processed": True}}
        )

    client.close()
    return unique_terms

def process_document(document):
    """Extract and count unique terms from a single document by iterating over all text fields."""
    unique_terms = defaultdict(lambda: {'word': Counter(), 'phrase': Counter()})

    for field, value in document.items():
        if is_text_field(value):
            if isinstance(value, list):
                text = ' '.join(value)
            else:
                text = value
            terms = collect_unique_terms_from_text(text)
            if terms['word'] or terms['phrase']:
                merge_counters(unique_terms[field], terms)

    return unique_terms

def generate_unique_terms(db, batch_size=1000):
    """Generate unique terms from documents and populate the unique_terms collection."""
    documents_collection, unique_terms_collection = get_collections(db)

    # Filter: Only process documents that haven't been processed yet
    query = {"$or": [{"unique_terms_processed": {"$exists": False}}, {"unique_terms_processed": False}]}
    cursor = documents_collection.find(query)

    # Use count_documents() instead of cursor.count()
    total_documents = documents_collection.count_documents(query)
    logger.info(f"Total unprocessed documents: {total_documents}")

    if total_documents == 0:
        logger.info("No unprocessed documents found. Exiting.")
        return

    processed_count = 0
    batches = chunkify(cursor, batch_size)
    total_batches = (total_documents // batch_size) + (1 if total_documents % batch_size else 0)
    logger.info(f"Total batches to process: {total_batches}")

    # Load existing terms from unique_terms_collection
    existing_terms = {}
    for doc in unique_terms_collection.find({}, {"term": 1, "frequency": 1, "field": 1, "type": 1}):
        key = (doc['term'], doc['field'], doc['type'])
        existing_terms[key] = doc['frequency']

    for batch_docs in tqdm(batches, total=total_batches, desc="Processing batches"):
        batch_unique_terms = process_documents_batch(batch_docs)
        # Prepare bulk operations based on new terms and frequency differences
        operations = []
        
        for field, types in batch_unique_terms.items():
            for term_type, counter in types.items():
                for term, freq in counter.items():
                    key = (term, field, term_type)
                    existing_freq = existing_terms.get(key, 0)
                    freq_diff = freq - existing_freq
                    
                    if freq_diff > 0:  # Only upsert if there is a difference
                        operations.append(
                            UpdateOne(
                                {"term": term, "field": field, "type": term_type},
                                {"$inc": {"frequency": freq_diff}},
                                upsert=True
                            )
                        )
                        # Update existing_terms to avoid repeated increments
                        existing_terms[key] = freq

        processed_count += len(batch_docs)
        logger.info(f"Processed {processed_count} documents.")
        
        if operations:
            try:
                result = unique_terms_collection.bulk_write(operations, ordered=False)
                logger.info(f"Bulk upserted {result.upserted_count + result.modified_count} unique terms.")
            except Exception as e:
                logger.error(f"Error bulk upserting unique terms: {e}")
                raise e
        else:
            logger.warning("No unique terms to upsert in this batch.")

    logger.info("Unique terms generation completed.")

# =======================
# Main Execution
# =======================
if __name__ == "__main__":
    try:
        client = get_client()
        db = get_db(client)
        logger.info("Connected to MongoDB.")

        # Set multiprocessing to False
        use_multiprocessing = False
        logger.info("Multiprocessing is DISABLED.")

        # Fetch batch size from .env or use default
        batch_size = int(os.getenv('BATCH_SIZE', 1000))

        generate_unique_terms(db, batch_size=batch_size)

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
