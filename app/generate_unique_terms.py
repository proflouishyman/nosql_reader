# generate_unique_terms.py

import os
import re
import logging
from collections import Counter
from tqdm import tqdm

from pymongo import UpdateOne, MongoClient

# =======================
# Logging Configuration
# =======================
logger = logging.getLogger('GenerateUniqueTermsLogger')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    file_handler = logging.FileHandler('generate_unique_terms.log', mode='a')
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# =======================
# Database Configuration
# =======================

# Load environment variables if using a .env file
from dotenv import load_dotenv
load_dotenv()

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
    """Check if the field value is a string."""
    return isinstance(value, str)

# =======================
# Main Processing Function
# =======================

def process_document(document):
    """Extract and count unique terms from a single document by iterating over all string fields."""
    unique_terms = {}
    
    for field, value in document.items():
        if is_text_field(value):
            terms = collect_unique_terms_from_text(value)
            if terms['word'] or terms['phrase']:
                unique_terms.setdefault(field, {'word': Counter(), 'phrase': Counter()})
                merge_counters(unique_terms[field], terms)
    
    return unique_terms

def generate_unique_terms(db):
    """Generate unique terms from documents and populate the unique_terms collection."""
    documents_collection, unique_terms_collection = get_collections(db)
    
    # Filter: Only process documents that haven't been processed yet
    cursor = documents_collection.find(
        {"unique_terms_processed": {"$ne": True}},
        {'_id': 0}  # Exclude _id for efficiency
    )
    
    aggregated_unique = {}
    processed_count = 0
    
    logger.info("Starting unique terms generation.")
    
    for doc in tqdm(cursor, desc="Processing documents"):
        doc_unique = process_document(doc)
        for field, terms in doc_unique.items():
            if field not in aggregated_unique:
                aggregated_unique[field] = {'word': Counter(), 'phrase': Counter()}
            merge_counters(aggregated_unique[field], terms)
        
        # Mark the document as processed
        documents_collection.update_one(
            {"_id": doc.get('_id')},
            {"$set": {"unique_terms_processed": True}}
        )
        processed_count += 1
    
    logger.debug(f"Aggregated unique terms: {aggregated_unique}")
    logger.info(f"Processed {processed_count} documents.")
    
    # Prepare bulk operations
    operations = []
    for field, types in aggregated_unique.items():
        for term_type, counter in types.items():
            for term, freq in counter.items():
                operations.append(
                    UpdateOne(
                        {"term": term, "field": field, "type": term_type},
                        {"$inc": {"frequency": freq}},
                        upsert=True
                    )
                )
    
    logger.debug(f"Prepared {len(operations)} bulk operations for unique_terms.")
    
    if operations:
        try:
            result = unique_terms_collection.bulk_write(operations, ordered=False)
            logger.info(f"Bulk upserted {result.upserted_count + result.modified_count} unique terms.")
        except Exception as e:
            logger.error(f"Error bulk upserting unique terms: {e}")
            raise e
    else:
        logger.warning("No unique terms to upsert.")
    
    logger.info("Unique terms generation completed.")

# =======================
# Main Execution
# =======================
if __name__ == "__main__":
    try:
        client = get_client()
        db = get_db(client)
        logger.info("Connected to MongoDB.")
        
        generate_unique_terms(db)
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
