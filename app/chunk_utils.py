# chunk_utils.py

from pymongo import MongoClient
import logging

# Setup logger
logger = logging.getLogger('ChunkUtilsLogger')

def save_unique_terms(db, unique_terms_dict, max_chunk_size=1500000):
    """
    Save the unique terms dictionary to the database as multiple documents if it exceeds max size.
    :param db: Database instance
    :param unique_terms_dict: The dictionary containing unique terms
    :param max_chunk_size: The maximum size for each chunk in bytes
    """
    unique_terms_collection = db['unique_terms']
    
    current_chunk = {}
    current_size = 0
    chunk_index = 0

    for key, value in unique_terms_dict.items():
        new_size = len(key) + len(str(value))
        if current_size + new_size > max_chunk_size:
            unique_terms_collection.replace_one(
                {"_id": f"unique_terms_chunk_{chunk_index}"},
                {"terms": current_chunk},
                upsert=True
            )
            chunk_index += 1
            current_chunk = {}
            current_size = 0
        
        current_chunk[key] = value
        current_size += new_size

    if current_chunk:
        unique_terms_collection.replace_one(
            {"_id": f"unique_terms_chunk_{chunk_index}"},
            {"terms": current_chunk},
            upsert=True
        )

    logger.info("Unique terms saved to the database in chunks.")

def retrieve_unique_terms(db):
    """
    Retrieve all unique terms from the database, merging chunks into a single dictionary.
    :param db: Database instance
    :return: Merged unique terms dictionary
    """
    unique_terms_collection = db['unique_terms']
    terms = {}
    
    # Retrieve all unique term chunks
    for doc in unique_terms_collection.find():
        terms.update(doc.get('terms', {}))

    logger.info("Unique terms retrieved from the database.")
    return terms
