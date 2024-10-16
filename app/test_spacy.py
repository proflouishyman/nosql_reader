# Script: test_spacy_on_db_entries.py
# Purpose: Connects to MongoDB, retrieves sample documents, and tests spaCy entity recognition on 'ocr_text' field.
# Created: 2024-10-16

import os
from pymongo import MongoClient
from dotenv import load_dotenv
import spacy
import logging

# Load environment variables from .env file
load_dotenv()

# Logging configuration to log to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_spacy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TestSpacy')

def get_client():
    """Initialize and return a MongoDB client."""
    mongo_uri = os.getenv('MONGO_URI')
    if not mongo_uri:
        logger.error("MONGO_URI environment variable not set.")
        return None
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')  # Test connection
        logger.info("Successfully connected to MongoDB.")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        return None

def get_sample_documents(db, num_samples=5):
    """Retrieve a few sample documents with the 'ocr_text' field."""
    try:
        documents_collection = db['documents']
        sample_docs = documents_collection.find(
            {"ocr_text": {"$exists": True, "$ne": ""}},
            {"ocr_text": 1, "_id": 1}
        ).limit(num_samples)
        return list(sample_docs)
    except Exception as e:
        logger.error(f"Error retrieving sample documents: {e}")
        return []

def test_spacy_on_documents(nlp, documents):
    """Process the 'ocr_text' field of each document with spaCy and log recognized entities."""
    relevant_labels = {'PERSON', 'ORG', 'GPE', 'LOC'}
    texts = [doc.get("ocr_text", "") for doc in documents]
    doc_ids = [doc['_id'] for doc in documents]
    
    # Use nlp.pipe for efficient batch processing
    for doc_id, spacy_doc in zip(doc_ids, nlp.pipe(texts, batch_size=50)):
        logger.info(f"Document ID: {doc_id}")
        # Filter entities by relevant types
        filtered_entities = [ent for ent in spacy_doc.ents if ent.label_ in relevant_labels]
        if filtered_entities:
            for ent in filtered_entities:
                logger.info(f"Entity: {ent.text}, Label: {ent.label_}")
        else:
            logger.info("No relevant entities found.")

def main():
    # Initialize spaCy with only NER and disable other components for speed
    try:
        logger.info("Loading spaCy model...")
        nlp = spacy.load("en_core_web_lg", disable=["parser", "tagger", "lemmatizer"])
    except Exception as e:
        logger.error(f"Failed to load spaCy model: {e}")
        return

    # Connect to MongoDB
    client = get_client()
    if not client:
        return

    db = client['railroad_documents']

    # Retrieve sample documents
    documents = get_sample_documents(db, num_samples=5)
    if not documents:
        logger.error("No sample documents retrieved.")
        return

    # Test spaCy on retrieved documents
    test_spacy_on_documents(nlp, documents)

if __name__ == "__main__":
    main()
