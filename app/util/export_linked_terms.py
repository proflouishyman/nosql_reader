import os
import logging
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =======================
# Logging Configuration
# =======================
logger = logging.getLogger('ExportLinkedTermsLogger')
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    file_handler = logging.FileHandler('export_linked_terms.log', mode='a')
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
    linked_entities_collection = db['linked_entities']
    return linked_entities_collection

# =======================
# Export Function
# =======================
def export_linked_terms(db):
    """Export linked terms to CSV files."""
    linked_entities_collection = get_collections(db)
    
    # Fetch all linked entities
    linked_entities = list(linked_entities_collection.find({}, {"term": 1, "kb_id": 1, "frequency": 1, "type": 1}))
    
    # Check if there are linked entities to export
    if not linked_entities:
        logger.info("No linked terms found to export.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(linked_entities)
    
    # Clean up the DataFrame if needed
    df['term'] = df['term'].str.lower()  # Standardize term to lowercase

    # Export the entire dataset as the first table
    df.to_csv('linked_terms_export_all.csv', index=False)
    logger.info("Exported all linked terms to linked_terms_export_all.csv")

    # Group by 'type' and export each group to a separate CSV file
    for entity_type, group in df.groupby('type'):
        output_file = f'linked_terms_export_{entity_type}.csv'
        group.to_csv(output_file, index=False)
        logger.info(f"Exported linked terms of type '{entity_type}' to {output_file}")

# =======================
# Main Execution
# =======================
if __name__ == "__main__":
    try:
        client = get_client()
        db = get_db(client)
        logger.info("Connected to MongoDB.")

        export_linked_terms(db)

    except Exception as e:
        logger.error(f"An error occurred during export: {e}", exc_info=True)
