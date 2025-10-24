# export_unique_terms.py

import sys
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Allow direct execution by ensuring the project root is importable.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Reuse the shared database helpers so the URI handling stays consistent.
# ---------------------------------------------------------------------------
from app.database_setup import get_client, get_db

# Load environment variables from .env file
load_dotenv()

# =======================
# Logging Configuration
# =======================
logger = logging.getLogger('ExportUniqueTermsLogger')
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler('export_unique_terms.log', mode='a')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def get_unique_terms_collection(db):
    return db['unique_terms']

# =======================
# Export Function
# =======================

def export_unique_terms(output_format='csv', output_path='unique_terms_export'):
    """
    Export unique terms from MongoDB to a CSV or JSON file.

    Parameters:
    - output_format (str): 'csv' or 'json'.
    - output_path (str): Base path for the output file without extension.
    """
    client = get_client()
    db = get_db(client)
    unique_terms_collection = get_unique_terms_collection(db)

    logger.info("Connected to MongoDB.")

    try:
        # Fetch all unique terms
        cursor = unique_terms_collection.find({})
        logger.info("Fetched unique terms from the database.")

        # Convert cursor to list of dictionaries
        terms_list = list(cursor)

        if not terms_list:
            logger.warning("No unique terms found in the collection.")
            return

        # Create a DataFrame for better handling
        df = pd.DataFrame(terms_list)

        # Remove the MongoDB '_id' field if present
        if '_id' in df.columns:
            df.drop(columns=['_id'], inplace=True)

        # Define output file paths
        if output_format.lower() == 'csv':
            file_path = f"{output_path}.csv"
            df.to_csv(file_path, index=False)
        elif output_format.lower() == 'json':
            file_path = f"{output_path}.json"
            df.to_json(file_path, orient='records', lines=True)
        else:
            logger.error("Unsupported output format. Please choose 'csv' or 'json'.")
            return

        logger.info(f"Unique terms exported successfully to {file_path}.")

    except Exception as e:
        logger.error(f"An error occurred during export: {e}", exc_info=True)
    finally:
        client.close()
        logger.info("MongoDB connection closed.")

# =======================
# Main Execution
# =======================
if __name__ == "__main__":
    import argparse

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Export unique terms from MongoDB to CSV or JSON.")
    parser.add_argument(
        '--format',
        type=str,
        choices=['csv', 'json'],
        default='csv',
        help="Output file format: 'csv' or 'json'. Default is 'csv'."
    )
    parser.add_argument(
        '--output',
        type=str,
        default='unique_terms_export',
        help="Base path for the output file without extension. Default is 'unique_terms_export'."
    )

    args = parser.parse_args()

    # Call the export function with provided arguments
    export_unique_terms(output_format=args.format, output_path=args.output)
