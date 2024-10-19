#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Configuration for backoff
MAX_RETRIES=10
SLEEP_TIME=10
MONGO_HOST="mongodb"
MONGO_PORT=27017
MONGO_URI="mongodb://admin:secret@${MONGO_HOST}:${MONGO_PORT}/admin"

echo "Waiting for MongoDB to be ready..."

# Backoff loop to wait for MongoDB
for ((i=1;i<=MAX_RETRIES;i++)); do
    echo "Attempt $i/$MAX_RETRIES: Checking MongoDB connection..."
    python -c "import pymongo; client = pymongo.MongoClient('${MONGO_URI}', serverSelectionTimeoutMS=5000); client.admin.command('ping')" && break
    echo "MongoDB is not ready yet. Waiting ${SLEEP_TIME} seconds..."
    sleep ${SLEEP_TIME}
done

# Verify if MongoDB is up after retries
python -c "import pymongo; client = pymongo.MongoClient('${MONGO_URI}', serverSelectionTimeoutMS=5000); client.admin.command('ping')" || { echo "MongoDB did not become ready in time after ${MAX_RETRIES} attempts. Exiting."; exit 1; }

echo "MongoDB is up and running."

# echo "Running database setup scripts..."

# echo "Environmental variables"
# python show_env.py




# echo "Running database_setup.py..."
# python database_setup.py
# echo "database_setup.py completed."

# echo "Running data_processing.py..."
# python data_processing.py
# echo "data_processing.py completed."

# echo "Running generate_unique_terms.py..."
# python generate_unique_terms.py
# echo "generate_unique_terms.py completed."

# echo "NER Processing"
# python ner_processing.py

# echo "Running entity_linking.py..."
# python entity_linking.py
# echo "entity_linking.py completed."

# echo "Setup scripts completed. Starting Flask app..."

# # Start the Flask app
# exec "$@"

# #TESTING VERSION

echo "***Running testing version***"
echo "Running database setup scripts..."

echo "Environmental variables"
python -m cProfile -o show_env.prof show_env.py

echo "Running database_setup.py..."
python -m cProfile -o database_setup.prof database_setup.py
echo "database_setup.py completed."

echo "Running data_processing.py..."
python -m cProfile -o data_processing.prof data_processing.py
echo "data_processing.py completed."

echo "Running generate_unique_terms.py..."
python -m cProfile -o generate_unique_terms.prof generate_unique_terms.py
echo "generate_unique_terms.py completed."

echo "NER Processing"
python -m cProfile -o ner_processing.prof ner_processing.py

echo "Running entity_linking.py..."
python -m cProfile -o entity_linking.prof entity_linking.py
echo "entity_linking.py completed."

echo "Setup scripts completed. Starting Flask app..."

# Start the Flask app
exec "$@"

