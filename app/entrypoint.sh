#!/bin/bash
# Created: 2025-10-23
# Purpose: Wait for MongoDB to become available before starting Flask.
#          Builds the connection URI dynamically from canonical MongoDB environment variables.

set -e  # Exit immediately on any error

# ===============================
# Configuration
# ===============================
MAX_RETRIES=${MONGO_MAX_RETRIES:-10}
SLEEP_TIME=${MONGO_RETRY_DELAY:-5}
MONGO_HOST=${MONGO_HOST:-mongodb}
MONGO_PORT=${MONGO_PORT:-27017}

# Build Mongo URI using canonical Docker vars
# Align with the new MONGO_ROOT_* names exposed via .env and docker-compose.
MONGO_ROOT_USERNAME=${MONGO_ROOT_USERNAME:-${MONGO_INITDB_ROOT_USERNAME:-admin}}
MONGO_ROOT_PASSWORD=${MONGO_ROOT_PASSWORD:-${MONGO_INITDB_ROOT_PASSWORD:-secret}}
# Accept APP_MONGO_URI so we can keep exporting MONGO_URI for the Python code paths.
APP_MONGO_URI=${APP_MONGO_URI:-"mongodb://${MONGO_ROOT_USERNAME}:${MONGO_ROOT_PASSWORD}@${MONGO_HOST}:${MONGO_PORT}/admin"}
MONGO_URI=${MONGO_URI:-${APP_MONGO_URI}}

echo "MONGO_URI is set to: ${MONGO_URI}"
echo "Waiting for MongoDB to be ready..."

# Export MONGO_URI for Python and Flask
export MONGO_URI

# ===============================
# Backoff loop for Mongo readiness
# ===============================
for ((i=1;i<=MAX_RETRIES;i++)); do
    echo "Attempt $i/$MAX_RETRIES: Checking MongoDB connection..."
    python3 -c "
import os, pymongo
uri = os.environ.get('MONGO_URI')
print('MONGO_URI:', uri)
client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
client.admin.command('ping')
print('✅ MongoDB connection successful.')
" && break
    echo "❌ MongoDB not ready yet. Waiting ${SLEEP_TIME} seconds..."
    sleep ${SLEEP_TIME}
done

# ===============================
# Final check
# ===============================
python3 -c "
import os, pymongo
uri = os.environ.get('MONGO_URI')
print('Final check - MONGO_URI:', uri)
client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
client.admin.command('ping')
print('✅ MongoDB is up and responding.')
" || { echo "❌ MongoDB did not become ready after ${MAX_RETRIES} attempts. Exiting."; exit 1; }

echo "MongoDB is up and running."

# ===============================
# Optional bootstrap
# ===============================
if [ "${RUN_BOOTSTRAP:-0}" = "1" ]; then
    echo "RUN_BOOTSTRAP enabled - executing data bootstrap pipeline..."
    /app/bootstrap_data.sh || echo "Bootstrap pipeline exited with status $?"
else
    echo "RUN_BOOTSTRAP not enabled - skipping data bootstrap."
fi

# ===============================
# Launch Flask
# ===============================
echo "Starting Flask app..."
exec "$@"
