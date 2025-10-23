#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Configuration for backoff
MAX_RETRIES=${MONGO_MAX_RETRIES:-10}
SLEEP_TIME=${MONGO_RETRY_DELAY:-5}
MONGO_HOST=${MONGO_HOST:-mongodb}
MONGO_PORT=${MONGO_PORT:-27017}

if [ -z "${MONGO_URI}" ]; then
    MONGO_URI=$(python - <<'PY'
from app.util.mongo_env import build_mongo_uri

print(build_mongo_uri(), end="")
PY
    )
fi

echo "MONGO_URI is set to: ${MONGO_URI}"
echo "Waiting for MongoDB to be ready...REALLY REAL"

# Export MONGO_URI so it's available to Python
export MONGO_URI

# Backoff loop to wait for MongoDB
for ((i=1;i<=MAX_RETRIES;i++)); do
    echo "Attempt $i/$MAX_RETRIES: Checking MongoDB connection..."
    python -c "
import os
import pymongo
MONGO_URI = os.environ.get('MONGO_URI')
print('MONGO_URI:', MONGO_URI)
client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
client.admin.command('ping')
" && break
    echo "MongoDB is not ready yet. Waiting ${SLEEP_TIME} seconds..."
    sleep ${SLEEP_TIME}
done

# Verify if MongoDB is up after retries
python -c "
import os
import pymongo
MONGO_URI = os.environ.get('MONGO_URI')
print('Final check - MONGO_URI:', MONGO_URI)
client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
client.admin.command('ping')
" || { echo "MongoDB did not become ready in time after ${MAX_RETRIES} attempts. Exiting."; exit 1; }

echo "MongoDB is up and running."

if [ "${RUN_BOOTSTRAP:-0}" = "1" ]; then
    echo "RUN_BOOTSTRAP enabled - executing data bootstrap pipeline"
    /app/bootstrap_data.sh || echo "Bootstrap pipeline exited with status $?"
else
    echo "RUN_BOOTSTRAP not enabled - skipping data bootstrap"
fi

echo "Starting Flask app..."
exec "$@"

