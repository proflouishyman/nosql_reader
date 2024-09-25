#!/bin/sh
set -e

# Function to wait for MongoDB to be ready
wait_for_mongo() {
    echo "Waiting for MongoDB to be ready..."
    until mongo --host mongodb --username "$MONGO_INITDB_ROOT_USERNAME" --password "$MONGO_INITDB_ROOT_PASSWORD" --authenticationDatabase admin --eval "db.adminCommand('ping')" >/dev/null 2>&1
    do
        echo "MongoDB is unavailable - sleeping"
        sleep 2
    done
    echo "MongoDB is up and running!"
}

# Wait for MongoDB
wait_for_mongo

# Run initialization scripts
python database_setup.py
python data_processing.py

# Execute the container's main process (what's set as CMD in Dockerfile)
exec "$@"
