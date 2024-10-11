# app/entrypoint.sh
#!/bin/sh

# Exit immediately if a command exits with a non-zero status
set -e

echo "Running database setup scripts..."

# Execute setup scripts
python database_setup.py
python data_processing.py
python generate_unique_terms.py
python entity_linking.py

echo "Setup scripts completed. Starting Flask app..."

# Start the Flask app
exec "$@"
