#!/bin/bash
set -euo pipefail

DATA_DIR=${ARCHIVES_PATH:-/app/archives}

if [ ! -d "${DATA_DIR}" ]; then
  echo "Archives directory '${DATA_DIR}' does not exist. Skipping ingestion." >&2
  exit 0
fi

python database_setup.py
python data_processing.py "${DATA_DIR}"
python generate_unique_terms.py
python ner_processing.py
python entity_linking.py
