# Historical Document Reader

The Historical Document Reader is a Flask + MongoDB application for ingesting, searching, and analyzing digitized historical documents, with an optional RAG-powered Historian Agent for conversational research workflows. <!-- Updated overview to match current app scope and features. -->

## Status snapshot

- Runs via Docker Compose with `app` and `mongodb` services.
- Web UI is exposed at `http://localhost:5001` (container port 5000).
- MongoDB is exposed at `localhost:27017` for local tools.
- The database name is hard-coded as `railroad_documents` in `app/database_setup.py` and related scripts.
- Vector retrieval is supported when ChromaDB persistence is mounted and `HISTORIAN_AGENT_USE_VECTOR_RETRIEVAL=true`.
<!-- Added current runtime facts (ports, DB name, vector retrieval toggle) based on docker-compose and code. -->

## Table of contents

1. [Core capabilities](#core-capabilities)
2. [Repository structure](#repository-structure)
3. [Quick start (Docker)](#quick-start-docker)
4. [Configuration reference](#configuration-reference)
5. [Using the web app](#using-the-web-app)
6. [Data ingestion workflows](#data-ingestion-workflows)
7. [Vector retrieval and RAG pipelines](#vector-retrieval-and-rag-pipelines)
8. [Data layout and schema notes](#data-layout-and-schema-notes)
9. [Scripts and utilities](#scripts-and-utilities)
10. [Troubleshooting](#troubleshooting)
11. [Documentation map](#documentation-map)
12. [Planned features](#planned-features)
13. [License and contact](#license-and-contact)
<!-- Rebuilt TOC to match the updated sections. -->

## Core capabilities

- Multi-field keyword search with AND/OR toggles and paginated results.
- Document detail views with metadata, OCR text, and linked media previews.
- Historian Agent with multiple query methods: Good (direct RAG), Better (adversarial RAG), Best (tiered), and Tier 0 corpus exploration.
- In-app ingestion controls for scanning mounted image directories or rebuilding the database from mounted archives.
- Runtime UI configuration via `app/config.json` (reloaded per request).
<!-- Consolidated feature list to reflect current UI routes and templates. -->

## Repository structure

```
nosql_reader/
├── app/                      # Flask app, templates, static assets, and historian_agent module
├── docs/                     # Design and architecture documents (some are forward-looking)
├── scripts/                  # Host-side helpers (initial setup, legacy embedding migration)
├── archives/                 # Default host directory for JSON and media (mounted into container)
├── mongo_data/               # Default host directory for MongoDB data
├── flask_session/            # Default host directory for Flask session files
├── docker-compose.yml        # Docker Compose definition (ports, volumes, env mapping)
├── .env                      # Environment variables (repo includes a baseline file)
├── setup.md                  # Detailed Mac/Windows setup guide
├── readme.md                 # You are here
└── quick_check.sh            # Ingestion/Ollama connectivity diagnostics
```
<!-- Updated structure to match current repository contents. -->

## Quick start (Docker)

1. Install Docker Engine and the Docker Compose plugin.
2. Review `.env` in the repo root and update paths/credentials as needed.
3. If you want the helper to create host directories, run:
   ```bash
   bash scripts/initial_setup.sh
   ```
   Note: `scripts/initial_setup.sh` requires `.env.example`. The repo currently ships with `.env` but no `.env.example`, so create `.env.example` from your baseline before running the script. <!-- Added clarification about the missing template that the script expects. -->
4. Start the stack:
   ```bash
   docker compose up --build
   ```
5. Open the web UI at `http://localhost:5001`.
6. (Optional) Run the ingestion bootstrap inside the container:
   ```bash
   docker compose exec app /app/bootstrap_data.sh
   ```
<!-- Updated startup steps to reflect the current compose port and scripts. -->

## Configuration reference

Set values in `.env` and restart containers after changes.

| Purpose | Key variables | Notes |
| --- | --- | --- |
| MongoDB root credentials | `MONGO_ROOT_USERNAME`, `MONGO_ROOT_PASSWORD` | Used by `docker-compose.yml` to seed MongoDB. |
| Application MongoDB URI | `APP_MONGO_URI` | Primary connection string used by the Flask app and most scripts. |
| Legacy MongoDB URI | `MONGO_URI` | Some helper scripts still read `MONGO_URI`; keep in sync with `APP_MONGO_URI`. |
| Database name | (hard-coded) | The app uses `railroad_documents` directly in code. |
| Host archive path | `ARCHIVES_HOST_PATH` | Host path for JSON/media; mounted to `ARCHIVES_PATH` in the container. |
| Host Mongo data path | `MONGO_DATA_HOST_PATH` | Host path for MongoDB persistence. |
| Host session path | `SESSION_HOST_PATH` | Host path for Flask session files. |
| Chroma persistence | `CHROMA_HOST_PATH`, `CHROMA_PERSIST_DIRECTORY` | Required when vector retrieval is enabled. |
| Historian Agent provider | `HISTORIAN_AGENT_MODEL_PROVIDER` | `ollama` or `openai`. Also used as the default ingestion provider. |
| Historian Agent model | `HISTORIAN_AGENT_MODEL` | Shared default model name for the agent and ingestion defaults. |
| Agent behavior | `HISTORIAN_AGENT_TEMPERATURE`, `HISTORIAN_AGENT_CONTEXT_K`, `HISTORIAN_AGENT_CONTEXT_FIELDS` | Tune answer style and retrieval context. |
| Vector retrieval | `HISTORIAN_AGENT_USE_VECTOR_RETRIEVAL`, `HISTORIAN_AGENT_EMBEDDING_PROVIDER`, `HISTORIAN_AGENT_EMBEDDING_MODEL` | Enable hybrid retrieval and set embedding backend/model. |
| Ollama/OpenAI | `OLLAMA_BASE_URL`, `OPENAI_API_KEY` | Supply local Ollama URL or OpenAI key. |
<!-- Expanded config table with the variables actually used in code and compose. -->

## Using the web app

### Search

- Navigate to **Search Database** from the top nav.
- Choose one or more fields, supply keywords, and pick AND/OR logic.
- Click a result to open the document detail view.
- Use the export button to download selected results as CSV (filename, summary, full JSON). <!-- Clarified export behavior based on `/export_selected_csv`. -->

### Historian Agent

- Open **Historian Agent** in the nav bar.
- Choose a query method (Good, Better, Best, or Tier 0 corpus exploration).
- Use the configuration panel to change provider, model, temperature, context size, and retrieval fields without restarting the app.
- Session history is retained for the current browser session (timeouts are enforced server-side).
<!-- Updated to match the actual query methods and settings panel behavior. -->

### UI settings

- **UI Preferences** writes updates to `app/config.json` so layout and styling can be tweaked without redeploying.
- **Data ingestion** lists mounted paths, lets you scan for new images, and can rebuild the database from mounted archives.
<!-- Added settings behaviors tied to `/settings` routes and `config.json`. -->

## Data ingestion workflows

There are two main ingestion paths.

- **UI-driven image ingestion**: The settings page can scan mounted directories and run OCR/structuring via Ollama or OpenAI. The defaults are seeded from `HISTORIAN_AGENT_MODEL_PROVIDER` and `HISTORIAN_AGENT_MODEL`, and you can override them in the UI.
- **Batch JSON/JSONL ingestion**: `app/bootstrap_data.sh` runs `database_setup.py`, `data_processing.py`, `generate_unique_terms.py`, `ner_processing.py`, and `entity_linking.py` in sequence. Use this to process JSON archives at scale.

Caution: the **Rebuild database** action in the ingestion settings page clears `documents`, `unique_terms`, and `field_structure` before reprocessing. <!-- Documented the destructive rebuild behavior from routes.py. -->

## Vector retrieval and RAG pipelines

The Historian Agent supports three RAG modes plus Tier 0 corpus exploration. Hybrid retrieval uses Mongo keyword search plus a vector store (Chroma by default).

Recommended setup steps:

1. Ensure `CHROMA_HOST_PATH` and `CHROMA_PERSIST_DIRECTORY` are set and mounted in Docker Compose.
2. Initialize vector storage and indexes:
   ```bash
   docker compose exec app python historian_agent/setup_rag_database.py
   ```
3. Embed existing documents:
   ```bash
   docker compose exec app python historian_agent/embed_existing_documents.py --provider ollama --model qwen3-embedding:0.6b
   ```

Notes:
- `app/historian_agent/embed_existing_documents.py` is the maintained migration script. The legacy `scripts/embed_existing_documents.py` imports classes that no longer exist and is likely to fail without updates.
- If `HISTORIAN_AGENT_USE_VECTOR_RETRIEVAL=false`, the agent falls back to keyword-only retrieval.
<!-- Added accurate RAG setup steps and clarified the active migration script. -->

## Data layout and schema notes

The ingestion pipeline expects JSON/JSONL metadata to mirror the media layout under the archive root.

Example layout:

```
ARCHIVES_HOST_PATH/
├── rolls/
│   ├── tray_01/
│   │   ├── page_001.png
│   │   └── page_001.png.json
│   └── tray_02/
│       ├── images/
│       │   └── frame_01.tif
│       └── metadata/
│           └── frame_01.tif.json
└── newspapers/
    └── 1901-05-12.jsonl
```

Schema details for the railroad dataset (including the `ocr_text` and `summary` fields and chunk ID types) are documented in `app/data_structure.md`. Update `HISTORIAN_AGENT_CONTEXT_FIELDS` to include the fields that actually contain text for your dataset. <!-- Added dataset-specific schema pointer and context field guidance. -->

## Scripts and utilities

- `scripts/initial_setup.sh`: Creates host directories and updates `.env` (requires `.env.example`).
- `run_app.sh`: Convenience wrapper for `docker compose up`.
- `quick_check.sh`: Validates Ollama connectivity, Docker status, and archive mounts.
- `backup_db.py` and `restore_db.py`: Mongodump-based backups for `railroad_documents`.
- `app/setup/setup_databases.py`: Creates MongoDB collections and indexes.
- `app/setup/setup_ingest_data.py`: Standalone JSON ingestion entry point.
- `app/util/`: Export, cleanup, and maintenance helpers (run inside the container).
<!-- Consolidated available scripts with their actual paths. -->

## Troubleshooting

- If ingestion fails, start with `quick_check.sh` and then inspect `app/routes.log`, `app/database_setup.log`, and `app/database_processing.log`.
- For RAG migrations, check `app/historian_agent/embed_migration*.log` and `app/historian_agent/setup_rag_database.log`.
- If the UI is unreachable, confirm `docker compose ps` shows the `app` container and that port `5001` is not in use.
- `quick_check.sh` prints `http://localhost:5000`, but the default Docker Compose mapping exposes the UI on `http://localhost:5001`.
<!-- Updated troubleshooting pointers with real log file locations. -->

## Documentation map

- `setup.md`: Mac/Windows setup walkthrough and environment details.
- `docs/backend_complete_december.md`: Deep architecture reference (some sections are design-oriented and may describe future work).
- `docs/DATABASE_SETUP_DESIGN.md`: Database setup design and migration strategy (forward-looking).
- `app/historian_agent/readme_rag.md`: RAG pipeline notes and performance analysis.
- `app/data_structure.md`: MongoDB schema notes for the current dataset.
<!-- Added a map of the key docs and their scope. -->

## Planned features

The following items are tracked as planned or in-progress work across the docs and TODO lists:

- Research notepad and annotation tooling.
- In-browser JSON editing for corrective updates.
- Entity linking improvements and cross-dataset references.
- Network analysis visualizations.
- Scheduled ingestion jobs and automation.
- Multi-user profiles and access controls.
<!-- Preserved and normalized the roadmap into a clear list. -->

## License and contact

Licensing and contact details are not finalized. Please open an issue if you need clarification. <!-- Kept original licensing note while tightening wording. -->
