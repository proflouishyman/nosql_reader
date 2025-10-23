# Historical Document Reader

The Historical Document Reader is a Flask application that helps historians and researchers ingest, explore, and analyse large collections of digitised material. It combines a traditional search-and-browse interface backed by MongoDB with a LangChain-powered Historian Agent that can answer questions conversationally while citing primary sources.

## Core capabilities

- **Rich search experience** – Multi-field search with AND/OR controls, paginated result sets, and CSV export so researchers can triage documents quickly.
- **Flexible document viewer** – Detail pages surface high-value metadata, associated media, and navigation shortcuts informed by the user’s search context.
- **Configurable UI** – All UI settings, field groupings, and layout choices are reloaded per-request from `config.json`, making it easy to experiment with presentation changes without redeploying.
- **Historian Agent** – A Retrieval-Augmented Generation (RAG) assistant built on LangChain. It can target either local Ollama models or hosted OpenAI models and always includes citations for the supporting documents it retrieved from MongoDB.
- **Archive management UI** – Upload additional JSON records directly from the web interface so the ingestion pipeline can pick them up without touching the server filesystem.

## Repository structure

```
nosql_reader/
├── app/
│   ├── app.py                # Flask app factory, session setup, caching, and configuration loading
│   ├── routes.py             # All HTTP routes including Historian Agent APIs and UI endpoints
│   ├── historian_agent/      # LangChain integration, retrieval utilities, and pipeline cache helpers
│   ├── database_setup.py     # MongoDB client helpers, ingestion utilities, and schema discovery logic
│   ├── data_processing.py    # Archive ingestion script (hashing, deduplication, field extraction)
│   ├── static/               # Global CSS (including Historian Agent experience styling)
│   └── templates/            # Jinja templates for every page, notably `historian_agent.html`
├── docs/                     # Design material including the Historian Agent implementation plan
├── docker-compose.yml        # Two-service stack (Flask + MongoDB) with externalised data directories
├── .env.example              # Environment variable template that documents every configurable option
└── readme.md                 # You are here
```

## Getting started

1. **Install prerequisites**
   - Docker Engine 24+ and Docker Compose Plugin 2+
   - Git and Python 3.10+ (optional for running scripts outside the container)

2. **Clone the repository**
   ```bash
   git clone https://github.com/proflouishyman/nosql_reader.git
   cd nosql_reader
   ```

3. **Prepare local directories**
   Create host directories to hold stateful data that should not live inside containers:
   - `ARCHIVES_HOST_PATH` – path to your raw JSON archives
   - `MONGO_DATA_HOST_PATH` – path for MongoDB’s data files
   - `SESSION_HOST_PATH` – path for persisted Flask session data

4. **Configure environment variables**
   - Copy `.env.example` to `.env` and update the paths and credentials.
   - Specify whether the Historian Agent should use Ollama or OpenAI by editing `HISTORIAN_AGENT_MODEL_PROVIDER` and its companion variables.
   - Provide `SECRET_KEY`, `MONGO_INITDB_ROOT_USERNAME`, and `MONGO_INITDB_ROOT_PASSWORD` for a secure deployment.

5. **Start the stack**
   ```bash
   docker compose up --build
   ```
   The Flask app will be available at `http://localhost:5000`. MongoDB listens on `localhost:27017` for local tooling.

## Using the application

### Accessing the web UI

Open `http://localhost:5000` in your browser once Docker reports that the Flask container is ready. Every page includes a top navigation bar that links to the Search interface, Archive Files page, Historian Agent, and any other templates defined in `config.json`.

### Performing searches

1. Choose one or more fields from the dropdown menus (for example, `title`, `description`, `subjects`).
2. Type the desired keywords. Toggle the AND/OR switch to control whether all terms must appear or whether any match is acceptable.
3. Press **Search** to run the query. Results stream in batches of 20 and you can scroll to load more.
4. Click a result card to open the full document view. The detail page displays every non-empty field, links to related media, and provides **Previous/Next** navigation based on your current result set.
5. Use the **Export CSV** button to download the current result set for offline analysis.

### Historian Agent experience

Navigate to **Historian Agent** in the top navigation to open the conversational interface. The page provides:

- A chat workspace with prompt suggestions, message bubbles, and citations that link back to the supporting documents.
- A configuration sidebar that mirrors the environment variables defined in `.env`. You can:
  - Toggle between **Ollama (local)** and **OpenAI (cloud)** providers.
  - Supply an Ollama base URL, choose the model name, or paste an OpenAI API key without restarting the app.
  - Adjust temperature, maximum context documents, prompt text, and field selection.
  - Reset to the environment defaults at any time.

When you change any setting, the LangChain pipeline is rebuilt automatically and the new configuration is stored in your browser session. The backend validates inputs, persists chat history for one hour per conversation, and includes defensive error messages when a provider is unavailable or misconfigured.

### Managing archive files

Visit **Archive Files** in the navigation to add new source material to the ingestion directory defined by `ARCHIVES_PATH` (inside the container) and `ARCHIVES_HOST_PATH` (on the host). The page lets you:

- Upload one or many `.json` or `.jsonl` files with an optional subfolder path.
- Review a snapshot of the most recent files already present in the archive directory.
- Download a ZIP bundle of every archive file while preserving the folder structure for offline analysis or backups.

Uploaded files are saved immediately. The next time you run `data_processing.py`—either manually or via `bootstrap_data.sh`—the new documents will be hashed, validated, and inserted if they have not been seen before.

#### Image placement for document viewers

Document detail pages automatically look for associated images using the relative path that was stored when the JSON archive was ingested. By default, `data_processing.py` records each file’s path relative to the archive root (for example `rolls/tray_1/page_03.png.json`). When a document is rendered, the application removes the trailing `.json` from that stored path and serves the remaining file (`rolls/tray_1/page_03.png`) straight from the archives directory.

To make sure previews work as expected:

- Keep JSON and image assets together under the same directory tree beneath your archive root.
- Name metadata files after their source image, appending `.json` (or `.jsonl`) to the original filename. Example: store the image at `archives/rolls/tray_1/page_03.png` and the companion metadata at `archives/rolls/tray_1/page_03.png.json`.
- If you prefer to separate large media collections, use nested subdirectories (e.g., `archives/rolls/images/page_03.png` and `archives/rolls/metadata/page_03.png.json`) and configure your ingestion/export tools to keep the relative directory structure identical for both the JSON and the image. The application only relies on the relative path match; it does not enforce a single top-level folder name.

Following this convention ensures the `/images/<path:filename>` route can resolve the expected asset without additional configuration.

### Data ingestion and maintenance

- Use `/app/bootstrap_data.sh` (inside the container) to run the end-to-end ingestion pipeline. It hashes archive files, loads new documents, recalculates unique term statistics, and runs enrichment scripts.
- You can invoke the bootstrap at container startup by setting `RUN_BOOTSTRAP=1` in `.env`. The script exits gracefully if the archives directory is empty, so it is safe to keep the flag enabled in development.
- Additional helper scripts such as `backup_db.py` and `restore_db.py` are available for database maintenance. Execute them from the repository root or inside the running container using `docker compose exec`.

## Troubleshooting

- **Historian Agent errors** – Check the configuration sidebar for validation issues. The `/historian-agent/config` endpoint returns a detailed error message if the chosen provider cannot be initialised.
- **MongoDB connectivity** – Ensure the paths defined in `.env` exist and your host user has write permissions. Inspect container logs with `docker compose logs nosql_reader_mongodb`.
- **Large dependency downloads** – The Docker image installs spaCy models at build time. Use a registry cache or pre-built image to avoid repeated downloads in CI.

## Contributing

1. Fork the repository and create a feature branch.
2. Make your changes with accompanying tests or manual verification steps.
3. Ensure docstrings and inline documentation clearly describe new behaviour.
4. Submit a pull request summarising the change, any migrations, and testing performed.

## License & contact

Project licensing and contact information are not yet finalised. Please open an issue if you need additional details.

