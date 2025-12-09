# Historical Document Reader

The Historical Document Reader is a Flask application that helps historians and researchers ingest, explore, and analyse large collections of digitised material. It combines a traditional search-and-browse interface backed by MongoDB with a LangChain-powered Historian Agent that can answer questions conversationally while citing primary sources.

## Development notes: November
Attempted to include in-web updating of database. I think it is just too complex. It would be best served by being outside of the web service. It should have an option for local Ollama and another script for Cloud. The database should be setup by checking a set of directories for new materials, both image and JSON, and when present updating. But this online system is too cumbersome. 

I need to revert to the old system where the setup was offline and the running was online. It will be more robust and easy to setup. There is no need for this complexity. 


What I need to do:
1. Make ingestion scripts for HPC
2. 

## Table of contents

1. [Core capabilities](#core-capabilities)
2. [Repository structure](#repository-structure)
3. [Getting started](#getting-started)
4. [Using the application](#using-the-application)
5. [Data ingestion and maintenance](#data-ingestion-and-maintenance)
6. [Data directory layout](#data-directory-layout)
7. [Historian Agent configuration reference](#historian-agent-configuration-reference)
8. [Docker operations cheat-sheet](#docker-operations-cheat-sheet)
9. [Troubleshooting](#troubleshooting)
10. [Planned features](#planned-features)
11. [Contributing](#contributing)
12. [License & contact](#license--contact)

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
│   ├── main.py               # Flask app factory, session setup, caching, and configuration loading  # renamed to avoid package shadowing
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
   - `pip install -r requirements.txt` to mirror the container build; this top-level file pulls in `app/requirements.txt` so both environments share the same pins. <!-- Documenting the relationship between dependency manifests per maintainer request. -->

2. **Clone the repository**
   ```bash
   git clone https://github.com/proflouishyman/nosql_reader.git
   cd nosql_reader
   ```

3. **Run the bootstrap script**
   Use the interactive helper to create the default host directories one level above the repository and synchronise them with `.env`:
   ```bash
   bash scripts/initial_setup.sh
   ```
   The script will confirm the target paths (for archives, MongoDB data, and Flask session storage), copy `.env.example` to `.env` if needed, and update the directory-related variables. After preparing the folders it offers to rotate the MongoDB root username/password while keeping the ready-to-use defaults (`admin` / `change-me`). You can safely rerun it anytime you want to regenerate the defaults or adjust credentials.

4. **Review environment variables**
   - Open `.env` to verify the paths written by the setup script and tweak any credentials or feature flags.
   - Specify whether the Historian Agent should use Ollama or OpenAI by editing `HISTORIAN_AGENT_MODEL_PROVIDER` and its companion variables. <!-- Updated to note ingestion defaults follow .env provider selection. -->
   - Control the default ingestion model by setting `HISTORIAN_AGENT_MODEL`; the settings UI now loads this value for both Ollama and OpenAI providers. <!-- Added note clarifying the ingestion selector uses the shared model default. -->
   - Provide a unique `SECRET_KEY`. MongoDB defaults to `MONGO_ROOT_USERNAME=admin` and `MONGO_ROOT_PASSWORD=change-me`; change them in `.env` or via the setup script before production use.

5. **Start the stack**
   ```bash
   docker compose up --build
   ```
   The Flask app will be available at `http://localhost:5000`. MongoDB listens on `localhost:27017` for local tooling.

## Using the application

### Accessing the web UI

Open `http://localhost:5000` in your browser once Docker reports that the Flask container is ready. The top navigation bar provides quick access to Home, Search Database, ChatBot, Database Information, UI Settings, and the in-app Help guide.

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

Open **UI Settings → Data ingestion** to add new source material to the ingestion directory defined by `ARCHIVES_PATH` (inside the container) and `ARCHIVES_HOST_PATH` (on the host). The section lets you:

- Upload one or many `.json` or `.jsonl` files with an optional subfolder path.
- Review a snapshot of the most recent files already present in the archive directory.
- Download a ZIP bundle of every archive file while preserving the folder structure for offline analysis or backups.

Uploaded files are saved immediately. The next time you run `data_processing.py`—either manually or via `bootstrap_data.sh`—the new documents will be hashed, validated, and inserted if they have not been seen before.

### Data ingestion and maintenance

- Use `/app/bootstrap_data.sh` (inside the container) to run the end-to-end ingestion pipeline. It hashes archive files, loads new documents, recalculates unique term statistics, and runs enrichment scripts.
- You can invoke the bootstrap at container startup by setting `RUN_BOOTSTRAP=1` in `.env`. The script exits gracefully if the archives directory is empty, so it is safe to keep the flag enabled in development.
- Additional helper scripts such as `backup_db.py` and `restore_db.py` are available for database maintenance. Execute them from the repository root or inside the running container using `docker compose exec`.
- The export and cleanup utilities in `app/util/` (for example `export_unique_terms.py`, `export_linked_terms.py`, and `delete_db.py`) use `app.database_setup` so they inherit the canonical `MONGO_URI`. Run them from the repository root or with `python -m app.util.<script>` to ensure the helpers resolve correctly inside Docker.
- Use the **Settings → Data ingestion** selector to switch between Ollama and OpenAI before running scans or rebuilds. The default provider and model honour `HISTORIAN_AGENT_MODEL_PROVIDER` and `HISTORIAN_AGENT_MODEL`, and any ingestion errors are reported inline on the page as well as in the Flask logs for easier troubleshooting. <!-- Added bullet documenting the restored provider toggle, env defaults, and surfaced errors. -->

## Data directory layout

The ingest pipeline expects a mirrored relationship between metadata files and any associated images or media assets. The precise layout is configurable, but keeping everything under a single archive root simplifies ingestion and preview rendering.

### Recommended structure

```
ARCHIVES_HOST_PATH/
├── rolls/
│   ├── tray_01/
│   │   ├── page_001.png
│   │   ├── page_001.png.json
│   │   ├── page_002.png
│   │   └── page_002.png.json
│   └── tray_02/
│       ├── images/
│       │   ├── frame_01.tif
│       │   └── frame_02.tif
│       └── metadata/
│           ├── frame_01.tif.json
│           └── frame_02.tif.json
└── newspapers/
    └── 1901-05-12.jsonl
```

- **One JSON per image** – For single-record files, append `.json` to the original filename (e.g., `page_001.png.json`).
- **JSONL batches** – Multi-record line-delimited JSON files (`*.jsonl`) can live alongside images. Each record should include a `source_file` or equivalent field indicating the relative path to the related asset if one exists.
- **Mirrored subfolders** – If you separate media and metadata into different subdirectories, keep the relative path identical beyond the divergent root (`images/` vs `metadata/`). The application drops the `.json` suffix and expects the remaining path to resolve to a media file somewhere under the archive root.

### How the application resolves images

1. During ingestion, `data_processing.py` stores the archive-relative path for each JSON file in the document record.
2. When rendering a document detail view, Flask removes the `.json` (or `.jsonl` record identifier) from that stored path to compute the expected media filename.
3. The `/images/<path:filename>` route streams the resulting file from disk. Any nested directory structure is preserved, so `rolls/tray_02/images/frame_01.tif` is served correctly as long as the file exists under `ARCHIVES_HOST_PATH`.

If you are ingesting material that does not have associated images, the UI simply omits the preview panel—no additional configuration is required.

## Historian Agent configuration reference

The following table maps each environment variable or UI control to its behaviour. All values can be overridden at runtime from the Historian Agent configuration sidebar; those overrides are stored in the user’s session.

| Setting | Environment variable | Description |
| --- | --- | --- |
| Provider | `HISTORIAN_AGENT_MODEL_PROVIDER` | `ollama` for local models, `openai` for cloud models. Determines which configuration fields are required. |
| Model name | `HISTORIAN_AGENT_MODEL` | Identifier passed to the underlying provider (e.g., `llama2`, `gpt-4o-mini`). |
| Ollama base URL | `HISTORIAN_AGENT_OLLAMA_BASE_URL` / `OLLAMA_BASE_URL` | Base HTTP endpoint for a local Ollama server. Ignored when using OpenAI. | <!-- Clarified both env names because the backend now reads either for compatibility. -->
| OpenAI API key | `OPENAI_API_KEY` | Credential for OpenAI requests. May be provided through the UI so keys are not stored on disk. |
| Temperature | `HISTORIAN_AGENT_TEMPERATURE` | Controls response creativity. Lower values produce deterministic summaries. |
| Max documents | `HISTORIAN_AGENT_CONTEXT_K` | Number of retrieved documents to feed into the RAG pipeline per question. |
| Field whitelist | `HISTORIAN_AGENT_CONTEXT_FIELDS` | Comma-separated list of document fields eligible for retrieval context. |
| Prompt prefix | `HISTORIAN_AGENT_SYSTEM_PROMPT` | System message prepended to every conversation for tone/behaviour guidance. |

### Semantic retrieval controls (new)

- `HISTORIAN_AGENT_USE_VECTOR_RETRIEVAL` – Enables the hybrid retriever that blends keyword search with vector similarity. Leave this `false` until the migration script has populated embeddings.
- `HISTORIAN_AGENT_EMBEDDING_PROVIDER` / `HISTORIAN_AGENT_EMBEDDING_MODEL` – Select the embedding backend (`local` → HuggingFace, `openai` → OpenAI API) and model used for chunk vectors.
- `HISTORIAN_AGENT_CHUNK_SIZE` / `HISTORIAN_AGENT_CHUNK_OVERLAP` – Mirror the chunker configuration so retrieval respects the intended context window.
- `HISTORIAN_AGENT_VECTOR_STORE` / `CHROMA_PERSIST_DIRECTORY` – Choose the vector store implementation (currently `chroma`) and its persistence path.
- `HISTORIAN_AGENT_HYBRID_ALPHA` – Weight applied to vector results when combining with keyword matches (0 = keyword only, 1 = vector only).

Run `python scripts/embed_existing_documents.py` to generate the initial chunk embeddings before turning the feature flag on. The script adds a `document_chunks` collection, writes vectors to ChromaDB, and can be resumed safely if interrupted.

When an override is active, the sidebar displays a badge showing the active provider and highlights any values that differ from the environment defaults. The backend validates keys and endpoints before accepting a change and responds with descriptive error messages for invalid combinations.

## Docker operations cheat-sheet

| Task | Command |
| --- | --- |
| Build and start stack | `docker compose up --build` |
| Start stack in background | `docker compose up -d` |
| Follow logs for all services | `docker compose logs -f` |
| Run ingestion manually inside app container | `docker compose exec flask_app ./bootstrap_data.sh` |
| Drop into a shell inside Flask container | `docker compose exec flask_app bash` |
| Drop into MongoDB shell | `docker compose exec mongodb mongosh -u "$MONGO_ROOT_USERNAME" -p "$MONGO_ROOT_PASSWORD" --authenticationDatabase admin` |
| Stop and remove containers | `docker compose down` |
| Stop containers and clear volumes | `docker compose down -v` (Use with caution; this deletes MongoDB data.) |

Navigate to **Historian Agent** in the top navigation to open the conversational interface. The page provides:

- A chat workspace with prompt suggestions, message bubbles, and citations that link back to the supporting documents.
- A configuration sidebar that mirrors the environment variables defined in `.env`. You can:
  - Toggle between **Ollama (local)** and **OpenAI (cloud)** providers.
  - Supply an Ollama base URL, choose the model name, or paste an OpenAI API key without restarting the app.
  - Adjust temperature, maximum context documents, prompt text, and field selection.
  - Reset to the environment defaults at any time.

When you change any setting, the LangChain pipeline is rebuilt automatically and the new configuration is stored in your browser session. The backend validates inputs, persists chat history for one hour per conversation, and includes defensive error messages when a provider is unavailable or misconfigured.

### Managing archive files

- **Historian Agent errors** – Check the configuration sidebar for validation issues. The `/historian-agent/config` endpoint returns a detailed error message if the chosen provider cannot be initialised.
- **MongoDB connectivity** – Ensure the paths defined in `.env` exist and your host user has write permissions. Inspect container logs with `docker compose logs nosql_reader_mongodb`.
- **Large dependency downloads** – The Docker image installs spaCy models at build time. Use a registry cache or pre-built image to avoid repeated downloads in CI.

## Planned features

The roadmap below captures high-priority enhancements that build on top of the current ingestion, Historian Agent, and archive management tooling:

- **Research notepad** – Inline scratchpad for saving excerpts, citations, and personal annotations while browsing documents or chatting with the Historian Agent.
- **In-browser JSON editing** – Safe, versioned editor that lets authorised users correct or enrich ingested records directly from the detail view without rerunning the ingestion pipeline.
- **External dataset linking** – Support for referencing related records stored in other systems (e.g., Wikidata, archival finding aids) so each document can expose verified authority links.
- **Network analysis visualisations** – Graph views that surface entity co-occurrence networks, correspondence maps, or other relationship insights derived from the MongoDB corpus.
- **Scheduled re-ingestion jobs** – Workflow automation that periodically reconciles new archive drops, re-runs enrichment models, and alerts maintainers when schema changes are detected.
- **Multi-user profiles** – Optional authentication and role-based access to gate edit features, store personalised Historian Agent settings, and track user activity for analytics.

These items are intentionally modular; each feature will be developed behind a configuration flag so deployments can adopt them incrementally.
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
- <!-- Added description of the redesigned ingestion settings panel. -->
- The **Settings → Data ingestion** panel now lists Docker volume mounts read-only, including a short folder tree for each target.
- Use **Scan for new images** to process any new media under those mounts with the default ingestion configuration.
- Use **Rebuild database** to clear and repopulate MongoDB documents from the mounted archives in one step.

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

