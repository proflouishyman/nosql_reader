# AUDIT RESULTS

## Scope and startup checks
- Read: `/Users/louishyman/coding/nosql/nosql_reader/docker-compose.yml`, `/Users/louishyman/coding/nosql/nosql_reader/requirements.txt`.
- `COMPLETE_SYSTEM_ARCHITECTURE.md` is **not present** in this repo (`find` found only `/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/SYSTEM_ARCHITECTURE_COMPLETE.md.pdf`).
- If this audit must include that exact `.md` file, it is blocked by missing source material.

---

## Task 1 - Internet Access Audit

| Service / outbound dependency | File:line | Use | Requires active internet? | Local alternative exists? |
|---|---|---|---|---|
| Wikidata API (`https://www.wikidata.org/w/api.php`) | `/Users/louishyman/coding/nosql/nosql_reader/app/setup/entity_linking.py:94-103`, `/Users/louishyman/coding/nosql/nosql_reader/app/setup/ner_worker.py:128-136`, `/Users/louishyman/coding/nosql/nosql_reader/app/setup/ner_processing.py:125-133` | Entity disambiguation/linking | Yes | No in-repo offline KB implementation |
| OpenAI Completions (legacy) | `/Users/louishyman/coding/nosql/nosql_reader/app/setup/entity_linking.py:137-145` | LLM entity linking | Yes | Yes (`fetch_wikidata_entity` path, no OpenAI call) |
| OpenAI Embeddings API | `/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/embeddings.py:315-319`, `:354-358` | Query/doc embeddings when provider=`openai` | Yes | Yes (`provider=ollama` or local model) |
| OpenAI Chat API (SDK + HTTP fallback) | `/Users/louishyman/coding/nosql/nosql_reader/app/llm_layer.py:313-320`, `:377-382` | Generation/verification when provider=`openai` | Yes | Yes (`ollama`, `lmstudio`) |
| OpenAI vision/chat ingestion | `/Users/louishyman/coding/nosql/nosql_reader/app/image_ingestion.py:443-447`, `:458-473` | Image-to-JSON ingestion | Yes | Yes (`_call_ollama_stage1_ocr` + `_call_ollama_stage2_structure`) |
| OpenAI in demographics builder | `/Users/louishyman/coding/nosql/nosql_reader/app/setup/build_demographics_db.py:115-125` | Demographic field extraction | Yes | Yes (`requests.post` to Ollama at `:136-142`) |
| Ollama HTTP API (`/api/chat`, `/api/generate`, `/api/embeddings`, `/api/tags`) | `/Users/louishyman/coding/nosql/nosql_reader/app/llm_layer.py:184-188`, `/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/embeddings.py:177-181`, `:397-401`, `/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/setup_rag_database.py:83`, `/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/embed_existing_documents.py:253`, `/Users/louishyman/coding/nosql/nosql_reader/app/routes.py:3610`, `/Users/louishyman/coding/nosql/nosql_reader/app/setup/person_synthesis.py:395` | LLM generation, embedding, model discovery | Not necessarily (works fully local/LAN) | N/A (already local-first) |
| LM Studio OpenAI-compatible endpoint | `/Users/louishyman/coding/nosql/nosql_reader/app/llm_layer.py:462-466` | Local LLM provider | Not necessarily (usually localhost) | Yes (Ollama/OpenAI providers) |
| HuggingFace model pull via `SentenceTransformer(...)` | `/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/embeddings.py:147-150` | Local embedding model load | First run usually yes (if uncached) | Yes (Ollama/OpenAI embedding providers) |
| HuggingFace model pull via `CrossEncoder(...)` | `/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/reranking.py:10-11` | Reranker model load | First run usually yes (if uncached) | Could disable reranking path with code changes; no explicit in-repo offline bundle |
| spaCy model wheel from GitHub | `/Users/louishyman/coding/nosql/nosql_reader/app/requirements.txt:22` | NLP model install | Yes at install/build time | Could pre-vendor wheel in image/artifact store |
| spaCy model download command | `/Users/louishyman/coding/nosql/nosql_reader/app/Dockerfile:22` | NLP model download in container build | Yes at build time | Could bake model into base image / offline artifact |
| Browser CDN/font dependencies | `/Users/louishyman/coding/nosql/nosql_reader/app/templates/base.html:11-16`, `/Users/louishyman/coding/nosql/nosql_reader/app/templates/historian_agent.html:81`, `/Users/louishyman/coding/nosql/nosql_reader/app/templates/demographics.html:293` | Google Fonts, Font Awesome, Marked, Chart.js | Yes (unless client cache/internal mirror) | Yes (self-host static assets) |

Notes:
- The codebase also contains duplicate outbound-call patterns in WIP CLI ingestion modules under `/Users/louishyman/coding/nosql/nosql_reader/app/CLI_ingestion_files_wip/`.
- No `hf_hub_download` calls were found.

---

## Task 2 - Current Embedding Pipeline

### 2.1 Which embedding model is in use and where loaded from?
There are **two active RAG stacks** with different embedding wiring:

1) `query-basic`/`query-adversarial`/`query-tiered` stack (frontend default)
- Frontend calls `/historian-agent/query-${method}` (`/Users/louishyman/coding/nosql/nosql_reader/app/static/historian_agent.js:537-538`).
- `RAGQueryHandler` builds `EmbeddingService(provider=APP_CONFIG.embedding.provider, model=APP_CONFIG.embedding.model)` (`/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/rag_query_handler.py:96-100`).
- Config defaults to `EMBEDDING_PROVIDER=ollama` and `HISTORIAN_AGENT_EMBEDDING_MODEL=qwen3-embedding:0.6b` (`/Users/louishyman/coding/nosql/nosql_reader/app/config.py:300-303`, `/Users/louishyman/coding/nosql/nosql_reader/.env:47`, `:270`).

2) `/historian-agent/query` stack (separate route)
- Uses `get_agent(...)` in `historian_agent/__init__.py` (`/Users/louishyman/coding/nosql/nosql_reader/app/routes.py:1767-1768`).
- Its vector path only accepts embedding providers `local/huggingface/openai` (`/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/__init__.py:393-403`).
- This conflicts with `.env` where `HISTORIAN_AGENT_EMBEDDING_PROVIDER=ollama` (`/Users/louishyman/coding/nosql/nosql_reader/.env:269`).

### 2.2 Is the model cached locally? Where?
- Chroma vectors are persisted at `CHROMA_PERSIST_DIRECTORY` (`/Users/louishyman/coding/nosql/nosql_reader/.env:273`, `/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/vector_store.py:29`, `:70-76`).
- HuggingFace cache path is **not configured in repo** (`HF_HOME`/`TRANSFORMERS_CACHE` not set), so exact cache path is environment-dependent and cannot be determined from codebase alone.

### 2.3 Embedding dimensionality and Chroma config match
- Embedding defaults: Ollama 1024, OpenAI 1536, local model dynamic (`/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/embeddings.py:41-43`, `:153-154`, `:195-197`).
- Config dimension is `1024` (`/Users/louishyman/coding/nosql/nosql_reader/app/config.py:303`, `/Users/louishyman/coding/nosql/nosql_reader/.env:53`).
- Chroma collection is created without explicit dimension; dimension is effectively enforced by first upsert (`/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/vector_store.py:98-103`, `/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/setup_rag_database.py:299-305`).
- `setup_rag_database.py` explicitly verifies Ollama embedding dimension before test upsert (`/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/setup_rag_database.py:190-200`).

### 2.4 Are embeddings generated at query time, ingest time, or both?
- **Both**.
- Query time: `VectorRetriever` embeds each user query before vector search (`/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/retrievers.py:86-93`).
- Ingest time: migration script chunks docs, generates embeddings, inserts Mongo chunks, then writes Chroma vectors (`/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/embed_existing_documents.py:498-513`, `:543-577`, `:592-606`).

### 2.5 Relevant snippets
```python
# /app/historian_agent/embeddings.py
DEFAULT_PROVIDER = "ollama"
DEFAULT_OLLAMA_MODEL = "qwen3-embedding:0.6b"
DEFAULT_DIMENSION_OLLAMA = 1024
```

```python
# /app/historian_agent/retrievers.py
query_embedding = self.embedding_service.embed_query(query)
search_results = self.vector_store.search(
    query_embedding=query_embedding,
    k=self.top_k,
)
```

```python
# /app/historian_agent/embed_existing_documents.py
chunks = self.chunker.chunk_document(document, content_fields=self.content_fields)
...
sub_emb = self.ollama_client.embed_texts(sub)
...
self.chunks_collection.insert_many(chunk_dicts, ordered=False)
self.vector_store.add_chunks(all_chunks)
```

```python
# /app/historian_agent/__init__.py
vector_store = get_vector_store(
    store_type=config.vector_store_type,
    persist_directory=persist_directory,
    collection_name="historian_document_chunks",
)
```

```python
# /app/historian_agent/vector_store.py
DEFAULT_COLLECTION_NAME = "historian_documents"
```

Additional finding:
- Collection name mismatch (`historian_document_chunks` vs `historian_documents`) may split reads/writes across different Chroma collections (`/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/__init__.py:426-430`, `/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/vector_store.py:28`).

---

## Task 3 - Multi-User Concurrency Assessment (up to 10 users)

### 3.1 Runtime/threading facts
- Docker Compose defines no WSGI worker manager (no gunicorn/uwsgi worker count); app runs `python main.py` via entrypoint/CMD (`/Users/louishyman/coding/nosql/nosql_reader/app/entrypoint.sh:74-75`, `/Users/louishyman/coding/nosql/nosql_reader/app/Dockerfile:42`, `/Users/louishyman/coding/nosql/nosql_reader/docker-compose.yml`).
- Flask app calls `app.run(...)` with no explicit threading param (`/Users/louishyman/coding/nosql/nosql_reader/app/main.py:136-138`).
- In installed Flask 3.0.3, `Flask.run` defaults `threaded=True` (`/opt/anaconda3/lib/python3.12/site-packages/flask/app.py`, `Flask.run` source line containing `options.setdefault("threaded", True)`).

### 3.2 Shared/global state and isolation
- Module-level globals are shared across requests: DB handles, login attempt map, cached agent handlers, exploration run buffers (`/Users/louishyman/coding/nosql/nosql_reader/app/routes.py:88-92`, `:100`, `:129-137`).
- SSE log stream is explicitly single-user/global queue with no session isolation (`/Users/louishyman/coding/nosql/nosql_reader/app/log_stream.py:2-9`, `:22-29`, `:97`).
- Session mechanism exists (`Flask-Session`) (`/Users/louishyman/coding/nosql/nosql_reader/app/main.py:76-86`, `:110`), but role/user isolation is minimal; `login_required` exists yet is commented out on many routes (`/Users/louishyman/coding/nosql/nosql_reader/app/routes.py:1478-1484`, `:1487`, `:1495`, `:1885`, `:2772`, `:3338`, `:3743`).

### 3.3 Blocking behavior and queue/fail modes
- LLM/Ollama calls are synchronous blocking HTTP calls (`/Users/louishyman/coding/nosql/nosql_reader/app/llm_layer.py:184-188`, `/Users/louishyman/coding/nosql/nosql_reader/app/setup/person_synthesis.py:395`).
- Query endpoints are synchronous request handlers (`/Users/louishyman/coding/nosql/nosql_reader/app/routes.py:329-363`, `:367-401`, `:406-440`, `:1741-1787`).
- Under 10 concurrent users, requests will occupy server threads and can stall on model latency/timeouts rather than async queueing.

### 3.4 Chroma concurrency
- App uses local `chromadb.PersistentClient` (`/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/vector_store.py:70-76`).
- Code has no application-level lock around vector reads/writes.
- From codebase alone, read/write concurrency guarantees are not explicitly documented; operational risk remains if writes happen concurrently with serving.

### 3.5 Risk summary

| Issue | Condition | What breaks | Severity |
|---|---|---|---|
| Dev server topology (single process, no worker manager) | 10 simultaneous long requests | Throughput drops; request latency spikes; fewer isolation controls | High |
| Global SSE queue is single-user | Multiple users open log stream | Cross-user log leakage, stream resets (`clear()`), premature `done` signals | High |
| Global mutable handlers/state | Simultaneous initialization/reset and shared caches | Non-deterministic behavior and cross-request coupling | Medium |
| Blocking LLM inference in request threads | Slow Ollama/OpenAI responses | Thread exhaustion/long tail latency; 5xx on timeouts | High |
| Auth decorator mostly disabled | Any public user | Full route access to settings/ingestion/export endpoints | High |
| Chroma local persistence + mixed read/write | Reindex/re-embed while serving | Potential lock/contention behavior not handled in app logic | Medium |

---

## Task 4 - Citation Bug Investigation

### 4.1 Target citation format
- Frontend formats citations as a **custom archival style** (`Baltimore & Ohio Railroad, Relief Department Records, ...`) with Chicago-like superscript placement behavior (`/Users/louishyman/coding/nosql/nosql_reader/app/static/historian_agent.js:377-379`, `:470-479`).

### 4.2 Citation construction paths found
- Canonical API boundary expects source dict entries with `display_name` and `url`: `sources_list_to_dict(...)` (`/Users/louishyman/coding/nosql/nosql_reader/app/routes.py:180-200`).
- RAG method routes (`query-basic`, `query-adversarial`, `query-tiered`) convert list->dict via this function (`/Users/louishyman/coding/nosql/nosql_reader/app/routes.py:203-278`).
- Frontend renderer expects object entries and calls `formatArchivalCitation(source?.display_name, source)` (`/Users/louishyman/coding/nosql/nosql_reader/app/static/historian_agent.js:381-390`, `:446-453`, `:470-473`).

### 4.3 Root cause
- `/historian-agent/query` returns `result['sources']` directly from `HistorianAgent.invoke` (`/Users/louishyman/coding/nosql/nosql_reader/app/routes.py:1781-1785`, `/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/__init__.py:252-255`).
- That source shape is a **list** of `{reference,id,title,snippet}` (`/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/__init__.py:267-282`) and does not provide `display_name/url` expected by frontend citation functions.
- Result: inconsistent citation formatting/fallback to `Unknown source` path (`/Users/louishyman/coding/nosql/nosql_reader/app/static/historian_agent.js:470-473`).

### 4.4 Impacted files
- `/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/__init__.py:267-282`
- `/Users/louishyman/coding/nosql/nosql_reader/app/routes.py:1741-1787`
- `/Users/louishyman/coding/nosql/nosql_reader/app/static/historian_agent.js:381-479`

### 4.5 Proposed fix
1. Normalize `/historian-agent/query` output to the same `sources_list_to_dict` contract used by other historian endpoints.
2. Or add a frontend adapter that accepts both list and dict formats before rendering citations.
3. Add a regression test asserting `display_name`+`url` are present for every assistant message source payload.

---

## Task 5 - Semantic Search Capability Assessment

### 5.1 What exists now
- Vector retrieval function: `VectorRetriever.get_relevant_documents(...)` (`/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/retrievers.py:79-107`).
- Underlying KNN call: `VectorStoreManager.search(...)` (`/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/vector_store.py:181-243`).
- Hybrid fusion exists: `HybridRetriever` with RRF (`/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/retrievers.py:297-405`).

### 5.2 Scores and metadata
- Nearest-neighbor distance/score are produced (`/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/vector_store.py:237-239`) and carried in retriever metadata (`/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/retrievers.py:125-130`).
- Current keyword search page (`POST /search`) is Mongo regex-only and does not expose vector scores (`/Users/louishyman/coding/nosql/nosql_reader/app/routes.py:1884-1920`).
- Chunk metadata is rich enough for cards (title/date/source/filename/etc.) (`/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/chunking.py:222-230`, `:232-240`).

### 5.3 Similar terms / related vocabulary
- No semantic nearest-term service found.
- Existing related-term feature is statistical co-occurrence (lift/log2 lift), not embedding-neighbor vocabulary (`/Users/louishyman/coding/nosql/nosql_reader/app/routes.py:3043-3183`, endpoint `:3186-3215`).

### 5.4 Capability gaps

| Capability gap | Current state | Needed build | Complexity |
|---|---|---|---|
| Standalone semantic page parallel to keyword search | No dedicated route/page for vector search (only historian-agent paths) | New route + template + JS for vector/hybrid search UX | Medium |
| Semantic search API returning scores | Scores exist internally but not exposed on search API | API endpoint that returns chunk/doc + score + distance + highlights | Medium |
| Similar-term suggestions from embeddings | Only co-occurrence lift endpoint exists | Add term-embedding index or ANN over vocabulary | High |
| Rich result cards from semantic backend | Metadata exists at chunk level | Map chunk-level metadata to parent doc card payloads, include snippets | Medium |
| Relevance controls (k, alpha, filters) in UI | Backend supports k/alpha/filtering primitives | UI controls + query param plumbing + validation | Low |

---

## Task 6 - Export Infrastructure Assessment

### 6.1 Existing export functionality
- Export selected docs to CSV: `/export_selected_csv` (`/Users/louishyman/coding/nosql/nosql_reader/app/routes.py:3742-3787`, frontend trigger `/Users/louishyman/coding/nosql/nosql_reader/app/static/script.js:435-454`).
- Demographics CSV export: `/api/demographics/export` (`/Users/louishyman/coding/nosql/nosql_reader/app/routes.py:2520-2552`).
- Archive ZIP export (includes all files under archive root): `/data-files/download-all` (`/Users/louishyman/coding/nosql/nosql_reader/app/routes.py:1316-1342`).

### 6.2 Available libraries relevant to export
- Present: `pandas`, `requests`, `Pillow` (`/Users/louishyman/coding/nosql/nosql_reader/app/requirements.txt:13-18`) and stdlib `csv/json/zipfile` used in routes (`/Users/louishyman/coding/nosql/nosql_reader/app/routes.py:34`, `:42`, `:52`).
- Not present in `requirements.txt`: `reportlab`, `fpdf`, `openpyxl`.

### 6.3 What a full Mongo document looks like (from code/docs)
- Ingestion writes JSON payload plus file metadata (`filename`, `relative_path`, `file_path`, `file_hash`) (`/Users/louishyman/coding/nosql/nosql_reader/app/data_processing.py:98-102`).
- `insert_document` persists document as-is (schema is flexible/dynamic) (`/Users/louishyman/coding/nosql/nosql_reader/app/database_setup.py:133-138`).
- Schema reference doc lists commonly present fields (`ocr_text`, `summary`, `sections`, `entities`, file metadata) (`/Users/louishyman/coding/nosql/nosql_reader/app/data_structure.md:31-53`).

### 6.4 Image linkage/retrieval
- `document_detail` derives image path from `relative_path` by stripping `.json` (`/Users/louishyman/coding/nosql/nosql_reader/app/routes.py:2875-2885`).
- `/images/<path>` serves image file from archive root (`/Users/louishyman/coding/nosql/nosql_reader/app/routes.py:2920-2929`).

### 6.5 What still needs to be built
- Text export: explicit TXT/Markdown export endpoint for selected docs (currently CSV only).
- JSON export: endpoint for selected docs as structured JSON/JSONL (currently CSV embeds JSON as a string column).
- Image export: targeted image bundle for selected document IDs (currently only whole-archive ZIP or per-image URL).

---

## Task 7 - PastPerfect CSV Ingestion Design

### 7.1 PastPerfect export fields (web research)
Based on PastPerfect references, common object-export fields include:
- `Object Name`, `Brief Description`, `Home Location`, `Status`, `Condition`, `Catalog Number/Object ID` (commonly present in list exports).
- Object records also commonly contain `Accession #`, `Date`, `Source` on object-entry workflows.

Because PastPerfect exports are configurable by module/template, the **exact** column set cannot be guaranteed from this repo alone.

### 7.2 Proposed mapping (PastPerfect -> MongoDB)

| PastPerfect field | Target MongoDB field | Status |
|---|---|---|
| Catalog Number / Object ID | `pastperfect.object_id` (new) | New field recommended |
| Accession # | `pastperfect.accession_number` (new) | New field recommended |
| Object Name / Title | `title` (or `pastperfect.object_name`) | `title` not guaranteed currently; add fallback |
| Date | `date` and/or `metadata.year` | Existing date fields are inconsistent/dynamic |
| Home Location | `archive_structure.physical_box` or `pastperfect.home_location` | Likely better as `pastperfect.home_location` then mapped view |
| Brief Description | `summary` and `pastperfect.brief_description` | Keep original in namespaced field |
| Source | `pastperfect.source` | New field recommended |
| Condition | `pastperfect.condition` | New field recommended |
| Status | `pastperfect.status` | New field recommended |

### 7.3 Ingest approach
- Build a new module, e.g. `/Users/louishyman/coding/nosql/nosql_reader/app/setup/ingest_pastperfect_csv.py`.
- Reason: existing ingest paths are JSON/TXT (`data_processing.py:227-233`) and image-to-JSON (`image_ingestion.py`), not tabular CSV importers.
- Pipeline:
  1. Read CSV/XLSX (pandas).
  2. Normalize headers + trim values.
  3. Create deterministic `match_key`.
  4. Upsert into `documents` with a namespaced `pastperfect` subdocument.
  5. Preserve provenance (`pastperfect.source_file`, ingest timestamp).

### 7.4 Match key recommendation
1. Primary: `accession_number` (if present).
2. Secondary: `object_id`/catalog number.
3. Fallback: normalized `title + date`.

`accession_number`/`object_id` fields are not currently first-class in this codebase, so linking by those keys requires adding them.

Sources used for this task:
- [PastPerfect Online: Records export guide](https://pastperfectonline.freshdesk.com/support/solutions/articles/1000298151-how-do-i-export-my-records-from-pastperfect-online-)
- [PastPerfect Museum Software: export reports](https://museumsoftware.com/v524.html)
- [PastPerfect object-entry tutorial (examples of Object ID, Accession #, Date, Source, Home Location)](https://pastperfectmuseumsoftware.freshdesk.com/support/solutions/articles/1000236203-object-entry-tutorial)

---

## Task 8 - User Access Tier Design

### 8.1 Current auth/session state
- Session framework exists (`Flask-Session`): `/Users/louishyman/coding/nosql/nosql_reader/app/main.py:76-86`, `:110`.
- Login is a single shared password hash + session flag (`/Users/louishyman/coding/nosql/nosql_reader/app/routes.py:94-100`, `:1789-1813`).
- No user table, no roles, no RBAC decorators.
- Many critical routes have `# @login_required` commented out (`/Users/louishyman/coding/nosql/nosql_reader/app/routes.py:1487`, `:1495`, `:1885`, `:2772`, `:3338`, `:3743`).

### 8.2 Sensitivity tagging in data
- No clear sensitivity/access-level field found in schema docs (`/Users/louishyman/coding/nosql/nosql_reader/app/data_structure.md:31-53`) or route-level filters.
- Therefore “Public vs sensitive” policy cannot be enforced with current document model.

### 8.3 Flask libraries available / missing
- Installed: Flask core + `Flask-Session` (`/Users/louishyman/coding/nosql/nosql_reader/app/requirements.txt:2-5`).
- Not installed: `flask-login`, `flask-principal` (or equivalent RBAC package).

### 8.4 Role-permission matrix (current system behavior)

| Feature / route | Public | Curator | Admin |
|---|---|---|---|
| Home `/` | Yes | Yes | Yes |
| Search UI + POST `/search` | Yes | Yes | Yes |
| Document view `/document/<id>` | Yes | Yes | Yes |
| Historian Agent pages + query endpoints | Yes | Yes | Yes |
| Corpus Explorer endpoints | Yes | Yes | Yes |
| Network + Demographics pages/APIs | Yes | Yes | Yes |
| Settings `/settings` + ingestion APIs under `/settings/data-ingestion/*` | Yes | Yes | Yes |
| Exports (`/export_selected_csv`, `/api/demographics/export`, `/data-files/download-all`) | Yes | Yes | Yes |
| User administration | No (not implemented) | No | No |
| Collection/system settings controls scoped to Admin | No (not implemented) | No | No |
| Sensitive-record filtering | No (not implemented) | No | No |

### 8.5 Recommended implementation path
1. Add user model + role field (`admin|curator|public`) and migrate from single shared password.
2. Add `flask-login` for identity/session and role-aware decorators.
3. Add document-level access metadata (e.g., `access_level`, `is_sensitive`) and enforce in all read queries.
4. Gate routes first, then gate navigation/components in `/Users/louishyman/coding/nosql/nosql_reader/app/templates/base.html:29-57`.
5. Add integration tests for route authorization and data-level filtering.

---

## Task 9 - Knowledge Repository Feasibility

### 9.1 Existing PDF ingestion pipeline?
- No dedicated PDF-to-text ingestion pipeline found.
- Current ingestion paths:
  - JSON/TXT file ingest (`/Users/louishyman/coding/nosql/nosql_reader/app/data_processing.py:227-233`)
  - Image-to-JSON ingest (`/Users/louishyman/coding/nosql/nosql_reader/app/image_ingestion.py:74-77`)
- `.pdf` appears in display/filename handling lists but not parsed into chunks by a PDF loader (`/Users/louishyman/coding/nosql/nosql_reader/app/routes.py:2825`).

### 9.2 Can a second Chroma collection be created/queried independently?
- Yes. `get_vector_store(...)` accepts `collection_name` and instantiates `VectorStoreManager` per collection (`/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/vector_store.py:438-455`, `:42-43`).

### 9.3 Current chunking strategy and fit for reference PDFs
- Current chunking uses `RecursiveCharacterTextSplitter` with defaults `chunk_size=1000`, `chunk_overlap=200` and generic separators (`/Users/louishyman/coding/nosql/nosql_reader/app/historian_agent/chunking.py:29-33`, `:133-139`).
- This works for narrative text but may fragment structured lists/tables found in reference glossaries and occupation lists.

### 9.4 Collection management model
- Collections are instantiated ad hoc through `get_vector_store(...)`; no centralized multi-collection manager.

### 9.5 Added token overhead estimate
Assuming reference index retrieval returns top 3 chunks and chunk sizes remain near current defaults:
- ~700-1000 tokens/chunk typical target window.
- Added prompt payload: ~2,100 to 3,000 tokens/query (plus small formatting overhead).
- Practical estimate: **~2.3k-3.3k tokens per query** for reference augmentation.

### 9.6 Feasibility and proposed architecture
Feasible with moderate effort:
1. Add PDF extraction step (new ingest module) -> normalized text docs.
2. Chunk extracted text using a reference-tuned splitter (smaller chunk size for list-heavy docs, heading-aware splitting).
3. Index into separate collection, e.g. `historian_reference_chunks`.
4. Query both corpus collections at runtime and fuse/rerank results.
5. Keep provenance labels so answers can distinguish archival corpus vs reference corpus.

Recommendation on embedding model sharing:
- Prefer sharing the same embedding model as main corpus initially for vector-space compatibility and simpler ops.
- Split models only if evaluation shows domain-specific gains outweigh complexity.

---

## External references used
- [PastPerfect Online export docs](https://pastperfectonline.freshdesk.com/support/solutions/articles/1000298151-how-do-i-export-my-records-from-pastperfect-online-)
- [PastPerfect Museum Software export reports](https://museumsoftware.com/v524.html)
- [PastPerfect object entry tutorial](https://pastperfectmuseumsoftware.freshdesk.com/support/solutions/articles/1000236203-object-entry-tutorial)
- [Chroma docs: SQLite storage internals](https://cookbook.chromadb.dev/core/storage-layout/)
- [SQLite docs: one writer at a time](https://www.sqlite.org/lang_transaction.html)

