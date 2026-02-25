# Historical Document Reader

Historical Document Reader is a Flask + MongoDB research application for archival collections. It supports fielded search, document navigation, historian-oriented chat workflows, and a full entity network analysis module with server-side statistics.

## Current Status

Implemented and live in this codebase:

- Search and browse across `documents` with list-driven next/previous navigation.
- Document detail pages with entity grouping and network context panels.
- Historian Agent routes and configuration flows.
- Network Analysis page (`/network-analysis`) with D3 graph exploration.
- Network document viewer launch/results flow that reuses search-context navigation patterns.
- Network statistics API and UI panel (assortativity, mixing matrix, degree distribution, communities, gatekeepers, random comparison).

Operational status verified on 2026-02-23:

- Master end-to-end suite (`app/util/e2e_cli_suite.py`) passes: 18/18 checks.
- Historian tier endpoints return healthy responses:
  - `POST /historian-agent/query-basic`
  - `POST /historian-agent/query-adversarial`
  - `POST /historian-agent/query-tiered`
  - `POST /historian-agent/query-tier0`

Configuration policy:

- One runtime environment file only: `/Users/louishyman/coding/nosql/nosql_reader/.env`. <!-- Updated path to the main repository root. -->
- Environment examples are documentation-only in `/Users/louishyman/coding/nosql/nosql_reader/docs/ENV_EXAMPLES.md`. <!-- Updated path to the main repository docs directory. -->

## Main Pages

- `/` Home
- `/search` Search Database
- `/document/<doc_id>` Document viewer
- `/historian-agent` Historian Agent
- `/corpus-explorer` Corpus Explorer (Tier 0 exploration UI)
- `/network-analysis` Network Analysis
- `/demographics` Demographics dashboard — workforce statistics and individual browser
- `/network/viewer-results?search_id=<id>` Network-derived document list viewer
- `/settings` Application settings
- `/help` In-app usage guide

## Demographics Notes

- Race values are intentionally sparse: non-`unknown` race appears only when a source explicitly states a racial label.
- Gender extraction is typically higher coverage than race in this corpus, but confidence fields are provided for calibration.
- Occupations preserve raw values in `occupation_primary` and optional fuzzy-normalized values in `occupation_normalized` with scoring metadata.

## Feature Usage

### 1. Database Search

Use this flow when you want direct document retrieval from MongoDB:

1. Open `/search`.
2. Build one or more fielded conditions (field + term + operator).
3. Submit search and open a result.
4. Navigate with Previous/Next in `/document/<doc_id>`; list context is preserved via `search_id`.

Relevant search endpoints:

- `GET /api/searchable-fields` returns fields for search dropdowns.
- `POST /search` executes the query and returns paginated documents plus a `search_id`.
- `GET /document/<doc_id>?search_id=<id>` opens detail view in the same ordered result context.

### 2. Historian Agent

Use this flow for RAG-assisted question answering over the corpus:

1. Open `/historian-agent`.
2. Choose a method (`basic`, `adversarial`, `tiered`, or `tier0`).
3. Submit a research question.
4. Review answer, sources, and metrics; source links open `/document/<id>?search_id=<id>`.

Historian query endpoints:

- `POST /historian-agent/query` (UI-compatible default handler)
- `POST /historian-agent/query-basic`
- `POST /historian-agent/query-adversarial`
- `POST /historian-agent/query-tiered`
- `POST /historian-agent/query-tier0`
- `POST /historian-agent/reset-rag` to reset in-memory handlers

### 3. Corpus Explorer (Tier 0)

Use this flow for exploratory corpus scans and notebook/report artifacts:

1. Open `/corpus-explorer` for guided exploration runs.
2. Run synchronous exploration with `POST /api/rag/explore_corpus`, or async with `POST /api/rag/explore_corpus/start`.
3. Poll async status with `GET /api/rag/explore_corpus/status/<run_id>`.
4. Retrieve outputs using:
   - `GET /api/rag/exploration_report`
   - `GET /api/rag/exploration_notebooks`
   - `POST /api/rag/exploration_notebooks/load`

Common payload fields:

- `strategy`
- `total_budget`
- `year_range` (2-item list/tuple)
- `research_lens` (or alias `focus_areas`)
- `save_notebook`

### 4. Network Analysis

Use this flow for entity-level graph exploration and statistics:

1. Open `/network-analysis`.
2. Apply filters (`type_filter`, `min_weight`, `strict_type_filter`, `person_min_mentions`, `document_term`).
3. Inspect global graph or entity detail context.
4. Launch network-scoped document browsing via `/network/viewer-launch` and continue in `/network/viewer-results?search_id=<id>`.
5. Open any document from the viewer and keep navigation context in `/document/<doc_id>?search_id=<id>`.

## Network Module Overview

### Backend files

- `app/network/config.py` Environment-driven network settings.
- `app/network/build_network_edges.py` Rebuild script for `network_edges`.
- `app/network/network_utils.py` Query/transform helpers.
- `app/network/network_routes.py` Core `/api/network/*` endpoints.
- `app/network/network_statistics.py` Server-side statistics computation.
- `app/network/statistics_routes.py` `/api/network/statistics/*` endpoints.

### Core API endpoints

- `GET /api/network/global`
- `GET /api/network/entity/<entity_id>`
- `GET /api/network/document/<doc_id>`
- `GET /api/network/related/<doc_id>`
- `GET /api/network/metrics/<entity_id>`
- `GET /api/network/stats`
- `GET /api/network/types`

### Statistics API endpoints

- `GET /api/network/statistics/summary`
- `GET /api/network/statistics/graph_summary`
- `GET /api/network/statistics/available_attributes`
- `GET /api/network/statistics/assortativity`
- `GET /api/network/statistics/mixing_matrix`
- `GET /api/network/statistics/degree_distribution`
- `GET /api/network/statistics/communities`
- `GET /api/network/statistics/gatekeepers`
- `GET /api/network/statistics/comparison`

### Derived collections

- `network_edges` Undirected co-occurrence edges between linked entities.
- `network_metrics` Cached per-entity metrics.
- `network_statistics_cache` Cached expensive statistics results.

## Running The App

### Prerequisites

- Docker + Docker Compose plugin
- Root `.env` configured for Mongo + app

### Start

```bash
docker compose -p nosql_reader -f docker-compose.yml up -d --build
```

App URL:

- `http://localhost:5001`

### Stop

```bash
docker compose -p nosql_reader -f docker-compose.yml down
```

## Build / Refresh Network Edges

Run after ingestion changes or when network settings change:

```bash
docker compose -p nosql_reader -f docker-compose.yml exec -T app python -m network.build_network_edges
```

Recommended verification:

```bash
docker compose -p nosql_reader -f docker-compose.yml exec -T app python -c "
from database_setup import get_client, get_db
c=get_client(); db=get_db(c)
print('network_edges:', db['network_edges'].count_documents({}))
print('network_metrics:', db['network_metrics'].count_documents({}))
"
```

## Key Network Query Parameters

Useful filters on `/api/network/global` and related endpoints:

- `type_filter=PERSON,GPE`
- `min_weight=3`
- `limit=500`
- `strict_type_filter=true|false`
- `person_min_mentions=3`
- `document_term=<text>`

## Documentation Map

- `docs/february_how_this_works.md` System architecture, contracts, and execution flow.
- `docs/ENV_EXAMPLES.md` Environment variable examples (documentation only; do not create extra `.env` files).
- `docs/network/NETWORK_ANALYSIS_IMPLEMENTATION_FROM_MAIN.md` Network implementation design notes.
- `app/templates/help.html` In-app user guide content.

## Notes

- Navigation now includes a top-level **Network Analysis** link.
- Document detail pages include clear related-document explanations:
  - Shared entities are direct overlaps with the current document.
  - Connected-via entities are bridge entities that create a path in the network but are not necessarily direct overlaps.
