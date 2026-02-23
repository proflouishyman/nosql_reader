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

## Main Pages

- `/` Home
- `/search` Search Database
- `/document/<doc_id>` Document viewer
- `/historian-agent` Historian Agent
- `/network-analysis` Network Analysis
- `/network/viewer-results?search_id=<id>` Network-derived document list viewer
- `/settings` Application settings
- `/help` In-app usage guide

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
- `.env` configured for Mongo + app

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
- `docs/network/NETWORK_ANALYSIS_IMPLEMENTATION_FROM_MAIN.md` Network implementation design notes.
- `app/templates/help.html` In-app user guide content.

## Notes

- Navigation now includes a top-level **Network Analysis** link.
- Document detail pages include clear related-document explanations:
  - Shared entities are direct overlaps with the current document.
  - Connected-via entities are bridge entities that create a path in the network but are not necessarily direct overlaps.
