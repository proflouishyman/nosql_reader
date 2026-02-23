# February 2026: How This Code Works

This document is a detailed, implementation-level guide to the current `main` branch of the Historical Document Reader.

It explains:

1. Runtime architecture and startup sequence.
2. Route and request flow.
3. Search and document-view navigation context.
4. Network analysis backend and frontend behavior.
5. Statistical analysis pipeline.
6. Data model and derived collections.
7. Caching, performance characteristics, and failure modes.
8. How to extend safely.

## 1. Runtime Architecture

The app is a monolithic Flask service (`app/main.py` + `app/routes.py`) backed by MongoDB.

Primary runtime components:

- Flask web application (`app/main.py`)
- Central route module (`app/routes.py`)
- Network module (`app/network/*`)
- Frontend templates and JS (`app/templates/*`, `app/static/js/*`)
- MongoDB collections (`documents`, `linked_entities`, `network_edges`, etc.)

Docker runtime (current compose file):

- Service `app` (container name: `nosql_reader_app`)
- Service `mongodb` (container name: `nosql_reader_mongodb`)
- App exposed at host port `5001` mapped to container `5000`

## 2. Startup Sequence

### 2.1 `app/main.py`

`main.py` does all core app initialization:

1. Creates Flask app object.
2. Loads `config.json` into `app.config['UI_CONFIG']`.
3. Sets session behavior (filesystem session backend).
4. Initializes secret key (env `SECRET_KEY` or generated `secret_key.txt`).
5. Initializes Flask-Caching (`simple` cache backend).
6. Registers context processor that injects:
   - `ui_config`
   - `field_structure`
7. Imports `routes` at the end to avoid circular import.

### 2.2 `app/routes.py`

At import time, `routes.py`:

1. Imports `app` and `cache` from `main.py`.
2. Initializes DB handles from `database_setup.py`.
3. Calls `init_network(app)` from `app/network/__init__.py`.
4. Registers all route functions (search, document viewer, historian endpoints, network pages, help, settings).

This means network API blueprint registration happens as part of normal app route import.

## 3. Request/Route Topology

High-value route groups:

### 3.1 Core UI

- `GET /` home
- `GET /search` search page
- `POST /search` search API workflow
- `GET /document/<doc_id>` detail view with prev/next
- `GET /settings` settings page
- `GET /help` help page

### 3.2 Historian workflows

- `GET /historian-agent`
- `POST /historian-agent/query` and related tier endpoints
- `POST /historian-agent/config`
- `POST /api/rag/explore_corpus`

### 3.3 Network pages

- `GET /network-analysis`
- `GET /network/viewer-launch`
- `GET /network/viewer-results`

### 3.4 Network API blueprint (`/api/network`)

Core graph endpoints:

- `GET /api/network/global`
- `GET /api/network/entity/<entity_id>`
- `GET /api/network/document/<doc_id>`
- `GET /api/network/related/<doc_id>`
- `GET /api/network/metrics/<entity_id>`
- `GET /api/network/stats`
- `GET /api/network/types`

Statistics endpoints:

- `GET /api/network/statistics/summary`
- `GET /api/network/statistics/graph_summary`
- `GET /api/network/statistics/available_attributes`
- `GET /api/network/statistics/assortativity`
- `GET /api/network/statistics/mixing_matrix`
- `GET /api/network/statistics/degree_distribution`
- `GET /api/network/statistics/communities`
- `GET /api/network/statistics/gatekeepers`
- `GET /api/network/statistics/comparison`

## 4. Contracts and Standard Data Formats

This section defines the contract conventions the code currently follows.

### 4.1 Transport and encoding conventions

1. All API routes return JSON via Flask `jsonify`.
2. IDs that originate as Mongo `ObjectId` are serialized to strings before returning to the browser.
3. Datetimes returned in API payloads are ISO-8601 strings (for example from cached metric timestamps).
4. Boolean query parameters accept flexible truthy values:
   - `1`, `true`, `yes`, `y`, `on`
5. Numeric query params are parsed with safe fallback defaults; malformed values do not crash the endpoint.

### 4.2 Error envelope contract

Common error body shape:

```json
{ "error": "Human-readable message" }
```

Common status behaviors:

1. `404` for not-found entity/document in targeted endpoints.
2. `503` when network feature is disabled for endpoints that require it.
3. `500` for unexpected server exceptions.
4. Some stats routes return `200` with an `error` field when data is too small for a statistical test (soft failure style).

### 4.3 Core network API response contracts

`GET /api/network/global`:

```json
{
  "nodes": [
    { "id": "entity_id", "name": "Canonical Name", "type": "PERSON", "mention_count": 12 }
  ],
  "edges": [
    { "source": "id_a", "target": "id_b", "weight": 7, "source_type": "PERSON", "target_type": "ORG" }
  ],
  "stats": {
    "total_edges_matching": 0,
    "edges_after_person_filter": 0,
    "edges_returned": 0,
    "nodes_returned": 0,
    "strict_type_filter": true,
    "person_min_mentions": 3
  }
}
```

`GET /api/network/entity/<entity_id>`:

```json
{
  "entity": { "id": "entity_id", "name": "Name", "type": "PERSON", "mention_count": 20 },
  "edges": [
    { "entity_id": "other_id", "name": "Other", "type": "ORG", "weight": 5, "document_ids": ["doc_id"] }
  ],
  "metrics": { "degree": 10, "weighted_degree": 53, "type_distribution": { "ORG": 4 } }
}
```

`GET /api/network/document/<doc_id>`:

```json
{
  "document_id": "doc_id",
  "filename": "file.json",
  "nodes": [{ "id": "entity_id", "name": "Name", "type": "PERSON", "mention_count": 6 }],
  "edges": [{ "source": "id_a", "target": "id_b", "weight": 3 }]
}
```

`GET /api/network/related/<doc_id>`:

```json
{
  "document_id": "doc_id",
  "related": [
    {
      "document_id": "other_doc_id",
      "filename": "other.json",
      "shared_entity_count": 1,
      "shared_entities": [{ "entity_id": "x", "name": "Baltimore and Ohio Railroad", "type": "ORG" }],
      "network_connection_count": 3,
      "network_connector_entities": [{ "entity_id": "y", "name": "140", "type": "EMPLOYEE_ID" }]
    }
  ]
}
```

`GET /api/network/metrics/<entity_id>`:

```json
{
  "entity_id": "entity_id",
  "entity_name": "Name",
  "entity_type": "PERSON",
  "degree": 42,
  "weighted_degree": 380,
  "top_connections": [],
  "type_distribution": {},
  "computed_at": "2026-02-23T00:00:00.000000"
}
```

`GET /api/network/stats`:

```json
{
  "exists": true,
  "total_edges": 4816,
  "type_pairs": [
    { "source_type": "EMPLOYEE_ID", "target_type": "EMPLOYEE_ID", "count": 3876, "avg_weight": 1.2, "max_weight": 9 }
  ]
}
```

`GET /api/network/types`:

```json
{ "types": ["DATE", "EMPLOYEE_ID", "GPE", "OCCUPATION", "ORG", "PERSON"] }
```

### 4.4 Statistics API contracts

`GET /api/network/statistics/summary` returns a combined report object with top-level keys:

1. `graph_summary`
2. `assortativity`
3. `degree_distribution`
4. `communities`
5. `comparison_to_random`
6. `gatekeepers`
7. `computed_at`
8. `from_cache`

Other statistics endpoints return focused payloads with shapes matching their names, for example:

1. `mixing_matrix` includes `categories`, `observed`, `expected`, `residuals`, `notable_pairs`.
2. `available_attributes` includes `detected_attributes`, `enrichment_attributes`, `all_attributes`.
3. `assortativity` returns `{ "results": [...] }`.

### 4.5 Template/frontend data contracts

Document NER contract used by `document-detail.html`:

- `ner_groups: List[Tuple[str, List[Dict]]]`
- Each entity dict has:
  - `text` (display text)
  - `entity_id` (nullable; used for network interactions)
  - `canonical_name`
  - `type`

Template hooks consumed by JS:

1. `.entity-item[data-entity-id]` for entity popup behavior.
2. `#network-context-graph[data-doc-id]` for per-document graph rendering.
3. `#network-related-docs[data-doc-id]` for related-doc panel rendering.
4. `#network-graph` on `/network-analysis` for the main force graph.
5. `#statistics-panel` for server-side statistics rendering.

### 4.6 Contract stability notes

1. Frontend expects ID fields as strings, not raw `ObjectId`.
2. Network graph nodes require `id`, `name`, `type`; missing those keys breaks rendering logic.
3. Related-document panel expects both direct overlap (`shared_entities`) and bridge context (`network_connector_entities`).
4. Stats responses must remain JSON-safe; numpy scalar and non-string dict key conversion is currently enforced in `statistics_routes.py`.

## 5. Data Model

### 4.1 Primary source collections

- `documents`
  - archive metadata, text fields, optional `entities`, `entity_refs`
- `linked_entities`
  - canonical entities (`_id`, `canonical_name`, `type`, `mention_count`, optional variants/context)

### 4.2 Derived collections used by network

- `network_edges`
  - undirected co-occurrence edges built from `documents.entity_refs`
- `network_metrics`
  - cached per-entity metrics for `/metrics/<entity_id>`
- `network_statistics_cache`
  - cache for expensive stats endpoints (`summary`, `comparison`)

## 6. Search and Viewer Context Mechanics

The app uses a shared list-context pattern for navigation.

### 5.1 Search list context

When you run search:

1. Matching doc IDs are cached under `search_<search_id>`.
2. `GET /document/<doc_id>?search_id=<search_id>` loads current index in that ordered list.
3. Previous/Next links are generated from this list.

### 5.2 Network list context

Network viewer launch intentionally reuses the same pattern:

1. `GET /network/viewer-launch` computes document list from current graph filters.
2. Stores ordered IDs in `search_<search_id>`.
3. Stores network metadata in `network_search_<search_id>`.
4. Redirects to `GET /network/viewer-results?search_id=<id>`.
5. Opening any doc from that page keeps `search_id`, so Previous/Next walks the network-derived list.

This makes network-driven document browsing behave like search-driven browsing.

## 7. Network Edge Build Pipeline

Implemented by `app/network/build_network_edges.py`.

Command:

```bash
docker compose -p nosql_reader -f docker-compose.yml exec -T app python -m network.build_network_edges
```

Pipeline behavior:

1. Load config from env (`NetworkConfig.from_env()`).
2. Build in-memory lookup of allowed linked entities.
3. Scan documents with non-empty `entity_refs`.
4. Normalize refs and generate unique sorted entity pairs per document.
5. Accumulate edge weights and supporting `document_ids`.
6. Drop/rebuild `network_edges` unless dry-run.
7. Write edges >= `NETWORK_MIN_EDGE_WEIGHT`.
8. Create indexes.

Indexes created include:

- unique `(source_id, target_id, edge_type)`
- `source_id`, `target_id`
- `source_type`, `target_type`
- `weight`
- compound `(source_type, target_type, weight desc)`

Edge identity is undirected via sorted pair canonicalization.

## 8. Network Config Model

`app/network/config.py` defines network settings:

- `NETWORK_ANALYSIS_ENABLED`
- `NETWORK_ENTITY_TYPES`
- `NETWORK_MAX_MENTION_COUNT`
- `NETWORK_MIN_EDGE_WEIGHT`
- `NETWORK_BUILD_BATCH_SIZE`
- `NETWORK_DEFAULT_LIMIT`
- `NETWORK_DEFAULT_MIN_WEIGHT`
- `NETWORK_METRICS_CACHE_TTL`
- `NETWORK_MAX_DISPLAY_NODES`

Notable behavior:

- If `NETWORK_ENTITY_TYPES` is blank, types are derived from `linked_entities.distinct("type")` at build/query time.

## 9. Query Semantics in Network APIs

### 8.1 `strict_type_filter`

On `/api/network/global`:

- `true` (default): both endpoints must match selected types.
- `false`: either endpoint can match selected types.

### 8.2 `person_min_mentions`

Applies PERSON-only thresholding:

- PERSON endpoints are filtered out unless `mention_count >= person_min_mentions`.
- Non-PERSON endpoints are unaffected.

### 8.3 `document_term`

Implemented as recursive all-field scan in `network_utils.py`:

- Traverses dict/list/scalar structures from `documents`.
- Case-insensitive containment check.
- Edge survives only if at least one supporting `document_id` matches term.

This is intentionally broad and favors recall over strict field-specific precision.

## 10. Document Detail Network Integration

`routes.py` now builds entity groups as rich dict objects, not plain strings.

Per-entity payload includes:

- `text`
- `entity_id`
- `canonical_name`
- `type`

`document-detail.html` uses `data-entity-id` hooks so `network.js` can:

1. Show popups for entity ego previews.
2. Render document context network section.
3. Render related-documents section.

Related-doc rendering semantics:

- `Shared`: direct overlapping entities between current and related doc.
- `Connected via`: bridge entities supporting network path/connection context.

## 11. Network Analysis Frontend (`network.js`)

`app/static/js/network.js` is the main client controller.

Key modules in file:

- `NetworkAPI`: fetch wrappers for `/api/network/*`
- `NetworkColors`: dynamic type-to-color map
- `EntityPopup`: document-view popups
- `renderForceGraph`: D3 force renderer with zoom/drag/tooltip
- `NetworkExplorer`: page controller for `/network-analysis`

Capabilities in `NetworkExplorer`:

- global graph load
- ego graph load
- type filters
- strict filter toggle
- pair presets and ranking modes
- document-term filtering
- reset behavior
- open network-derived document viewer
- node selection sidebar metrics
- legend + type-pair chips
- centrality-based highlight controls
- whole-network descriptive metrics (client-side on loaded slice)

## 12. Server-Side Statistics Frontend (`network_statistics.js`)

The bottom panel (`partials/network_statistics.html`) delegates to `network_statistics.js`.

Flow:

1. Reads current page filters.
2. Calls `/api/network/statistics/summary` for expensive combined report.
3. Renders sections:
   - Graph summary
   - Assortativity
   - Degree distribution
   - Community structure
   - Random comparison
   - Gatekeepers
4. Mixing matrix is loaded separately on-demand per attribute.

First run can be slower; cached responses are reused.

## 13. Server-Side Statistics Backend

Implemented in:

- `app/network/network_statistics.py` (compute layer)
- `app/network/statistics_routes.py` (API layer)

Computation includes:

- assortativity with permutation tests
- mixing matrix and standardized residuals
- degree distribution, gini, hub detection
- community detection and NMI comparisons
- gatekeeper scoring
- comparison against random/configuration-model graphs

Robustness fixes currently in place:

- JSON-safe conversion for numpy and non-string dict keys
- cache write safety for statistics payloads

## 14. Caching and Performance Notes

### 13.1 Flask cache usage

- `search_<id>` stores ordered doc IDs for list navigation.
- `network_search_<id>` stores network viewer metadata.

### 13.2 Mongo cache usage

- `network_metrics` stores entity metric snapshots.
- `network_statistics_cache` stores expensive statistics results.

### 13.3 Performance-sensitive operations

- `network.build_network_edges` on full corpus
- `/api/network/statistics/summary` and `/comparison`
- global graph requests with low `min_weight` and high `limit`
- `document_term` queries (recursive full-document scans)

## 15. Failure Modes and Debug Checklist

### 14.1 Empty network page

Check:

1. `NETWORK_ANALYSIS_ENABLED` true.
2. `network_edges` exists and has rows.
3. `/api/network/global` returns non-empty edge list for loose filters.

### 14.2 Stats 500 errors

Check:

1. Python dependencies installed (`networkx`, `numpy`, `scipy`, `python-louvain`).
2. JSON-safe serialization path not regressed.
3. Container logs for tracebacks in `statistics_routes.py`.

### 14.3 Viewer navigation broken

Check:

1. `search_id` present on document URL.
2. `search_<id>` cache key exists.
3. TTL not expired.

## 16. Safe Extension Points

### 15.1 Add a network endpoint

1. Add helper logic to `network_utils.py`.
2. Add route function in `network_routes.py` or `statistics_routes.py`.
3. Keep return payload JSON-serializable.
4. Add smoke test via Flask test client.

### 15.2 Add a new frontend control

1. Add control markup in `network-analysis.html`.
2. Read/write state in `NetworkExplorer.getFilterValues()` and reset flow.
3. Pass param into relevant API calls.
4. Update help/tooltips for user clarity.

### 15.3 Add node attribute-based statistics

1. Populate `network_node_attributes` collection.
2. Ensure `load_node_attributes_from_collection()` recognizes attributes.
3. Verify `/statistics/available_attributes` exposes them.
4. Test `mixing_matrix` and assortativity with that attribute.

## 17. Practical Command Reference

Run app stack:

```bash
docker compose -p nosql_reader -f docker-compose.yml up -d --build
```

Rebuild network edges:

```bash
docker compose -p nosql_reader -f docker-compose.yml exec -T app python -m network.build_network_edges
```

API smoke:

```bash
docker compose -p nosql_reader -f docker-compose.yml exec -T app python -c "
from routes import app
c = app.test_client()
for ep in [
    '/api/network/global',
    '/api/network/stats',
    '/api/network/types',
    '/api/network/statistics/graph_summary',
    '/api/network/statistics/summary?n_permutations=10&n_simulations=10',
]:
    r = c.get(ep)
    print(ep, r.status_code)
"
```

## 18. Summary

The codebase now has an integrated search-to-document-to-network workflow:

1. Search builds navigable document lists.
2. Document pages expose entity-level network context.
3. Network Analysis enables global/ego graph exploration with filters.
4. Network-derived document viewers preserve next/previous behavior.
5. Statistical tests add significance-oriented analysis on top of visual exploration.

This document should be updated whenever:

- route contracts change,
- network filters/semantics change,
- statistics payload schemas change,
- or viewer-context mechanics change.
