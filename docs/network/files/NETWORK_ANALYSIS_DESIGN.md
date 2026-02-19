# Network Analysis Module - Implementation Reference

**Status:** Implemented and active in the Flask app
**Updated:** 2026-02-19

This document describes the network module as it exists in code, including
runtime behavior, API contracts, data model, and UI integration.

## 1. Module scope

The network module provides a derived entity co-occurrence layer over existing
MongoDB data.

- Source collections: `documents`, `linked_entities`
- Derived collections: `network_edges`, `network_metrics`
- API namespace: `/api/network/*`
- UI surfaces:
  - `/network-analysis` (global and ego graph explorer)
  - document detail network context and related documents panels
  - network-derived document list with preserved Previous/Next navigation

The implementation is domain-agnostic: types are data-driven, and filtering
behavior is configurable.

## 2. Code layout

- `app/network/__init__.py`: blueprint registration (`url_prefix=/api/network`)
- `app/network/config.py`: environment-backed `NetworkConfig`
- `app/network/build_network_edges.py`: full rebuild script for `network_edges`
- `app/network/network_utils.py`: query and metrics helpers
- `app/network/network_routes.py`: API endpoints
- `app/static/js/network.js`: graph UI, controls, related docs rendering
- `app/templates/network-analysis.html`: network page
- `app/templates/partials/network_context.html`: document detail partial
- `app/routes.py`: `/network-analysis`, `/network/viewer-*`, and document-viewer wiring

## 3. Data model

### 3.1 `network_edges`

Each document records an undirected co-occurrence edge between two linked
entities (stored as canonical ordered endpoints).

```javascript
{
  source_id: "<linked_entities _id as string>",
  target_id: "<linked_entities _id as string>",
  source_name: "...",
  target_name: "...",
  source_type: "PERSON",
  target_type: "ORG",
  edge_type: "co_occurrence",
  weight: 12,
  document_ids: ["<document _id>", ...],
  created_at: ISODate,
  updated_at: ISODate
}
```

Indexes created by `build_network_edges.py`:

- unique: `(source_id, target_id, edge_type)`
- single: `source_id`, `target_id`, `source_type`, `target_type`, `weight`
- compound: `(source_type, target_type, weight desc)`

### 3.2 Edge identity and directionality

Edges are **undirected** by construction. Pair identity is canonicalized via
sorted entity IDs before writing, so `(A, B)` and `(B, A)` map to one record.

### 3.3 `network_metrics`

Entity-level metrics cache used by `/api/network/metrics/<entity_id>`.
Computed lazily and refreshed by TTL.

## 4. Configuration

`NetworkConfig` reads from environment variables.

- `NETWORK_ANALYSIS_ENABLED` (default `true`)
- `NETWORK_ENTITY_TYPES` (comma-separated; blank means derive from `linked_entities.distinct("type")`)
- `NETWORK_MAX_MENTION_COUNT` (default `5000`)
- `NETWORK_MIN_EDGE_WEIGHT` (default `2`)
- `NETWORK_BUILD_BATCH_SIZE` (default `500`)
- `NETWORK_DEFAULT_LIMIT` (default `500`)
- `NETWORK_DEFAULT_MIN_WEIGHT` (default `3`)
- `NETWORK_METRICS_CACHE_TTL` (default `86400`)
- `NETWORK_MAX_DISPLAY_NODES` (default `500`)

## 5. Edge build and rebuild

Full rebuild command (inside current compose stack):

```bash
docker compose -f /Users/louishyman/coding/nosql/nosql_reader/docker-compose.yml exec -T app python -m network.build_network_edges
```

Dry-run:

```bash
docker compose -f /Users/louishyman/coding/nosql/nosql_reader/docker-compose.yml exec -T app python -m network.build_network_edges --dry-run
```

Optional overrides:

```bash
docker compose -f /Users/louishyman/coding/nosql/nosql_reader/docker-compose.yml exec -T app python -m network.build_network_edges --entity-types PERSON,GPE --min-weight 3 --max-mentions 1000
```

Build behavior summary:

1. Build an in-memory lookup from `linked_entities` with type and mention caps.
2. Scan `documents.entity_refs` in batches.
3. Generate ordered pairs per document.
4. Accumulate weight + supporting `document_ids`.
5. Filter edges below configured minimum weight.
6. Write edges in bulk and create indexes.

## 6. API contract

Base path: `/api/network`

### 6.1 Endpoints

- `GET /entity/<entity_id>`: ego network for one entity
- `GET /document/<doc_id>`: network context for a document
- `GET /global`: filtered global graph
- `GET /metrics/<entity_id>`: degree/weighted degree/top connections
- `GET /related/<doc_id>`: related documents via shared + one-hop connector entities
- `GET /stats`: edge count and type-pair summary
- `GET /types`: distinct entity types

### 6.2 Important query parameters

`/global`:

- `type_filter=PERSON,GPE`
- `min_weight=<int>` (default from `NETWORK_DEFAULT_MIN_WEIGHT`, normally `3`)
- `limit=<int>` (default from `NETWORK_DEFAULT_LIMIT`, normally `500`)
- `strict_type_filter=true|false` (default `true`)
- `document_term=<text>`: matched across **all document fields**

Strict type semantics:

- `true`: both endpoints must be in `type_filter`
- `false`: either endpoint can be in `type_filter`

`/entity/<id>` supports `type_filter`, `min_weight`, `limit`, `document_term`.

`/related/<doc_id>` supports `limit`.

### 6.3 Feature-flag disabled behavior

- `/global`, `/stats`, `/types` return valid empty/disabled payloads.
- `/entity`, `/document`, `/metrics`, `/related` return `503` with error JSON.

## 7. Route integration in Flask app

### 7.1 Network page and viewer routes

In `app/routes.py`:

- `/network-analysis`: standalone network explorer page
- `/network/viewer-launch`: builds network-derived document list context
- `/network/viewer-results`: paginated list view tied to cached search context

Viewer context settings:

- cache timeout: `3600` seconds
- max unique documents: `2000`
- max document IDs consumed per edge: `250`

This preserves Previous/Next navigation in document detail using the same
`search_<id>` cache pattern used by search results.

### 7.2 Document detail entity contract

`build_document_ner_groups` now returns:

```python
List[Tuple[str, List[Dict[str, Any]]]]
```

Each entity item includes:

- `text`
- `entity_id` (when resolvable through `entity_refs`)
- `canonical_name`
- `type`

`document-detail.html` uses `data-entity-id` attributes for popup and network
interactions.

## 8. UI behavior

### 8.1 `/network-analysis`

Implemented controls:

- type checkboxes
- strict type filter
- pair preset (`cross_type_only`, `all_pairs`, `within_type_only`)
- ranking mode (`most_connected`, `most_cross_type`, `rare_but_strong`, `low_frequency_pairs`)
- research template shortcuts
- min edge weight slider
- max edges slider
- document-term filter (`Document contains`)
- reset button
- open document viewer button

Stats type-pair chips are clickable and can focus one type-pair slice.

### 8.2 Document view integration

`network.js` initializes on document pages and provides:

- click popup for entities with `data-entity-id`
- inline document network graph panel
- related documents panel with two distinct signals:
  - `shared_entities`: direct overlap with current document
  - `network_connector_entities`: one-hop bridge entities that explain relatedness

## 9. Validation checklist (CLI)

### 9.1 Basic API smoke

```bash
docker compose -f /Users/louishyman/coding/nosql/nosql_reader/docker-compose.yml exec -T app python - <<'PY'
from routes import app
c = app.test_client()
for path in [
    '/api/network/types',
    '/api/network/stats',
    '/api/network/global',
]:
    r = c.get(path)
    print(path, r.status_code)
PY
```

### 9.2 Index and edge count check

```bash
docker compose -f /Users/louishyman/coding/nosql/nosql_reader/docker-compose.yml exec -T app python - <<'PY'
from database_setup import get_client, get_db
c = get_client(); db = get_db(c)
coll = db['network_edges']
print('edges:', coll.count_documents({}))
print('indexes:', list(coll.index_information().keys()))
print('top edge:', coll.find_one(sort=[('weight', -1)]))
PY
```

## 10. Known constraints

- `document_term` filtering recursively scans full document payloads and can
  increase latency on large edge sets.
- Related-document discovery caps expansion (`connected_ids[:100]`,
  `document_ids[:200]` per connector entity) to keep runtime bounded.
- When `network_edges` is absent, the UI returns graceful empty states and
  guidance to rebuild edges.
