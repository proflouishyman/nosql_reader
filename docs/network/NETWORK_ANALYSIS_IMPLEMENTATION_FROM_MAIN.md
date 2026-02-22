# Network Analysis Implementation Design (From `main` Branch, From Scratch)

## 1. Objective

Implement a full network analysis capability in the Historical Document Reader starting from a baseline `main` branch that does not yet contain network-specific backend routes, derived collections, or UI pages.

The implementation must deliver:

1. A rebuildable entity co-occurrence graph in MongoDB.
2. A stable API under `/api/network/*`.
3. Document-level network context and related-document discovery.
4. A standalone `/network-analysis` page with filtering, graph exploration, document viewer launch, and bottom analytics panel.
5. Node-level highlighting by centrality and whole-network metrics for the active graph slice.

## 2. Scope

### 2.1 In scope

1. New backend module `app/network/`.
2. New derived collections `network_edges` and `network_metrics`.
3. New batch build script for edge construction.
4. New network API endpoints.
5. Route integration for `/network-analysis`, `/network/viewer-launch`, and `/network/viewer-results`.
6. Document detail enhancements for entity IDs and network context panels.
7. Network UI controls including `person_min_mentions` (default `3`).
8. Bottom analytics panel (whole-network metrics and node-level centrality controls).
9. CLI and test-client checks to verify backend and rendering behavior.

### 2.2 Out of scope

1. Replacing current NER/entity-linking pipelines.
2. Multi-hop path explanation APIs beyond existing related-doc heuristic.
3. Persisting user-specific saved graph views.
4. Full graph analytics in MongoDB (analytics are computed in-browser per current slice).

## 3. Baseline assumptions on `main`

1. `documents` collection contains `entity_refs` and/or `entities`.
2. `linked_entities` collection exists with canonical entities and `type`.
3. Existing app route structure is monolithic (`app/routes.py`).
4. Existing document detail page can render NER groups but does not include network metadata.
5. Existing search result navigation uses cached list context (`search_<id>` pattern).

## 4. Data contracts

### 4.1 Source collections

1. `documents`
   - `_id`
   - `filename`
   - `entity_refs` (list of linked entity IDs as strings or ObjectIds)
   - optional `entities` dictionary by type
2. `linked_entities`
   - `_id`
   - `canonical_name`
   - `type`
   - `mention_count`
   - optional `document_ids`, `variants`, `contexts`

### 4.2 Derived collection: `network_edges`

Each row is an undirected co-occurrence edge represented as a canonical ordered pair.

```javascript
{
  source_id: "<linked_entity_id>",
  target_id: "<linked_entity_id>",
  source_name: "...",
  target_name: "...",
  source_type: "PERSON",
  target_type: "ORG",
  edge_type: "co_occurrence",
  weight: 12,
  document_ids: ["<doc_id>", ...],
  created_at: ISODate,
  updated_at: ISODate
}
```

Edge identity rule:

1. Canonicalize with sorted endpoint IDs before write.
2. Unique index on `(source_id, target_id, edge_type)`.

### 4.3 Derived collection: `network_metrics`

Entity-level metrics cache keyed by `entity_id`.

```javascript
{
  entity_id: "<linked_entity_id>",
  entity_name: "...",
  entity_type: "PERSON",
  degree: 42,
  weighted_degree: 380,
  top_connections: [...],
  type_distribution: {...},
  computed_at: ISODate
}
```

## 5. Configuration and feature flags

Read from environment with defaults:

1. `NETWORK_ANALYSIS_ENABLED=true`
2. `NETWORK_ENTITY_TYPES=` (blank means derive from `linked_entities.distinct("type")`)
3. `NETWORK_MAX_MENTION_COUNT=5000`
4. `NETWORK_MIN_EDGE_WEIGHT=2`
5. `NETWORK_BUILD_BATCH_SIZE=500`
6. `NETWORK_DEFAULT_LIMIT=500`
7. `NETWORK_DEFAULT_MIN_WEIGHT=3`
8. `NETWORK_METRICS_CACHE_TTL=86400`
9. `NETWORK_MAX_DISPLAY_NODES=500`

Runtime query default:

1. `person_min_mentions=3` for global graph and viewer-derived document lists.

## 6. Backend architecture

### 6.1 New files

1. `app/network/__init__.py`
2. `app/network/config.py`
3. `app/network/build_network_edges.py`
4. `app/network/network_utils.py`
5. `app/network/network_routes.py`

### 6.2 Route integration in existing app

1. Register network blueprint through `init_network(app)` in `app/routes.py` (or app factory if present).
2. Add page route:
   - `GET /network-analysis`
3. Add viewer routes:
   - `GET /network/viewer-launch`
   - `GET /network/viewer-results`

### 6.3 API endpoints

All under `/api/network`:

1. `GET /entity/<entity_id>`
2. `GET /document/<doc_id>`
3. `GET /global`
4. `GET /metrics/<entity_id>`
5. `GET /related/<doc_id>`
6. `GET /stats`
7. `GET /types`

### 6.4 Query semantics

`/global` parameters:

1. `type_filter=PERSON,GPE`
2. `min_weight=<int>`
3. `limit=<int>`
4. `strict_type_filter=true|false`
5. `person_min_mentions=<int>`
6. `document_term=<text>`

Strict type behavior:

1. `true`: both endpoints must match selected types.
2. `false`: either endpoint can match selected types.

Person threshold behavior:

1. For any edge endpoint with `type=PERSON`, require that endpoint entity `mention_count >= person_min_mentions`.
2. Non-PERSON endpoints are not filtered by this rule.

### 6.5 Graceful disabled mode

When `NETWORK_ANALYSIS_ENABLED=false`:

1. `/global`, `/stats`, `/types` return valid empty/disabled payloads.
2. `/entity`, `/document`, `/metrics`, `/related` return controlled 503 JSON.
3. `/network-analysis` page still renders with a friendly unavailable message.

## 7. Edge build algorithm

Implemented in `python -m network.build_network_edges`.

1. Build an in-memory lookup from `linked_entities` applying type and mention filters.
2. Stream `documents` with non-empty `entity_refs` in configured batches.
3. Normalize refs to valid lookup IDs.
4. Generate unique ordered pairs per document.
5. Accumulate `weight` and supporting `document_ids` in memory.
6. Filter out pairs below `NETWORK_MIN_EDGE_WEIGHT`.
7. Bulk upsert to `network_edges`.
8. Create indexes required by query patterns.

### 7.1 Index strategy

1. Unique pair key: `(source_id, target_id, edge_type)`.
2. Endpoint lookup: `source_id`, `target_id`.
3. Type filtering: `source_type`, `target_type`.
4. Weight ranking: `weight`.
5. Compound common query path: `(source_type, target_type, weight desc)`.

## 8. UI architecture

### 8.1 New/updated files

1. `app/templates/network-analysis.html`
2. `app/static/js/network.js`
3. `app/templates/partials/network_context.html`
4. `app/templates/document-detail.html`
5. `app/templates/base.html` (navigation link)

### 8.2 Network Analysis page behavior

Top controls:

1. Entity type filters.
2. Strict type match toggle.
3. Pair preset (`cross_type_only`, `all_pairs`, `within_type_only`).
4. Ranking mode (`most_connected`, `most_cross_type`, `rare_but_strong`, `low_frequency_pairs`).
5. Research template selector.
6. Min edge weight slider.
7. Max edges slider.
8. Document term text filter.
9. Person tie minimum documents (`person_min_mentions`, default `3`).
10. Reset button.
11. Open Document Viewer button.

Bottom analytics panel:

1. Whole-network metrics for current graph slice:
   - node count
   - edge count
   - density
   - average path length
   - mean harmonic centrality
   - average clustering
   - modularity (label-propagation partition)
   - transitivity
   - connected component count
   - largest component size
2. Directed-only placeholders:
   - reciprocity
   - asymmetry
   - displayed as `N/A` while graph remains undirected
3. Node-level centrality controls:
   - metric selector (`harmonic`, `degree`, `weighted_degree`, `betweenness`)
   - top-N highlight slider
4. Node-level ranking list and visual highlighting in graph.

### 8.3 Document integration

1. `build_document_ner_groups` must expose entity dictionaries with:
   - `text`
   - `entity_id`
   - `canonical_name`
   - `type`
2. Render entity list items with `data-entity-id` for network popups.
3. Include network context partial:
   - document graph panel
   - related documents panel
4. Related docs must clearly separate:
   - direct shared entities
   - one-hop connector entities (`Connected via`)

### 8.4 Network-derived document viewer

1. `Open Document Viewer` launches `/network/viewer-launch` with active filters.
2. Route builds deterministic ordered document list from selected graph edges.
3. Save ordered IDs in cache using `search_<id>` pattern.
4. `/network/viewer-results` renders list view with paging.
5. Clicking a doc opens standard detail page using same search context so Previous/Next is list-consistent.

## 9. Metrics method notes

Since analytics are computed client-side per filtered slice, define method clarity:

1. Average path length:
   - shortest-path mean over connected pairs in largest component.
2. Harmonic centrality:
   - per-node sum of inverse distances, normalized by `N-1`.
3. Clustering:
   - mean local clustering coefficient.
4. Transitivity:
   - global triangle/triple ratio.
5. Modularity:
   - score computed on label-propagation communities.
6. Betweenness centrality:
   - unweighted Brandes implementation on loaded slice.

## 10. Performance constraints and guardrails

1. Use server-side edge limits to cap payload size (`limit`, max configured bound).
2. Keep in-browser analytics bounded by practical max nodes/edges from UI controls.
3. Cache entity metrics in `network_metrics` for repeated sidebar loads.
4. Cap related-doc expansion breadth/depth to prevent expensive fanout queries.
5. Use `person_min_mentions` to suppress noisy low-support person ties.

## 11. Testing and validation plan

### 11.1 Build and data checks

1. Run full edge rebuild script.
2. Assert `network_edges.count_documents({}) > 0`.
3. Assert expected indexes exist.
4. Assert top-weight query returns rows.

### 11.2 API smoke checks

1. `/api/network/types`, `/stats`, `/global` return 200 and valid JSON.
2. Dynamic IDs for `/entity/<id>`, `/document/<id>`, `/metrics/<id>`, `/related/<doc_id>`.
3. Invalid IDs return controlled 4xx/5xx contract responses.

### 11.3 Contract checks

1. `ner_groups` structure renders without template exceptions.
2. `/document/<id>` contains `data-entity-id` for linked entities.
3. Related-doc panel displays `Shared` and `Connected via` data clearly.

### 11.4 UI route checks

1. `/network-analysis` returns 200 and includes `#network-graph` and bottom analytics panel IDs.
2. `/static/js/network.js` returns 200.
3. `/network/viewer-launch` flows to `/network/viewer-results` and preserves navigation context.

### 11.5 Frontend static checks

1. Validate browser scripts with `node --check`.
2. Manual visual check for graph render, control updates, and highlight behavior.

## 12. Rollout strategy

### Phase A: backend foundation

1. Add network module, config, build script, and API endpoints.
2. Build `network_edges` and verify indexes and query behavior.

### Phase B: document integration

1. Add entity-id aware `ner_groups` contract.
2. Add network context and related-doc panels in document detail.

### Phase C: standalone network page

1. Add `/network-analysis` page with graph controls.
2. Add viewer launch/list flow tied to cached search context.

### Phase D: analytics enhancements

1. Add person tie threshold control and API propagation.
2. Add bottom-panel whole-network metrics and node-centrality highlighting.

### Phase E: hardening and docs

1. Add fallback behavior for disabled/empty network mode.
2. Finalize in-app Help and project docs with exact runtime semantics.

## 13. Acceptance criteria

1. `network_edges` is rebuildable, indexed, and queryable.
2. All `/api/network/*` endpoints satisfy documented contracts.
3. `/network-analysis` supports filtering, person threshold, and bottom analytics panel.
4. Node highlighting by selected centrality metric works on active graph slice.
5. Document detail page shows network context and related docs with explicit shared/connector semantics.
6. Network-derived document viewer preserves Previous/Next sequence consistency.
7. Feature-disabled state remains non-breaking for core app pages.
