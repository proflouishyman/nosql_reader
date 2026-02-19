# Network Analysis Feature — Design Document

**Author:** Louis Hyman (design), Claude (implementation spec)
**Date:** February 2026
**Branch:** `tier3_feb` (builds on entity unification work)
**Status:** Ready for implementation

---

## 1. Purpose

Add a network analysis layer to the Historical Document Reader that reveals
relationships between entities extracted from the document corpus. The feature
derives co-occurrence edges from existing entity data, exposes them through API
endpoints, integrates contextual network information into the document detail
view, and provides a standalone interactive network explorer page.

**Domain-agnostic design:** While the current corpus is Baltimore & Ohio Railroad
archives, all code MUST treat entity types, relationship semantics, and display
labels as configuration rather than hard-coded values. Any document collection
with extracted entities and linked entity records should work without code
changes.

---

## 2. Existing Data Contracts

Network analysis is a **read-only derived layer**. It consumes existing
collections and never modifies them.

### 2.1 `documents` collection (source of truth)

```javascript
{
  _id: ObjectId,
  entities: {                           // dict keyed by entity type
    "PERSON": ["George Gerald Jr."],    // arrays of display strings
    "ORG": ["Baltimore and Ohio Railroad"],
    "GPE": ["Pittsburgh"],
    "OCCUPATION": ["Brakeman"],
    "EMPLOYEE_ID": ["205406"],
    "DATE": ["March 15, 1923"]
  },
  entities_extracted: true,
  entities_extracted_at: ISODate,
  entity_refs: ["693c6dbe53c1e5cc78267e80", ...],  // string IDs into linked_entities
  // ... other fields (ocr_text, summary, filename, person_name, etc.)
}
```

**Coverage:** 9,642 / 9,642 documents have `entities` (dict) and `entity_refs`
(non-empty list).

### 2.2 `linked_entities` collection (canonical entity store)

```javascript
{
  _id: ObjectId("693c6dbe53c1e5cc78267e82"),
  canonical_name: "George Gerald Jr.",
  type: "PERSON",                       // matches keys in documents.entities
  variants: ["George Gerald Gray Jr.", "George Aloysius Crudden"],
  document_ids: ["69149fa1...", ...],   // all docs mentioning this entity
  mention_count: 177,
  created_at: ISODate,
  updated_at: ISODate,
  parsed_name: { last: "Jr.", first: "George", middle: "Gerald", full: "..." },
  contexts: [{ document_id: "...", mention: "...", summary: "..." }, ...],
  employee_ids: [...]
}
```

**Count:** 11,423 records. **Type distribution:** EMPLOYEE_ID 8,113 · PERSON
1,089 · GPE 904 · DATE 776 · OCCUPATION 540 · ORG 1.

### 2.3 Join integrity

- `entity_refs` values are Python `str` representations of `linked_entities._id`.
- **44,078** total ref mentions across all documents.
- **100%** resolution rate — zero orphans in either direction.
- **11,423** unique refs = **11,423** linked_entities records (perfect 1:1).

### 2.4 `ner_groups` (routes.py runtime object)

The `build_document_ner_groups` function in `routes.py` merges all entity
sources into a unified display structure. **This function must be enhanced** to
include `linked_entities._id` alongside display text (see §5.1).

---

## 3. New Collections

### 3.1 `network_edges`

Stores pairwise co-occurrence relationships between entities.

```javascript
{
  _id: ObjectId,
  source_id: "693c6dbe53c1e5cc78267e82",  // str(linked_entities._id)
  target_id: "693c6dbe53c1e5cc78267e99",  // str(linked_entities._id)
  source_name: "George Gerald Jr.",         // denormalized for fast reads
  target_name: "W. H. McIntire",
  source_type: "PERSON",
  target_type: "PERSON",
  edge_type: "co_occurrence",              // extensible: "supervisor", "examiner", etc.
  weight: 12,                              // count of shared documents
  document_ids: ["69149fa1...", ...],      // evidence trail (ObjectId strings)
  created_at: ISODate,
  updated_at: ISODate
}
```

**Indexes:**

```python
# Unique compound — one edge per ordered pair
network_edges.create_index(
    [("source_id", ASCENDING), ("target_id", ASCENDING), ("edge_type", ASCENDING)],
    unique=True
)
# Lookup by either endpoint
network_edges.create_index("source_id")
network_edges.create_index("target_id")
# Filtering
network_edges.create_index("source_type")
network_edges.create_index("target_type")
network_edges.create_index("weight")
# Compound for type-filtered queries
network_edges.create_index([("source_type", ASCENDING), ("target_type", ASCENDING), ("weight", DESCENDING)])
```

### 3.2 `network_metrics` (Phase 3 — computed analytics cache)

```javascript
{
  _id: ObjectId,
  entity_id: "693c6dbe53c1e5cc78267e82",  // str(linked_entities._id)
  entity_name: "George Gerald Jr.",
  entity_type: "PERSON",
  degree: 47,                               // number of unique connections
  weighted_degree: 312,                     // sum of edge weights
  top_connections: [                        // precomputed for fast display
    { entity_id: "...", name: "...", type: "...", weight: 12 },
    ...
  ],
  type_distribution: {                      // what types this entity connects to
    "PERSON": 23, "GPE": 10, "OCCUPATION": 8, "ORG": 6
  },
  computed_at: ISODate
}
```

**Index:** unique on `entity_id`.

---

## 4. Module Structure

```
app/network/
├── __init__.py                 # Blueprint registration
├── config.py                   # Network-specific configuration
├── build_network_edges.py      # Batch construction script (Phase 0)
├── network_utils.py            # Query helpers and metrics computation
└── network_routes.py           # Flask Blueprint with API endpoints

app/templates/
├── network-analysis.html       # Standalone network explorer page (Phase 3)
└── partials/
    └── network_context.html    # Includable panel for document-detail (Phase 2)

app/static/js/
└── network.js                  # D3.js visualization + entity hover logic
```

All modules follow existing project conventions:
- Import `database_setup.get_client`, `get_db` for database access
- Use `logging.getLogger(__name__)` with file + console handlers
- Read configuration from environment variables with sensible defaults
- Comprehensive error handling with graceful fallbacks

---

## 5. Implementation Phases

### Phase 0: Edge Construction (`build_network_edges.py`)

**Script type:** Standalone batch script, run inside Docker container.
**Idempotent:** Drops and rebuilds `network_edges` collection on each run.

#### Algorithm

```
1. Load configuration (type filters, frequency thresholds)
2. Drop existing network_edges collection
3. Create indexes
4. Build entity lookup: linked_entities._id → {canonical_name, type, mention_count}
5. For each document in batches of 500:
   a. Read entity_refs array
   b. Resolve each ref to its linked_entities record via lookup
   c. Filter by configured entity types
   d. Filter out entities exceeding max_mention_count threshold
   e. Generate all unique pairs (order-normalized: source_id < target_id)
   f. For each pair: upsert edge (increment weight, append doc_id)
6. Log summary statistics
```

#### Configuration (environment variables)

```bash
# Entity types to include in network (comma-separated)
# Default includes all non-ID types. Domain-agnostic — just list the types
# present in your linked_entities collection.
NETWORK_ENTITY_TYPES=PERSON,ORG,GPE,OCCUPATION

# Exclude entities appearing in more than N documents (removes noise)
# Set high to include everything; lower to filter ubiquitous entities
NETWORK_MAX_MENTION_COUNT=5000

# Minimum edge weight to persist (removes weak/spurious connections)
NETWORK_MIN_EDGE_WEIGHT=2

# Batch size for document processing
NETWORK_BUILD_BATCH_SIZE=500
```

#### Execution

```bash
# Inside Docker container
docker compose exec flask_app python -m app.network.build_network_edges

# Or with overrides
docker compose exec flask_app python -m app.network.build_network_edges \
  --entity-types PERSON,GPE \
  --min-weight 3 \
  --max-mentions 1000
```

#### Edge ordering convention

To avoid duplicate edges (A→B and B→A), normalize pair ordering:
`source_id = min(id_a, id_b)`, `target_id = max(id_a, id_b)`. This is a
string comparison on the hex ObjectId strings, which is stable and
deterministic.

---

### Phase 0.5: `ner_groups` Enhancement (`routes.py` patch)

#### Current state

`build_document_ner_groups` returns `List[Tuple[str, List[str]]]` — entity type
label paired with a list of display strings.

#### Required change

Return `List[Tuple[str, List[Dict[str, str]]]]` where each entity dict contains:

```python
{
    "text": "George Gerald Jr.",       # display string
    "entity_id": "693c6dbe...",        # str(linked_entities._id), or None if unresolved
    "canonical_name": "George Gerald Jr.",  # from linked_entities, or same as text
    "type": "PERSON"                   # entity type
}
```

#### Template change

In `document-detail.html`, the entity list items gain a `data-entity-id`
attribute:

```html
<!-- Before -->
<li>{{ entity_text }}</li>

<!-- After -->
<li class="entity-item"
    {% if entity.entity_id %}
    data-entity-id="{{ entity.entity_id }}"
    data-entity-type="{{ entity.type }}"
    {% endif %}>
  {{ entity.text }}
</li>
```

This is the hook for Phase 2 JavaScript interactions.

#### Implementation notes

- The function already resolves `entity_refs` → `linked_entities` to build the
  unified view. The change is to **retain the `_id`** during that resolution
  instead of discarding it.
- Entities that cannot be resolved to a `linked_entities` record (e.g., from
  legacy `extracted_entities` format only) should have `entity_id: None`.
- **Backward compatible** — templates that don't use `data-entity-id` still work.

---

### Phase 1: API Endpoints (`network_routes.py`)

Flask Blueprint registered in the app factory. All endpoints return JSON.

#### `GET /api/network/entity/<entity_id>`

**Purpose:** Ego network — all edges for one entity.

**Parameters:**
- `entity_id` (path) — `str(linked_entities._id)`
- `type_filter` (query, optional) — comma-separated entity types to include
- `min_weight` (query, optional, default 1) — minimum edge weight
- `limit` (query, optional, default 50) — max edges returned

**Response:**
```json
{
  "entity": {
    "id": "693c6dbe...",
    "name": "George Gerald Jr.",
    "type": "PERSON",
    "mention_count": 177
  },
  "edges": [
    {
      "entity_id": "693c6dbe...",
      "name": "W. H. McIntire",
      "type": "PERSON",
      "weight": 12,
      "document_ids": ["69149fa1...", ...]
    }
  ],
  "metrics": {
    "degree": 47,
    "weighted_degree": 312,
    "type_distribution": {"PERSON": 23, "GPE": 10}
  }
}
```

**Implementation:** Query `network_edges` where `source_id == entity_id OR
target_id == entity_id`. For each edge, return the *other* entity's info.
Sort by weight descending.

#### `GET /api/network/document/<doc_id>`

**Purpose:** Network context for a single document — all entities in this doc
and their cross-document connections.

**Parameters:**
- `doc_id` (path) — `str(documents._id)`
- `type_filter` (query, optional)
- `min_weight` (query, optional, default 1)

**Response:**
```json
{
  "document_id": "69149fa1...",
  "filename": "RDApp-204897Manusco173.jpg.json",
  "nodes": [
    {"id": "693c6dbe...", "name": "George Gerald Jr.", "type": "PERSON", "mention_count": 177}
  ],
  "edges": [
    {"source": "693c6dbe...", "target": "693c6dbf...", "weight": 12}
  ]
}
```

**Implementation:** 
1. Fetch document's `entity_refs`
2. Resolve to `linked_entities` for node info
3. Query `network_edges` for all pairs among these entities
4. Return graph structure ready for D3.js consumption

#### `GET /api/network/global`

**Purpose:** Filtered global network for the explorer page.

**Parameters:**
- `type_filter` (query, optional) — e.g., `PERSON,GPE`
- `min_weight` (query, optional, default 3)
- `limit` (query, optional, default 500) — max edges
- `sort` (query, optional, default `weight`) — sort field

**Response:** Same structure as document endpoint but corpus-wide.

**Implementation:** Query `network_edges` with filters, sorted by weight
descending, limited. Collect unique node IDs from returned edges and fetch
their `linked_entities` info for the `nodes` array.

#### `GET /api/network/metrics/<entity_id>`

**Purpose:** Precomputed or on-demand metrics for a single entity.

**Response:**
```json
{
  "entity_id": "693c6dbe...",
  "name": "George Gerald Jr.",
  "type": "PERSON",
  "degree": 47,
  "weighted_degree": 312,
  "top_connections": [...],
  "type_distribution": {"PERSON": 23, "GPE": 10}
}
```

**Implementation:** Check `network_metrics` cache first. If missing or stale,
compute on-demand from `network_edges` and cache. Use the configurable
`NETWORK_METRICS_CACHE_TTL` (default: 24 hours).

#### Error handling

All endpoints follow existing project patterns:
- Return `{"error": "message"}` with appropriate HTTP status codes
- Log errors with `logger.error(..., exc_info=True)`
- Graceful fallback if `network_edges` collection doesn't exist (return empty
  results with a warning, not a 500)

---

### Phase 2: Document Detail Integration

#### 2a. Entity hover popups

**Trigger:** Mouse hover or click on any `[data-entity-id]` element in the
entity list.

**Behavior:**
1. On first interaction, fetch `/api/network/entity/<id>?limit=5`
2. Cache the response in a JS object keyed by entity_id
3. Display a popup/tooltip showing:
   - Entity name and type
   - Mention count (appears in N documents)
   - Top 5 connections with weights
   - "View full network →" link to `/network-analysis?entity=<id>`

**Implementation:** Pure JavaScript, no framework. Attach event delegation to
the `.entity-group` container. Position popup relative to the hovered element.
Dismiss on mouse-out or click-outside.

#### 2b. Network Context panel

**Location:** New collapsible `<div class="info-section">` in
`document-detail.html`, placed between Extracted Entities and Person Synthesis.

**Behavior:**
1. On page load, if document has `entity_refs`, fetch
   `/api/network/document/<doc_id>?min_weight=2`
2. Render a small D3.js force-directed graph in a fixed-height container
3. Nodes colored by entity type, sized by mention count
4. Edges weighted by co-occurrence strength
5. Click node → navigate to that entity's ego network page
6. Hover node → show name tooltip

**Implementation:** D3.js force simulation. Canvas or SVG depending on node
count (SVG for < 100 nodes, Canvas for more). The graph container should have a
fixed height (e.g., 400px) with scroll/zoom controls.

**Graceful degradation:** If the API returns no edges or the collection doesn't
exist, hide the panel entirely. No error state shown to the user.

#### 2c. Related Documents via Network

**Location:** New section in document-detail, below navigation or in a sidebar.

**Behavior:** For the current document's entities, show up to 10 documents that
share the most entities but are NOT in the same person folder.

**Implementation:** 
1. Get current doc's `entity_refs`
2. Query `network_edges` for the strongest connections of these entities
3. From those connected entities, look up their `linked_entities.document_ids`
4. Exclude documents with the same `person_folder` as the current doc
5. Rank by number of shared entity connections
6. Display as a simple linked list with document filename

---

### Phase 3: Standalone Network Explorer Page

**Route:** `/network-analysis`
**Template:** `network-analysis.html` extending `base.html`

#### Layout

```
┌──────────────────────────────────────────────────┐
│ Network Analysis                        [filters]│
├──────────────────────────────────────────────────┤
│                                                  │
│              D3.js Force Graph                   │
│              (main viewport)                     │
│                                                  │
├────────────┬─────────────────────────────────────┤
│ Selected   │  Entity Detail                      │
│ Entity     │  - Name, type, mention count         │
│ Info       │  - Top connections                   │
│            │  - Sample documents                  │
│            │  - Link to document detail            │
└────────────┴─────────────────────────────────────┘
```

#### Filter controls

- **Entity type checkboxes:** PERSON, ORG, GPE, OCCUPATION (populated
  dynamically from available types in `linked_entities`)
- **Minimum edge weight slider:** 1–50 (adjusts API `min_weight` parameter)
- **Max nodes slider:** 50–500
- **Search box:** Type to find and center on a specific entity

#### Interactions

- **Click node:** Select it, show details in sidebar panel, highlight edges
- **Double-click node:** Navigate to `/network-analysis?entity=<id>` (ego view)
- **Hover node:** Tooltip with name + type
- **Click edge:** Show shared documents in sidebar
- **Zoom/pan:** Standard D3 zoom behavior
- **URL state:** `?entity=<id>&types=PERSON,GPE&min_weight=3` — bookmarkable

#### Ego network mode

When URL has `?entity=<id>`:
- Center graph on that entity
- Show 1-hop connections (and optionally 2-hop via toggle)
- Highlight the focal entity distinctly
- Sidebar shows full metrics for the focal entity

---

## 6. Configuration Reference

All configuration via environment variables, following existing `.env` pattern.

```bash
# === Network Edge Construction ===
# Comma-separated entity types to include
NETWORK_ENTITY_TYPES=PERSON,ORG,GPE,OCCUPATION

# Exclude entities with mention_count above this threshold
NETWORK_MAX_MENTION_COUNT=5000

# Minimum co-occurrence count to persist an edge
NETWORK_MIN_EDGE_WEIGHT=2

# Batch size for processing documents
NETWORK_BUILD_BATCH_SIZE=500

# === Network API ===
# Default limit for global network queries
NETWORK_DEFAULT_LIMIT=500

# Default minimum weight for API queries
NETWORK_DEFAULT_MIN_WEIGHT=1

# Metrics cache TTL in seconds (default 24 hours)
NETWORK_METRICS_CACHE_TTL=86400

# === Network Visualization ===
# Maximum nodes to render in the global explorer
NETWORK_MAX_DISPLAY_NODES=500

# Whether the network analysis feature is enabled
NETWORK_ANALYSIS_ENABLED=true
```

---

## 7. Existing Patterns to Follow

### Database access

```python
# ALWAYS use the shared helpers — never create MongoClient directly
from app.database_setup import get_client, get_db

client = get_client()
db = get_db(client)
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler('network_analysis.log', mode='a')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
```

### Blueprint registration

Follow the pattern used by existing route modules. The network blueprint should
be registered conditionally based on `NETWORK_ANALYSIS_ENABLED`.

### Error handling

```python
try:
    # operation
except Exception as e:
    logger.error(f"Description of what failed: {e}", exc_info=True)
    return jsonify({"error": "User-friendly message"}), 500
```

### Template inheritance

All new templates extend `base.html` and use existing CSS class conventions
(`.info-section`, `.info-container`, `.nav-button`, etc.).

---

## 8. Testing Strategy

### build_network_edges.py

After running the build script, validate with:

```bash
docker compose exec flask_app python -c "
from app.database_setup import get_client, get_db

client = get_client()
db = get_db(client)
ne = db['network_edges']

print(f'Total edges: {ne.count_documents({}):,}')

# Type distribution
from collections import Counter
types = Counter()
for edge in ne.find({}, {'source_type': 1, 'target_type': 1}):
    pair = f\"{edge['source_type']}↔{edge['target_type']}\"
    types[pair] += 1
print(f'Edge type pairs: {dict(types)}')

# Weight distribution
import statistics
weights = [e['weight'] for e in ne.find({}, {'weight': 1})]
if weights:
    print(f'Weight: min={min(weights)} median={statistics.median(weights)} max={max(weights)} mean={statistics.mean(weights):.1f}')

# Top 10 heaviest edges
print(f'\\nTop 10 edges by weight:')
for edge in ne.find().sort('weight', -1).limit(10):
    print(f\"  {edge['source_name']} ↔ {edge['target_name']} (weight={edge['weight']}, type={edge['source_type']}↔{edge['target_type']})\")
"
```

### API endpoints

```bash
# Ego network
curl http://localhost:5000/api/network/entity/<some_entity_id> | python -m json.tool

# Document context
curl http://localhost:5000/api/network/document/<some_doc_id> | python -m json.tool

# Global with filters
curl 'http://localhost:5000/api/network/global?type_filter=PERSON&min_weight=5&limit=20' | python -m json.tool
```

---

## 9. Future Work (v2)

Explicitly out of scope for v1 but designed for:

- **Role-typed edges:** LLM classification of relationship types (supervisor →
  worker, medical_examiner → patient). New `edge_type` values; same collection.
- **Temporal networks:** Date-filtered edges using document date metadata. Time
  slider in the UI.
- **Research notebook integration:** Enrich node attributes with Tier 0 groups
  (occupations, departments, divisions). Node metadata, not structural change.
- **Community detection:** NetworkX clustering algorithms. Results stored in
  `network_metrics` with a `community_id` field.
- **Centrality metrics:** Betweenness, eigenvector centrality. Computed with
  NetworkX, cached in `network_metrics`.
- **Cross-archive networks:** When multiple document collections are loaded,
  edges that span collections become especially interesting.

---

## 10. Implementation Checklist

```
Phase 0 — Edge Construction
  [ ] Create app/network/__init__.py
  [ ] Create app/network/config.py
  [ ] Create app/network/build_network_edges.py
  [ ] Run build script and validate with test queries
  [ ] Add NETWORK_* env vars to .env.example and docker-compose.yml

Phase 0.5 — ner_groups Enhancement
  [ ] Modify build_document_ner_groups in routes.py to include entity_id
  [ ] Update document-detail.html entity rendering to include data-entity-id
  [ ] Verify backward compatibility (entities without linked_entities._id)

Phase 1 — API Endpoints
  [ ] Create app/network/network_utils.py
  [ ] Create app/network/network_routes.py
  [ ] Register blueprint in app factory
  [ ] Test all endpoints with curl

Phase 2 — Document Detail Integration
  [ ] Add entity hover popup JS to static/js/network.js
  [ ] Add network context panel template (partials/network_context.html)
  [ ] Add "Related Documents via Network" section
  [ ] Include D3.js in base.html (CDN or local)

Phase 3 — Standalone Explorer Page
  [ ] Create network-analysis.html template
  [ ] Add /network-analysis route
  [ ] Implement D3.js force-directed graph
  [ ] Add filter controls and entity detail sidebar
  [ ] Add navigation link to base.html nav
```
