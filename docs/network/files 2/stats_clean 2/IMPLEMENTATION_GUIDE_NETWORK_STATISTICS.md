# Network Statistics Extension — Implementation Guide

**Extends:** Network Analysis module (active in `tier3_feb`)
**Phase:** Phase E / v2 statistical layer
**Status:** Ready for Claude Code implementation

---

## 1. What this extension does

The existing network analysis page computes descriptive analytics
client-side on the loaded graph slice: density, clustering, modularity,
transitivity, harmonic centrality, betweenness, connected components. These
are valuable but answer only "what does this slice look like?"

This extension adds **server-side statistical hypothesis tests** that
answer the harder question: **"Are the patterns we see real, or artifacts
of the archive's structure?"** The methods are adapted from Exponential
Random Graph Modeling (ERGM) methodology — the same framework used in
computational sociology to analyze tie formation in social networks.

### 1.1 ERGM concepts mapped to Python

ERGM (implemented in R's `statnet` package) fits probability models to
observed networks. The core intellectual moves are:

| ERGM concept | R term | What it tests | Python equivalent in this extension |
|---|---|---|---|
| **Edge baseline** | `edges` | How many ties exist? | Graph summary (node/edge counts) |
| **Homophily** | `nodematch("attr")` | Do same-attribute nodes connect more than chance? | `compute_assortativity()` + permutation test |
| **Cross-group mixing** | `nodemix("attr")` | Which cross-category pairings are over/under-represented? | `compute_mixing_matrix()` + standardized residuals |
| **Degree correction** | `degree(1)` | Control for nodes with atypical connectivity | `compute_degree_distribution()` + Gini + power-law fit |
| **Simulation comparison** | `simulate()` + `gof()` | Does the observed network differ from random graphs with same properties? | `compare_to_random_graphs()` via configuration model |
| **Attribute-driven effects** | `absdiff("attr")` | Do continuous attribute differences predict tie formation? | Future: numeric attribute tests |

We don't need full ERGM estimation (which requires MCMC convergence)
because **permutation testing gives ~90% of ERGM's value with ~10% of the
complexity** for our use case. The key insight: take the observed
statistic, shuffle node attributes 1,000 times, recompute each time. If
the observed value falls outside the 95th percentile of the shuffled
distribution, the pattern is statistically significant.

### 1.2 Historical research questions this enables

For the B&O Railroad archives, the parallel questions are:

- **Occupational homophily**: Do workers of the same occupation co-appear
  in documents more than expected? If yes → the bureaucratic system tracks
  people by job category. If no → the system mixes across occupations.
- **Departmental siloing**: Is the M.P. Department documentary world
  separate from the C.T. Department? Mixing matrices show this directly.
- **National origin patterns**: Do Italian-American workers cluster in the
  network? The research notebook already flags national origin groups. A
  permutation test would tell you if that's beyond chance.
- **Gatekeeper identification**: Which entities bridge otherwise separate
  communities *relative to their degree*? These are institutional power
  nodes — medical examiners, clerks, officials.
- **Archive vs. reality**: Is the high clustering in the network a real
  institutional pattern, or just what the degree sequence alone would
  produce? Configuration model comparison answers this.

The edge semantics are "co-appear in archival documents" — measuring
**bureaucratic proximity**, not personal relationships. What the
institution brought into documentary contact with whom, and whether that
contact follows occupational, departmental, or ethnic lines.

---

## 2. Prerequisite: existing network module

This extension requires the network analysis module to be fully
operational. Specifically:

- `network_edges` collection must exist and be populated
- `network_bp` blueprint must be registered at `/api/network`
- `network.config.NetworkConfig` must be importable
- The `/network-analysis` page must render

Verify with:

```bash
docker compose -f /Users/louishyman/coding/nosql/nosql_reader/docker-compose.yml \
  exec -T app python -c "
from database_setup import get_client, get_db
c = get_client(); db = get_db(c)
print('edges:', db['network_edges'].count_documents({}))
print('types:', db['linked_entities'].distinct('type'))
"
```

---

## 3. What's new vs. what already exists

### Already computed CLIENT-SIDE (in `network.js` bottom analytics panel)

These run in the browser on the loaded graph slice:

- Node count, edge count, density
- Average path length
- Mean harmonic centrality
- Average clustering coefficient
- Modularity (label-propagation partition)
- Transitivity (global triangle ratio)
- Connected component count, largest component size
- Node-level centrality rankings (harmonic, degree, weighted_degree, betweenness)

### New SERVER-SIDE (this extension)

These run on the full network stored in MongoDB and provide **significance
testing** — something the client-side panel cannot do:

| Analysis | What it adds beyond client-side |
|---|---|
| Assortativity + permutation test | *p*-value proving homophily is real, not just descriptive coefficient |
| Mixing matrix + residuals | Identifies specific overrepresented cross-category pairings |
| Degree distribution + Gini | Inequality measure, power-law fit, systematic hub identification |
| Configuration model comparison | Proves clustering/modularity are significant vs. random expectation |
| Community detection + NMI | Measures how well entity types predict community membership |
| Gatekeeper analysis | Bridge score (betweenness/log(degree)) identifies structural brokers |

The statistics panel renders **below** the existing bottom analytics
panel on the `/network-analysis` page.

---

## 4. File inventory

### 4.1 New files to ADD (4 files)

| File | Lines | Description |
|------|-------|-------------|
| `app/network/network_statistics.py` | ~650 | Core computation module |
| `app/network/statistics_routes.py` | ~280 | API endpoints on existing blueprint |
| `app/static/js/network_statistics.js` | ~590 | Frontend rendering |
| `app/templates/partials/network_statistics.html` | ~330 | HTML partial + CSS |

### 4.2 Existing files to MODIFY (3 small changes)

| File | Change | Lines affected |
|------|--------|----------------|
| `app/network/__init__.py` | Add 1 import line | 1 |
| `app/templates/network-analysis.html` | Add `{% include %}` + `<script>` tag | 2 |
| `requirements.txt` | Add 3–4 packages | 3–4 |

---

## 5. Critical integration notes

### 5.1 Import conventions

**This project does NOT use `app.` prefix for imports inside the
container.** The working directory is `/app` and modules are imported
directly:

```python
# CORRECT:
from database_setup import get_client, get_db
from network import network_bp
from network.config import NetworkConfig
from network.network_statistics import compute_assortativity

# WRONG — do NOT use these:
from app.database_setup import get_client, get_db
from app.network import network_bp
```

### 5.2 Docker compose exec

The reference doc uses the full compose file path and service name `app`:

```bash
docker compose -f /Users/louishyman/coding/nosql/nosql_reader/docker-compose.yml \
  exec -T app python -m network.build_network_edges
```

**Service name is `app`**, container name is `flask_app`.

### 5.3 Test client pattern

```python
from routes import app
c = app.test_client()
r = c.get('/api/network/statistics/graph_summary')
print(r.status_code, r.get_json())
```

### 5.4 Default parameter values

These must match the existing network module exactly:

| Parameter | Default | Source |
|-----------|---------|--------|
| `min_weight` | `3` | `NETWORK_DEFAULT_MIN_WEIGHT` |
| `person_min_mentions` | `3` | Runtime query default |
| `limit` | `500` | `NETWORK_DEFAULT_LIMIT` |
| `strict_type_filter` | `true` | API default |

### 5.5 Existing client-side analytics — do NOT duplicate

The statistics extension must not re-render values already shown in the
bottom analytics panel (density, clustering, modularity, etc.). Instead
it provides **significance tests** that reference those values but add
statistical context ("clustering = 0.31, significantly higher than random
expectation of 0.04, z = 8.2").

---

## 6. Implementation steps

### Step 1: Add Python dependencies

Add to `requirements.txt` if not already present:

```
networkx>=3.0
numpy>=1.24
scipy>=1.10
python-louvain>=0.16
```

Then either rebuild the container or:

```bash
docker compose -f /Users/louishyman/coding/nosql/nosql_reader/docker-compose.yml \
  exec -T app pip install networkx numpy scipy python-louvain --break-system-packages
```

### Step 2: Copy new files into place

```
app/network/network_statistics.py
app/network/statistics_routes.py
app/static/js/network_statistics.js
app/templates/partials/network_statistics.html
```

### Step 3: Register statistics routes

In `app/network/__init__.py`, add **one line** after the existing route
import:

```python
# Existing line (already present):
from network import network_routes  # noqa: F401, E402

# ADD this line immediately after:
from network import statistics_routes  # noqa: F401, E402
```

This registers all `/api/network/statistics/*` endpoints on the same
`network_bp` blueprint. No new blueprint needed.

### Step 4: Include statistics panel in the network page

In `app/templates/network-analysis.html`, add two things:

**A) Include the HTML partial.** Place this AFTER the existing bottom
analytics panel (after the `analytics-panel` div), before the final
closing `</div>` of the page container:

```html
  {# ... existing analytics panel above ... #}

  {% include 'partials/network_statistics.html' %}

</div> {# end page container #}
```

**B) Add the JS file.** In the `{% block scripts %}` section, after the
existing `network.js` script tag:

```html
<script src="{{ url_for('static', filename='js/network_statistics.js') }}"></script>
```

### Step 5: Verify

See validation section below.

---

## 7. New API endpoints

All register on the existing `/api/network` blueprint.

### 7.1 Endpoint reference

| Endpoint | Cost | Cache? | Description |
|----------|------|--------|-------------|
| `GET /statistics/summary` | High (~45s) | Yes | Full statistical report |
| `GET /statistics/assortativity?attribute=type` | Medium (~5s/attr) | No | Permutation-tested homophily |
| `GET /statistics/mixing_matrix?attribute=type` | Low (~1s) | No | Cross-category residuals |
| `GET /statistics/degree_distribution` | Low (~1s) | No | Gini, hubs, power-law |
| `GET /statistics/communities` | Medium (~3s) | No | Louvain + NMI |
| `GET /statistics/gatekeepers?limit=20` | Medium (~3s) | No | Bridge score ranking |
| `GET /statistics/comparison` | High (~20s) | Yes | Config model comparison |
| `GET /statistics/graph_summary` | Very low | No | Node/edge counts |
| `GET /statistics/available_attributes` | Very low | No | Testable attributes |

### 7.2 Common query parameters

All endpoints accept:

- `type_filter=PERSON,GPE` — comma-separated, matches `/global` semantics
- `min_weight=<int>` — default `3` (from `NETWORK_DEFAULT_MIN_WEIGHT`)
- `person_min_mentions=<int>` — default `3`

Additional for `/statistics/summary`:

- `n_permutations=<int>` — default `1000` (reduce to 500 for faster testing)
- `n_simulations=<int>` — default `100` (reduce to 50 for faster testing)
- `force=true` — bypass cache

### 7.3 Cache behavior

`/statistics/summary` and `/statistics/comparison` store results in a new
`network_statistics_cache` collection with TTL following
`NETWORK_METRICS_CACHE_TTL` (default 86400s = 24h). Cache key includes
`type_filter`, `min_weight`, and `person_min_mentions` so different
filter combinations get separate caches.

---

## 8. Module architecture

### 8.1 `network_statistics.py` — computation layer

Two distinct layers in one file:

**Data loading (top)**:
- `build_networkx_graph(db, ...)` — reads `network_edges` and
  `linked_entities`, respects `person_min_mentions`
- `load_node_attributes_from_collection(db, G)` — loads enrichment
  attributes from optional `network_node_attributes` collection

**Pure computation (everything else)**:
- Accepts `nx.Graph`, returns `dict` suitable for JSON serialization
- No Flask or MongoDB dependencies in computation functions
- Independently unit-testable

### 8.2 `statistics_routes.py` — API layer

- Registers on existing `network_bp`
- Each endpoint: parse params → build graph → call computation → return JSON
- Uses `_build_graph()` helper that honors `person_min_mentions` and
  `min_weight` defaults matching existing `/global` endpoint

### 8.3 Graceful degradation

If `network_edges` doesn't exist or graph is too small:
- API returns `{"error": "..."}` with specific message
- Individual test failures don't block other tests in `/summary`
- If `python-louvain` isn't installed, falls back to networkx's
  `greedy_modularity_communities`
- If `scipy` isn't installed, power-law test is skipped
- If `network_node_attributes` collection doesn't exist, only `type`
  attribute is tested (always available from `linked_entities`)

---

## 9. Statistical methods — detailed reference

### 9.1 Assortativity + permutation test (ERGM `nodematch`)

**Question**: Do entities sharing attribute X connect more than chance?

**Method**:
1. Compute `nx.attribute_assortativity_coefficient(G, attr)`. Range:
   -1 (perfect disassortativity) to +1 (perfect assortativity). 0 = random.
2. Run N=1000 permutation trials: shuffle attribute labels across nodes,
   recompute coefficient each time.
3. Two-tailed p-value: fraction of permuted |values| >= observed |value|.

**ERGM parallel**: In the Grey's Anatomy blog post, `nodematch("sex")`
tests whether same-sex ties occur more than expected. Here,
`nodematch("type")` tests whether PERSON-PERSON edges are more frequent
than PERSON-GPE edges beyond what group sizes predict.

**Interpretation for historians**:
- Positive + significant → institutional siloing along this dimension
- Near zero → the archive mixes entities across this dimension
- Negative + significant → the bureaucratic system systematically brings
  different categories into contact (e.g., medical examiners always paired
  with workers of different occupations)

### 9.2 Mixing matrix + residuals (ERGM `nodemix`)

**Question**: Which specific cross-category pairings occur more/less than
expected?

**Method**:
1. Observed matrix M[i][j] = weighted edge count between categories i, j.
2. Expected matrix under random mixing proportional to group sizes.
3. Standardized residuals: `(observed - expected) / sqrt(expected)`.
   |residual| > 2 → significant.

**ERGM parallel**: `nodemix("position", base=c(...))` in the blog post
tests whether resident-resident pairings differ from resident-attending
pairings. Here: do Brakeman-Brakeman document co-occurrences differ from
Brakeman-Clerk co-occurrences?

**Output**: Heatmap where red = overrepresented pairing, blue =
underrepresented. Notable pairs listed with exact residual values.

### 9.3 Degree distribution (ERGM `degree(1)`)

**Question**: How unequally are connections distributed?

**Method**:
1. Degree histogram + summary statistics.
2. Gini coefficient (0 = equality, 1 = one node has all edges).
3. Power-law fit via KS test (scipy).
4. Hub identification: degree > mean + 2*std.

**ERGM parallel**: The `degree(1)` term in the blog post controls for
"monogamy" (degree-1 nodes). Here we identify whether the degree
distribution reveals institutional gatekeepers — a few entities appearing
across hundreds of worker files.

**Interpretation**: High Gini + power-law fit → hub-and-spoke structure,
small number of institutional gatekeepers dominate. Low Gini → more
egalitarian documentary presence.

### 9.4 Configuration model comparison (ERGM `simulate` + `gof`)

**Question**: Is the observed clustering/modularity different from what
the degree sequence alone produces?

**Method**:
1. Generate N=100 random graphs with same degree sequence via
   `nx.configuration_model`.
2. Compute clustering, modularity, avg path length on each.
3. Compare observed to null distribution via z-scores.

**ERGM parallel**: The blog post uses `simulate(ga.base.d1)` to generate
random networks from the fitted model, then `gof()` to compare against
observed. Our configuration model serves the same role — a null
hypothesis that says "everything is explained by degree sequence alone."

**Interpretation**:
- Clustering significantly higher than random → real institutional
  groupings, not just hub artifacts
- Modularity significantly higher → genuine community structure

**Relationship to client-side panel**: The client panel already shows
clustering and modularity values. This test adds the critical question:
"but is that significant?" A clustering of 0.31 means nothing without
knowing whether random graphs with the same degree sequence produce 0.04
or 0.30.

### 9.5 Community detection + NMI

**Question**: What are the natural groupings, and do they correspond to
known categories?

**Method**:
1. Louvain community detection (or greedy modularity fallback).
2. Normalized Mutual Information (NMI) between detected communities and
   each known attribute. NMI ranges 0 (independent) to 1 (perfect match).
3. Composition breakdown per community.

**Interpretation**: If NMI(community, entity_type) = 0.6 but
NMI(community, occupation) = 0.15, then entity type is a much stronger
predictor of community membership than occupation.

### 9.6 Gatekeeper / bridge analysis

**Question**: Which entities connect otherwise separate communities?

**Method**:
1. Approximate betweenness centrality (k=500 sample).
2. Bridge score = `betweenness / log(degree + 1)`.
3. High bridge score = entity connects many communities relative to its
   overall connectivity → true structural broker, not just a hub.

**Relationship to client-side panel**: The client panel ranks by raw
betweenness. This provides the *normalized* bridge score that distinguishes
between "this entity is just very connected" vs. "this entity bridges
distinct communities."

---

## 10. Frontend behavior

### 10.1 Panel placement

The statistics panel appears **below** the existing bottom analytics panel
on `/network-analysis`. It starts in a collapsed/placeholder state — the
user clicks "Compute Statistics" to trigger the expensive server-side
pipeline.

### 10.2 Layout

```
┌──────────────────────────────────────────────────────────────┐
│ [Existing network graph and controls]                        │
├──────────────────────────────────────────────────────────────┤
│ [Existing bottom analytics panel: density, clustering, etc.] │
├──────────────────────────────────────────────────────────────┤
│ Statistical Analysis                           [Compute]     │
│                                                              │
│ ┌─ Graph Summary ──────────────────────────────────────────┐ │
│ │ 2,534 nodes · 12,847 edges · 312 components             │ │
│ └──────────────────────────────────────────────────────────┘ │
│                                                              │
│ ┌─ Assortativity Tests ────────────────────────────────────┐ │
│ │ Attribute     Coefficient  p-value   Sig.  Interpretation│ │
│ │ entity type   +0.34        <0.001    ✦✦✦   Homophily     │ │
│ │ occupation    +0.21        0.003     ✦✦    Homophily     │ │
│ └──────────────────────────────────────────────────────────┘ │
│                                                              │
│ ┌─ Mixing Matrix: [type ▾] [Show Matrix] ──────────────────┐ │
│ │ (heatmap of standardized residuals)                      │ │
│ │ Notable: PERSON×PERSON overrepresented (+3.4)            │ │
│ └──────────────────────────────────────────────────────────┘ │
│                                                              │
│ ┌─ Degree Distribution ────────────────────────────────────┐ │
│ │ [histogram]  Gini: 0.62 (high inequality)                │ │
│ │ Hubs: Dr. McIntire (209), Relief Dept (187)...           │ │
│ └──────────────────────────────────────────────────────────┘ │
│                                                              │
│ ┌─ Community Structure ────────────────────────────────────┐ │
│ │ 8 communities · modularity 0.47                          │ │
│ │ NMI with type: 0.32 | NMI with occupation: 0.18         │ │
│ └──────────────────────────────────────────────────────────┘ │
│                                                              │
│ ┌─ Observed vs. Random ────────────────────────────────────┐ │
│ │ Clustering: 0.31 vs random 0.04, z=8.2 ✦               │ │
│ │ Modularity: 0.47 vs random 0.22, z=5.1 ✦               │ │
│ └──────────────────────────────────────────────────────────┘ │
│                                                              │
│ ┌─ Gatekeepers ────────────────────────────────────────────┐ │
│ │ Entity         Degree  Betweenness  Bridge Score         │ │
│ │ Dr. McIntire   209     0.142        0.00068              │ │
│ └──────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### 10.3 Filter coordination

The statistics panel reads the same filter controls already on the page
(type checkboxes, min weight slider). When the user clicks "Compute
Statistics," the JS reads current filter values and passes them as query
parameters to `/api/network/statistics/summary`.

### 10.4 Mixing matrix interaction

The mixing matrix section includes its own attribute dropdown populated
from `/api/network/statistics/available_attributes`. This is the only
interactive sub-component — the user selects an attribute and clicks
"Show Matrix" to load that specific heatmap on demand.

---

## 11. Data model additions

### 11.1 New collection: `network_statistics_cache`

```javascript
{
  cache_key: "stats_summary:['PERSON','GPE']:3:3",
  result: { ... },  // full statistics JSON
  computed_at: ISODate
}
```

Indexes: unique on `cache_key`.

### 11.2 Future collection: `network_node_attributes`

Not created by this extension, but supported if populated separately:

```javascript
{
  entity_id: "693c6dbe...",
  attributes: {
    occupation: "Brakeman",
    department: "C.T. Department",
    national_origin: "Italian"
  }
}
```

When this collection exists, `load_node_attributes_from_collection()`
automatically attaches these attributes to the NetworkX graph, and
assortativity/mixing tests include them alongside the default `type`
attribute. This is the bridge to the research notebook's
`group_indicators` data.

---

## 12. Validation

### 12.1 Route registration

```bash
docker compose -f /Users/louishyman/coding/nosql/nosql_reader/docker-compose.yml \
  exec -T app python - <<'PY'
from routes import app
c = app.test_client()
for path in [
    '/api/network/statistics/graph_summary',
    '/api/network/statistics/available_attributes',
]:
    r = c.get(path)
    print(path, r.status_code, r.get_json())
PY
```

Expected: both return `200` with valid JSON.

### 12.2 Single assortativity test (fast)

```bash
docker compose -f /Users/louishyman/coding/nosql/nosql_reader/docker-compose.yml \
  exec -T app python - <<'PY'
from routes import app
c = app.test_client()
r = c.get('/api/network/statistics/assortativity?attribute=type&n_permutations=100&min_weight=3')
data = r.get_json()
print('Status:', r.status_code)
for result in data.get('results', []):
    print(f"  {result.get('attribute')}: observed={result.get('observed')}, p={result.get('p_value')}, sig={result.get('significant')}")
PY
```

### 12.3 Full summary (slow, ~45s)

```bash
docker compose -f /Users/louishyman/coding/nosql/nosql_reader/docker-compose.yml \
  exec -T app python - <<'PY'
from routes import app
c = app.test_client()
r = c.get('/api/network/statistics/summary?min_weight=3&n_permutations=500&n_simulations=50')
data = r.get_json()
print('Status:', r.status_code)
print('Graph:', data.get('graph_summary'))
print('Assortativity tests:', len(data.get('assortativity', [])))
print('Communities:', data.get('communities', {}).get('n_communities'))
print('Comparison:', {k: v.get('z_score') for k, v in data.get('comparison_to_random', {}).items() if isinstance(v, dict)})
PY
```

### 12.4 UI verification

1. Navigate to `/network-analysis`
2. Scroll below the existing bottom analytics panel
3. "Statistical Analysis" section should appear with "Compute Statistics" button
4. Click "Compute" — spinner shows, results populate after 30–60s
5. All six panels render: Graph Summary, Assortativity, Mixing Matrix,
   Degree Distribution, Communities, Observed vs Random, Gatekeepers
6. Mixing matrix: select "type" from dropdown, click "Show Matrix" →
   heatmap renders with colored cells and notable pairs
7. Hub entities and gatekeeper entities should link to ego network views

---

## 13. Performance characteristics

For a network of ~2,500 nodes and ~13,000 edges (typical with `min_weight=3`):

| Operation | Time | Notes |
|-----------|------|-------|
| Graph summary | <1s | |
| Available attributes | <2s | Builds small graph sample |
| Assortativity (1 attr, 1000 perms) | ~5s | |
| All assortativity (4 attrs) | ~20s | |
| Mixing matrix (1 attr) | ~1s | |
| Degree distribution | ~1s | |
| Community detection | ~3s | |
| Gatekeeper analysis | ~3s | Betweenness with k=500 sample |
| Configuration model (100 sims) | ~15s | |
| **Full summary** | **~45s** | **Cached after first run** |

Reduce `n_permutations` to 500 and `n_simulations` to 50 for ~20s total.

---

## 14. Domain agnosticism

All maintained throughout:

- Entity types read from `/api/network/types` and auto-detected from graph
- Attribute names auto-detected via `_detect_categorical_attributes()`
- Mixing matrix dropdown populated from API, not hard-coded
- No B&O Railroad-specific strings in any code file
- Colors assigned dynamically, heatmap scales to data range
- All thresholds environment-configurable

---

## 15. Future: node attribute enrichment (Phase 4d)

The statistics module already supports the `network_node_attributes`
collection. To populate it from research notebook data:

1. Read `group_indicators` from notebook (occupation, department,
   national_origin entries with `evidence_doc_ids`)
2. For each PERSON entity, find group indicators whose evidence documents
   overlap with the entity's documents
3. Write attribute assignment to `network_node_attributes`
4. Re-run statistics — assortativity/mixing now test these richer
   attributes alongside basic entity `type`

This enrichment script is designed but NOT included in this package.
