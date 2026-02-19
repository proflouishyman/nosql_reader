# CLAUDE_CODE_INSTRUCTIONS.md
# Network Analysis Feature — Implementation Guide for Claude Code

## Context

You are implementing a network analysis feature for the Historical Document
Reader application. The design has been validated against live production data.
All code files are provided and tested against the existing codebase contracts.

**Branch:** `tier3_feb`
**This feature is additive** — it creates new files and makes minimal,
backward-compatible changes to existing files. It does NOT modify existing
database collections.

---

## File Inventory

### New files to ADD (copy these into the codebase)

| File | Description |
|------|-------------|
| `app/network/__init__.py` | Blueprint registration |
| `app/network/config.py` | Environment-based configuration |
| `app/network/build_network_edges.py` | Batch edge construction script |
| `app/network/network_utils.py` | Query helpers and metrics computation |
| `app/network/network_routes.py` | Flask API Blueprint |
| `app/static/js/network.js` | D3.js visualization + entity interactions |
| `app/templates/network-analysis.html` | Standalone explorer page |
| `app/templates/partials/network_context.html` | Document detail panel |

### Existing files to MODIFY (minimal, backward-compatible changes)

| File | Change |
|------|--------|
| `app/routes.py` | Enhance `build_document_ner_groups` + add `/network-analysis` route |
| `app/templates/document-detail.html` | Entity `data-entity-id` attributes + include partials |
| `app/templates/base.html` | Add D3.js CDN + network.js script tags + nav link |
| `.env.example` | Add NETWORK_* environment variables |
| `docker-compose.yml` | Add NETWORK_* environment variables |

---

## Implementation Order

Follow this exact sequence. Each phase is independently testable.

### Phase 0: Add new files

1. Create the `app/network/` directory
2. Copy in all four Python files (`__init__.py`, `config.py`,
   `build_network_edges.py`, `network_utils.py`, `network_routes.py`)
3. Copy `app/static/js/network.js`
4. Copy `app/templates/network-analysis.html`
5. Create `app/templates/partials/` directory if it doesn't exist
6. Copy `app/templates/partials/network_context.html`

### Phase 1: Register the blueprint

In whatever file initializes the Flask app (likely `app/__init__.py` or the top
of `app/routes.py` where other blueprints/routes are registered), add:

```python
# Near the top, with other imports
from app.network import init_app as init_network

# After app is created, with other blueprint registrations
init_network(app)
```

**If there is no app factory pattern** and routes are registered directly on
`app` in `routes.py`, add this near the end of imports/setup:

```python
import os
if os.environ.get("NETWORK_ANALYSIS_ENABLED", "true").lower() in ("1", "true", "yes"):
    from app.network.network_routes import network_bp
    app.register_blueprint(network_bp)
```

### Phase 2: Add the `/network-analysis` page route

In `app/routes.py`, add this route (near the other page-serving routes):

```python
@app.route('/network-analysis')
def network_analysis_page():
    """Standalone network explorer page."""
    return render_template('network-analysis.html')
```

### Phase 3: Enhance `build_document_ner_groups` in `routes.py`

See `PHASE_0_5_NER_GROUPS_PATCH.md` for the detailed patch specification.

**Summary of changes:**

1. Find the `build_document_ner_groups` function
2. Change its return type from `List[Tuple[str, List[str]]]` to
   `List[Tuple[str, List[Dict]]]`
3. Each entity becomes a dict: `{"text": "...", "entity_id": "...", "type": "..."}`
4. Entities resolved from `entity_refs` get their `linked_entities._id` as `entity_id`
5. Entities from other sources get `entity_id: None`
6. Update deduplication to work on `entity["text"]` instead of raw strings

### Phase 4: Update `document-detail.html`

1. **Entity rendering** — update the ner_groups loop:

```html
{% for type_label, entities in ner_groups %}
  <div class="entity-group">
    <h3>{{ type_label }}</h3>
    <ul>
      {% for entity in entities %}
        <li class="entity-item"
            {% if entity.entity_id %}
            data-entity-id="{{ entity.entity_id }}"
            data-entity-type="{{ entity.type }}"
            {% endif %}>
          {{ entity.text }}
        </li>
      {% endfor %}
    </ul>
  </div>
{% endfor %}
```

2. **Include network context panel** — add after the entities section,
   before person_synthesis:

```html
{% include 'partials/network_context.html' %}
```

3. **Add entity CSS** (inline in template or in existing stylesheet):

```css
.entity-item[data-entity-id] {
    cursor: pointer;
    transition: background-color 0.15s ease;
    padding: 2px 4px;
    border-radius: 3px;
}
.entity-item[data-entity-id]:hover {
    background-color: rgba(46, 117, 182, 0.1);
}
```

### Phase 5: Update `base.html`

1. **Add D3.js** (in the `<head>` or before closing `</body>`):

```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"
        integrity="sha512-vc58qvTBrKR0J70W2WFfTJHibIA5FBLlFB7u+wmF7YsFEUlUWB4VIPweuaBAalYqAT3JLP0Vqv4bul8CF7EDQ=="
        crossorigin="anonymous" referrerpolicy="no-referrer"></script>
```

2. **Add network.js** (after D3):

```html
<script src="{{ url_for('static', filename='js/network.js') }}"></script>
```

3. **Add nav link** (in the navigation area, alongside existing links like
   Search, Document List, etc.):

```html
<a href="{{ url_for('network_analysis_page') }}" class="nav-link">Network</a>
```

**NOTE:** D3 and network.js only need to load on pages that use them. If
`base.html` uses conditional blocks for scripts, wrap them:

```html
{% block scripts %}{% endblock %}
```

And move the D3/network.js includes to only the templates that need them
(`document-detail.html` and `network-analysis.html`). The
`network-analysis.html` template already includes them in its `{% block scripts %}`.

For `document-detail.html`, add at the bottom:

```html
{% block scripts %}
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"
        crossorigin="anonymous" referrerpolicy="no-referrer"></script>
<script src="{{ url_for('static', filename='js/network.js') }}"></script>
{% endblock %}
```

### Phase 6: Add environment variables

Add to `.env.example`:

```bash
# === Network Analysis ===
NETWORK_ANALYSIS_ENABLED=true
NETWORK_ENTITY_TYPES=PERSON,ORG,GPE,OCCUPATION
NETWORK_MAX_MENTION_COUNT=5000
NETWORK_MIN_EDGE_WEIGHT=2
NETWORK_BUILD_BATCH_SIZE=500
NETWORK_DEFAULT_LIMIT=500
NETWORK_DEFAULT_MIN_WEIGHT=1
NETWORK_METRICS_CACHE_TTL=86400
NETWORK_MAX_DISPLAY_NODES=500
```

Add the same variables to `docker-compose.yml` under the flask_app service
environment section, matching the format of existing env vars there.

### Phase 7: Build the edges

Run the batch construction script:

```bash
docker compose exec flask_app python -m app.network.build_network_edges
```

Verify with:

```bash
docker compose exec flask_app python -m app.network.build_network_edges --dry-run
```

---

## Validation Checklist

After implementation, verify each phase:

### Phase 0-1: Blueprint registered
```bash
docker compose exec flask_app python -c "
from app import app  # or however the app is imported
with app.test_client() as c:
    r = c.get('/api/network/types')
    print(r.status_code, r.get_json())
"
```
Expected: `200 {"types": ["DATE", "EMPLOYEE_ID", "GPE", "OCCUPATION", "ORG", "PERSON"]}`

### Phase 7: Edges built
```bash
docker compose exec flask_app python -c "
from app.database_setup import get_client, get_db
client = get_client()
db = get_db(client)
ne = db['network_edges']
print(f'Total edges: {ne.count_documents({}):,}')
for edge in ne.find().sort('weight', -1).limit(5):
    print(f'  {edge[\"source_name\"]} <-> {edge[\"target_name\"]} weight={edge[\"weight\"]}')
"
```

### API endpoints
```bash
# Stats
curl -s http://localhost:5000/api/network/stats | python -m json.tool

# Pick an entity_id from linked_entities and test ego network
curl -s 'http://localhost:5000/api/network/entity/<ENTITY_ID>?limit=5' | python -m json.tool

# Pick a doc_id and test document context
curl -s 'http://localhost:5000/api/network/document/<DOC_ID>' | python -m json.tool

# Global with filters
curl -s 'http://localhost:5000/api/network/global?type_filter=PERSON&min_weight=5&limit=20' | python -m json.tool
```

### UI verification
1. Navigate to `/network-analysis` — should show force graph
2. Navigate to any document detail — entities should have hover interaction
3. Click an entity — popup should appear with connections
4. Network context panel should appear below entities (if doc has enough entities)

---

## Critical Implementation Notes

### Database access pattern
**ALWAYS** use `from app.database_setup import get_client, get_db`. Never
create `MongoClient` directly. This is the existing project convention.

### entity_refs are STRINGS
`document.entity_refs` contains string representations of ObjectIds, not actual
ObjectIds. When looking up in `linked_entities`, convert:
`linked.find_one({"_id": ObjectId(ref_string)})`.

### Edge ordering convention
Source and target are ordered lexicographically: `source_id < target_id`.
This is enforced in `build_network_edges.py` via `generate_pairs` which uses
`sorted(set(entity_ids))` + `combinations`. The API endpoints handle both
directions when querying (checking both `source_id` and `target_id`).

### Graceful degradation
If `network_edges` collection doesn't exist (build script hasn't been run yet):
- API endpoints return empty results, not errors
- UI panels hide themselves, not show error states
- The `_get_network_edges` helper in `network_utils.py` checks for collection
  existence and returns `None` if missing

### Domain agnosticism
- Entity types are NEVER hard-coded in Python or JavaScript
- Colors are assigned dynamically based on encountered types
- Filter controls are populated from `/api/network/types`
- All thresholds are environment-configurable
- No railroad-specific logic anywhere

### Backward compatibility
- The `ner_groups` enhancement is the only change to existing code
- Entities without `entity_id` (from legacy sources) still render fine
- The template `{% if entity.entity_id %}` guard prevents broken attributes
- The network blueprint only registers if `NETWORK_ANALYSIS_ENABLED=true`

---

## Project conventions to follow

- Logging: `logger = logging.getLogger(__name__)` with file + console handlers
- Error handling: `try/except` with `logger.error(..., exc_info=True)`
- Config: environment variables with sensible defaults
- Templates: extend `base.html`, use existing CSS class names
- Routes: return `jsonify({"error": "..."})` with HTTP status codes on failure
