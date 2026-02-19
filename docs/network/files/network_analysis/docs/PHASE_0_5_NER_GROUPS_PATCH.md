# Phase 0.5 — `ner_groups` Enhancement Patch

## Overview

The existing `build_document_ner_groups` function in `routes.py` merges all
entity sources into a unified display structure. Currently it returns text-only
data. This patch adds `linked_entities._id` pass-through so the frontend can
link entities to network analysis features.

## Current contract (what exists now)

```python
# routes.py line ~1432
def build_document_ner_groups(document, db) -> List[Tuple[str, List[str]]]:
    """Returns [(entity_type, [entity_text, ...]), ...]"""
```

Template renders:
```html
{% for type_label, entities in ner_groups %}
  <h3>{{ type_label }}</h3>
  <ul>
    {% for entity_text in entities %}
      <li>{{ entity_text }}</li>
    {% endfor %}
  </ul>
{% endfor %}
```

## Target contract (what it should become)

```python
def build_document_ner_groups(document, db) -> List[Tuple[str, List[Dict[str, str]]]]:
    """
    Returns [(entity_type, [entity_dict, ...]), ...]

    Each entity_dict contains:
        text         — display string (always present)
        entity_id    — str(linked_entities._id) or None if unresolved
        canonical_name — from linked_entities, or same as text
        type         — entity type string
    """
```

## Implementation instructions

### Step 1: Modify `build_document_ner_groups` in `routes.py`

The function already resolves `entity_refs` → `linked_entities`. The change is
to **keep the `_id`** during resolution instead of discarding it.

Find the section where entity_refs are resolved (it loops over `entity_refs`,
looks up each in `linked_entities`, and builds the merged groups). Change the
inner logic so that instead of appending just the text string, it appends a dict:

```python
# BEFORE (pseudocode of current pattern):
for ref_id in document.get("entity_refs", []):
    linked = db["linked_entities"].find_one({"_id": ObjectId(ref_id)})
    if linked:
        entity_type = linked.get("type", "UNKNOWN")
        name = linked.get("canonical_name", "Unknown")
        groups[entity_type].append(name)  # <-- text only

# AFTER:
for ref_id in document.get("entity_refs", []):
    linked = db["linked_entities"].find_one({"_id": ObjectId(ref_id)})
    if linked:
        entity_type = linked.get("type", "UNKNOWN")
        groups[entity_type].append({
            "text": linked.get("canonical_name", "Unknown"),
            "entity_id": str(linked["_id"]),
            "canonical_name": linked.get("canonical_name", "Unknown"),
            "type": entity_type,
        })
```

For entities that come from OTHER sources (the `entities` dict, `extracted_entities`
list, or `sections.fields.linked_information.named_entities`), they won't have a
linked_entities._id. Set `entity_id` to `None`:

```python
# For entities without a linked_entities record:
groups[entity_type].append({
    "text": entity_text,
    "entity_id": None,
    "canonical_name": entity_text,
    "type": entity_type,
})
```

**Deduplication:** The current function likely deduplicates by text string.
Update the dedup logic to work on the `text` field of the dict:

```python
# Deduplicate within each type group by text value
seen = set()
deduped = []
for entity in group_list:
    if entity["text"].lower() not in seen:
        seen.add(entity["text"].lower())
        deduped.append(entity)
groups[entity_type] = deduped
```

### Step 2: Update `document-detail.html` template

Find the `ner_groups` rendering block (between Extracted Entities header and
the person_synthesis section). Change the inner loop:

```html
<!-- BEFORE -->
{% for type_label, entities in ner_groups %}
  <div class="entity-group">
    <h3>{{ type_label }}</h3>
    <ul>
      {% for entity_text in entities %}
        <li>{{ entity_text }}</li>
      {% endfor %}
    </ul>
  </div>
{% endfor %}

<!-- AFTER -->
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

### Step 3: Add CSS for entity items

Add to the existing stylesheet (or in a `<style>` block in the template):

```css
/* Entity items with network data are interactive */
.entity-item[data-entity-id] {
    cursor: pointer;
    transition: background-color 0.15s ease;
    padding: 2px 4px;
    border-radius: 3px;
}

.entity-item[data-entity-id]:hover {
    background-color: rgba(46, 117, 182, 0.1);
}

/* Entity popup (Phase 2) */
.entity-popup {
    position: absolute;
    z-index: 1000;
    background: white;
    border: 1px solid #ccc;
    border-radius: 6px;
    padding: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    max-width: 320px;
    font-size: 0.9em;
}

.entity-popup h4 {
    margin: 0 0 8px 0;
    font-size: 1em;
}

.entity-popup .entity-type-badge {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 0.75em;
    font-weight: 600;
    text-transform: uppercase;
}

.entity-popup .connections-list {
    list-style: none;
    padding: 0;
    margin: 8px 0;
}

.entity-popup .connections-list li {
    padding: 3px 0;
    border-bottom: 1px solid #eee;
}

.entity-popup .view-network-link {
    display: block;
    text-align: center;
    margin-top: 8px;
    color: #2E75B6;
    text-decoration: none;
    font-weight: 500;
}
```

## Backward compatibility

- Templates that don't use `data-entity-id` still render correctly (just show text)
- Entities without `linked_entities` records have `entity_id: None` — the
  `{% if entity.entity_id %}` guard prevents broken data attributes
- The popup JS (Phase 2) only attaches to elements WITH `data-entity-id`
