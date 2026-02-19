# app/network/network_utils.py
"""
Query helpers and metrics computation for network analysis.

All functions accept a database handle and return plain dicts/lists suitable
for JSON serialization. No Flask dependencies — these are pure data layer
operations.

Domain-agnostic: no entity-type-specific logic. Filtering by type is done
via parameters, not hard-coded checks.
"""

import logging
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from bson import ObjectId

from app.network.config import NetworkConfig

logger = logging.getLogger(__name__)


# ===========================================================================
# Collection helpers
# ===========================================================================

def _get_network_edges(db):
    """Return the network_edges collection, or None if it doesn't exist."""
    if "network_edges" not in db.list_collection_names():
        logger.warning("network_edges collection does not exist. Run build_network_edges first.")
        return None
    return db["network_edges"]


def _get_linked_entities(db):
    """Return the linked_entities collection."""
    return db["linked_entities"]


def _get_network_metrics(db):
    """Return (and create if needed) the network_metrics collection."""
    return db["network_metrics"]


# ===========================================================================
# Entity info
# ===========================================================================

def get_entity_info(db, entity_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch basic info for a single linked entity.

    Parameters
    ----------
    entity_id : str
        String representation of linked_entities._id.

    Returns
    -------
    dict or None
        {id, name, type, mention_count} or None if not found.
    """
    linked = _get_linked_entities(db)
    try:
        entity = linked.find_one(
            {"_id": ObjectId(entity_id)},
            {"canonical_name": 1, "type": 1, "mention_count": 1},
        )
    except Exception:
        logger.debug("Invalid entity_id format: %s", entity_id)
        return None

    if not entity:
        return None

    return {
        "id": str(entity["_id"]),
        "name": entity.get("canonical_name", "Unknown"),
        "type": entity.get("type", "UNKNOWN"),
        "mention_count": entity.get("mention_count", 0),
    }


def get_entities_bulk(db, entity_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetch info for multiple entities in a single query.

    Returns a dict keyed by str(entity_id).
    """
    if not entity_ids:
        return {}

    linked = _get_linked_entities(db)
    try:
        oids = [ObjectId(eid) for eid in entity_ids]
    except Exception:
        logger.warning("Some entity_ids have invalid format. Filtering.")
        oids = []
        for eid in entity_ids:
            try:
                oids.append(ObjectId(eid))
            except Exception:
                pass

    cursor = linked.find(
        {"_id": {"$in": oids}},
        {"canonical_name": 1, "type": 1, "mention_count": 1},
    )

    result = {}
    for entity in cursor:
        eid = str(entity["_id"])
        result[eid] = {
            "id": eid,
            "name": entity.get("canonical_name", "Unknown"),
            "type": entity.get("type", "UNKNOWN"),
            "mention_count": entity.get("mention_count", 0),
        }
    return result


# ===========================================================================
# Ego network
# ===========================================================================

def get_ego_network(
    db,
    entity_id: str,
    type_filter: Optional[List[str]] = None,
    min_weight: int = 1,
    limit: int = 50,
) -> Optional[Dict[str, Any]]:
    """
    Get the ego network for a single entity — all its direct connections.

    Parameters
    ----------
    entity_id : str
        The focal entity's linked_entities._id as string.
    type_filter : list of str, optional
        Only include connected entities of these types.
    min_weight : int
        Minimum edge weight to include.
    limit : int
        Maximum number of edges to return.

    Returns
    -------
    dict or None
        {entity, edges, metrics} or None if entity not found.
    """
    entity_info = get_entity_info(db, entity_id)
    if not entity_info:
        return None

    edges_coll = _get_network_edges(db)
    if edges_coll is None:
        return {"entity": entity_info, "edges": [], "metrics": _empty_metrics()}

    # Query edges where this entity is source or target
    query = {
        "$or": [
            {"source_id": entity_id},
            {"target_id": entity_id},
        ],
        "weight": {"$gte": min_weight},
    }

    cursor = edges_coll.find(query).sort("weight", -1)

    edges = []
    type_distribution: Counter = Counter()

    for edge in cursor:
        # Determine the "other" entity in this edge
        if edge["source_id"] == entity_id:
            other_id = edge["target_id"]
            other_name = edge["target_name"]
            other_type = edge["target_type"]
        else:
            other_id = edge["source_id"]
            other_name = edge["source_name"]
            other_type = edge["source_type"]

        # Apply type filter
        if type_filter and other_type not in type_filter:
            continue

        type_distribution[other_type] += 1

        if len(edges) < limit:
            edges.append({
                "entity_id": other_id,
                "name": other_name,
                "type": other_type,
                "weight": edge["weight"],
                "document_ids": edge.get("document_ids", []),
            })

    metrics = {
        "degree": len(type_distribution) and sum(type_distribution.values()),
        "weighted_degree": sum(e["weight"] for e in edges),
        "type_distribution": dict(type_distribution),
    }

    return {
        "entity": entity_info,
        "edges": edges,
        "metrics": metrics,
    }


# ===========================================================================
# Document context network
# ===========================================================================

def get_document_network(
    db,
    doc_id: str,
    type_filter: Optional[List[str]] = None,
    min_weight: int = 1,
) -> Optional[Dict[str, Any]]:
    """
    Get the network context for a single document — all entities in the doc
    and their cross-document connections.

    Parameters
    ----------
    doc_id : str
        The document's _id as string.

    Returns
    -------
    dict or None
        {document_id, filename, nodes, edges} or None if doc not found.
    """
    documents = db["documents"]
    try:
        doc = documents.find_one(
            {"_id": ObjectId(doc_id)},
            {"entity_refs": 1, "filename": 1},
        )
    except Exception:
        return None

    if not doc:
        return None

    entity_refs = [str(ref) for ref in doc.get("entity_refs", [])]
    if not entity_refs:
        return {
            "document_id": doc_id,
            "filename": doc.get("filename", "Unknown"),
            "nodes": [],
            "edges": [],
        }

    # Fetch entity info for all refs in this doc
    entity_info_map = get_entities_bulk(db, entity_refs)

    # Apply type filter to nodes
    if type_filter:
        filtered_ids = {
            eid for eid, info in entity_info_map.items()
            if info["type"] in type_filter
        }
    else:
        filtered_ids = set(entity_info_map.keys())

    nodes = [entity_info_map[eid] for eid in filtered_ids if eid in entity_info_map]

    # Fetch edges between these entities
    edges_coll = _get_network_edges(db)
    if edges_coll is None or len(filtered_ids) < 2:
        return {
            "document_id": doc_id,
            "filename": doc.get("filename", "Unknown"),
            "nodes": nodes,
            "edges": [],
        }

    id_list = list(filtered_ids)
    edge_query = {
        "source_id": {"$in": id_list},
        "target_id": {"$in": id_list},
        "weight": {"$gte": min_weight},
    }

    edges = []
    for edge in edges_coll.find(edge_query):
        edges.append({
            "source": edge["source_id"],
            "target": edge["target_id"],
            "weight": edge["weight"],
        })

    return {
        "document_id": doc_id,
        "filename": doc.get("filename", "Unknown"),
        "nodes": nodes,
        "edges": edges,
    }


# ===========================================================================
# Global network
# ===========================================================================

def get_global_network(
    db,
    type_filter: Optional[List[str]] = None,
    min_weight: int = 3,
    limit: int = 500,
) -> Dict[str, Any]:
    """
    Get a filtered view of the global network — top edges by weight.

    Returns
    -------
    dict
        {nodes, edges, stats}
    """
    edges_coll = _get_network_edges(db)
    if edges_coll is None:
        return {"nodes": [], "edges": [], "stats": {"total_edges": 0}}

    query: Dict[str, Any] = {"weight": {"$gte": min_weight}}

    # Type filter — match edges where at least one endpoint matches
    if type_filter:
        query["$or"] = [
            {"source_type": {"$in": type_filter}},
            {"target_type": {"$in": type_filter}},
        ]

    cursor = edges_coll.find(query).sort("weight", -1).limit(limit)

    edges = []
    node_ids: Set[str] = set()

    for edge in cursor:
        # If type_filter is set, only include edges where BOTH endpoints match
        # (the query above is loose to use indexes; we tighten here)
        if type_filter:
            if edge["source_type"] not in type_filter or edge["target_type"] not in type_filter:
                # Allow cross-type edges if at least one matches
                # Uncomment below for strict both-must-match:
                # continue
                pass

        edges.append({
            "source": edge["source_id"],
            "target": edge["target_id"],
            "weight": edge["weight"],
            "source_type": edge["source_type"],
            "target_type": edge["target_type"],
        })
        node_ids.add(edge["source_id"])
        node_ids.add(edge["target_id"])

    # Fetch node info
    entity_info_map = get_entities_bulk(db, list(node_ids))
    nodes = list(entity_info_map.values())

    total_edges = edges_coll.count_documents(query)

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_edges_matching": total_edges,
            "edges_returned": len(edges),
            "nodes_returned": len(nodes),
        },
    }


# ===========================================================================
# Entity metrics
# ===========================================================================

def get_entity_metrics(
    db,
    entity_id: str,
    config: Optional[NetworkConfig] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get or compute metrics for a single entity.

    Checks the network_metrics cache first. If missing or stale, computes
    from network_edges and caches the result.
    """
    if config is None:
        config = NetworkConfig.from_env()

    entity_info = get_entity_info(db, entity_id)
    if not entity_info:
        return None

    # Check cache
    metrics_coll = _get_network_metrics(db)
    cached = metrics_coll.find_one({"entity_id": entity_id})

    if cached:
        computed_at = cached.get("computed_at")
        if computed_at and isinstance(computed_at, datetime):
            age = (datetime.utcnow() - computed_at).total_seconds()
            if age < config.metrics_cache_ttl:
                # Cache hit
                cached.pop("_id", None)
                return cached

    # Compute from edges
    edges_coll = _get_network_edges(db)
    if edges_coll is None:
        return {
            "entity_id": entity_id,
            **entity_info,
            "degree": 0,
            "weighted_degree": 0,
            "top_connections": [],
            "type_distribution": {},
            "computed_at": datetime.utcnow(),
        }

    query = {
        "$or": [
            {"source_id": entity_id},
            {"target_id": entity_id},
        ],
    }

    connections = []
    type_counts: Counter = Counter()

    for edge in edges_coll.find(query).sort("weight", -1):
        if edge["source_id"] == entity_id:
            other = {
                "entity_id": edge["target_id"],
                "name": edge["target_name"],
                "type": edge["target_type"],
                "weight": edge["weight"],
            }
        else:
            other = {
                "entity_id": edge["source_id"],
                "name": edge["source_name"],
                "type": edge["source_type"],
                "weight": edge["weight"],
            }

        connections.append(other)
        type_counts[other["type"]] += 1

    metrics = {
        "entity_id": entity_id,
        "entity_name": entity_info["name"],
        "entity_type": entity_info["type"],
        "degree": len(connections),
        "weighted_degree": sum(c["weight"] for c in connections),
        "top_connections": connections[:20],  # Top 20 by weight
        "type_distribution": dict(type_counts),
        "computed_at": datetime.utcnow(),
    }

    # Cache it
    try:
        metrics_coll.update_one(
            {"entity_id": entity_id},
            {"$set": metrics},
            upsert=True,
        )
    except Exception as e:
        logger.warning("Failed to cache metrics for %s: %s", entity_id, e)

    return metrics


# ===========================================================================
# Related documents via network
# ===========================================================================

def get_related_documents_via_network(
    db,
    doc_id: str,
    exclude_person_folder: Optional[str] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Find documents related to the given document through shared entity
    connections, excluding documents from the same person folder.

    This enables cross-person-folder discovery — the "serendipity" feature.

    Parameters
    ----------
    doc_id : str
        The source document's _id as string.
    exclude_person_folder : str, optional
        Person folder name to exclude (avoids recommending docs from the
        same person's archive).
    limit : int
        Maximum related documents to return.

    Returns
    -------
    list of dict
        [{document_id, filename, shared_entity_count, shared_entities}, ...]
    """
    documents = db["documents"]

    try:
        source_doc = documents.find_one(
            {"_id": ObjectId(doc_id)},
            {"entity_refs": 1, "person_folder": 1},
        )
    except Exception:
        return []

    if not source_doc:
        return []

    entity_refs = set(str(ref) for ref in source_doc.get("entity_refs", []))
    if not entity_refs:
        return []

    person_folder = exclude_person_folder or source_doc.get("person_folder")

    # Find entities connected to our doc's entities
    edges_coll = _get_network_edges(db)
    if edges_coll is None:
        return []

    # Get all connected entity IDs (1-hop neighbors)
    connected_ids: Set[str] = set()
    for ref in entity_refs:
        for edge in edges_coll.find(
            {"$or": [{"source_id": ref}, {"target_id": ref}]},
            {"source_id": 1, "target_id": 1},
        ):
            connected_ids.add(edge["source_id"])
            connected_ids.add(edge["target_id"])

    # Remove the source doc's own entities
    connected_ids -= entity_refs

    if not connected_ids:
        return []

    # Find documents that reference these connected entities
    # Use linked_entities.document_ids for efficiency
    linked = _get_linked_entities(db)
    doc_entity_counts: Counter = Counter()
    doc_entity_names: Dict[str, List[str]] = defaultdict(list)

    for eid in list(connected_ids)[:100]:  # Cap to avoid huge queries
        try:
            entity = linked.find_one(
                {"_id": ObjectId(eid)},
                {"document_ids": 1, "canonical_name": 1},
            )
        except Exception:
            continue

        if not entity:
            continue

        name = entity.get("canonical_name", "Unknown")
        for did in entity.get("document_ids", [])[:200]:  # Cap per entity
            did_str = str(did)
            if did_str != doc_id:
                doc_entity_counts[did_str] += 1
                if len(doc_entity_names[did_str]) < 5:
                    doc_entity_names[did_str].append(name)

    if not doc_entity_counts:
        return []

    # Get top related docs, excluding same person folder
    top_doc_ids = [did for did, _ in doc_entity_counts.most_common(limit * 3)]

    # Fetch doc info and filter
    try:
        oids = [ObjectId(did) for did in top_doc_ids]
    except Exception:
        return []

    related = []
    for doc in documents.find(
        {"_id": {"$in": oids}},
        {"filename": 1, "person_folder": 1},
    ):
        did = str(doc["_id"])

        # Exclude same person folder
        if person_folder and doc.get("person_folder") == person_folder:
            continue

        related.append({
            "document_id": did,
            "filename": doc.get("filename", "Unknown"),
            "shared_entity_count": doc_entity_counts[did],
            "shared_entities": doc_entity_names.get(did, []),
        })

        if len(related) >= limit:
            break

    # Sort by shared entity count
    related.sort(key=lambda x: x["shared_entity_count"], reverse=True)
    return related


# ===========================================================================
# Utility
# ===========================================================================

def _empty_metrics() -> Dict[str, Any]:
    return {
        "degree": 0,
        "weighted_degree": 0,
        "type_distribution": {},
    }


def get_available_entity_types(db) -> List[str]:
    """
    Return the distinct entity types present in linked_entities.

    Useful for populating filter controls in the UI without hard-coding types.
    """
    linked = _get_linked_entities(db)
    try:
        return sorted(linked.distinct("type"))
    except Exception as e:
        logger.error("Failed to fetch entity types: %s", e)
        return []


def get_network_stats(db) -> Dict[str, Any]:
    """
    Return summary statistics about the network_edges collection.

    Useful for the network explorer page header and diagnostics.
    """
    edges_coll = _get_network_edges(db)
    if edges_coll is None:
        return {"exists": False}

    total_edges = edges_coll.count_documents({})

    # Type pair distribution
    pipeline = [
        {"$group": {
            "_id": {"source_type": "$source_type", "target_type": "$target_type"},
            "count": {"$sum": 1},
            "avg_weight": {"$avg": "$weight"},
            "max_weight": {"$max": "$weight"},
        }},
        {"$sort": {"count": -1}},
    ]
    type_pairs = list(edges_coll.aggregate(pipeline))

    return {
        "exists": True,
        "total_edges": total_edges,
        "type_pairs": [
            {
                "source_type": tp["_id"]["source_type"],
                "target_type": tp["_id"]["target_type"],
                "count": tp["count"],
                "avg_weight": round(tp["avg_weight"], 1),
                "max_weight": tp["max_weight"],
            }
            for tp in type_pairs
        ],
    }
