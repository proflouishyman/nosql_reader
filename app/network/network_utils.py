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
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from bson import ObjectId

from network.config import NetworkConfig

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
# Document text matching helpers
# ===========================================================================

def _value_contains_term(value: Any, normalized_term: str) -> bool:
    """
    Recursively search for a term in arbitrarily nested document values.

    This intentionally scans "all fields" by traversing dict/list structures
    and checking scalar values as strings.
    """
    if value is None:
        return False

    if isinstance(value, str):
        return normalized_term in value.casefold()

    if isinstance(value, (int, float, bool)):
        return normalized_term in str(value).casefold()

    if isinstance(value, dict):
        for nested in value.values():
            if _value_contains_term(nested, normalized_term):
                return True
        return False

    if isinstance(value, (list, tuple, set)):
        for nested in value:
            if _value_contains_term(nested, normalized_term):
                return True
        return False

    # Fallback for types like ObjectId/datetime/etc.
    try:
        return normalized_term in str(value).casefold()
    except Exception:
        return False


def _document_contains_term(
    db,
    doc_id: str,
    normalized_term: str,
    match_cache: Dict[str, bool],
) -> bool:
    """
    Check if a document contains a search term anywhere across its fields.

    Uses an in-request cache to avoid repeated scans of the same document.
    """
    if not normalized_term:
        return True

    doc_id = str(doc_id)
    if doc_id in match_cache:
        return match_cache[doc_id]

    if not ObjectId.is_valid(doc_id):
        match_cache[doc_id] = False
        return False

    doc = db["documents"].find_one({"_id": ObjectId(doc_id)})
    if not doc:
        match_cache[doc_id] = False
        return False

    matched = _value_contains_term(doc, normalized_term)
    match_cache[doc_id] = matched
    return matched


def _filter_document_ids_by_term(
    db,
    document_ids: List[Any],
    document_term: Optional[str],
    match_cache: Dict[str, bool],
) -> List[str]:
    """
    Filter edge document_ids down to those whose full document content matches
    the supplied term.
    """
    if not document_term:
        return [str(doc_id) for doc_id in document_ids]

    normalized_term = str(document_term).strip().casefold()
    if not normalized_term:
        return [str(doc_id) for doc_id in document_ids]

    matched_ids: List[str] = []
    for doc_id in document_ids:
        doc_id_str = str(doc_id)
        if _document_contains_term(db, doc_id_str, normalized_term, match_cache):
            matched_ids.append(doc_id_str)
    return matched_ids


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
    document_term: Optional[str] = None,
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
    document_term : str, optional
        If provided, keep only edges that have at least one supporting
        document containing this term anywhere in its fields.

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
    document_match_cache: Dict[str, bool] = {}
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

        edge_document_ids = edge.get("document_ids", [])
        matched_document_ids = _filter_document_ids_by_term(
            db,
            edge_document_ids if isinstance(edge_document_ids, list) else [],
            document_term,
            document_match_cache,
        )
        if document_term and not matched_document_ids:
            continue

        type_distribution[other_type] += 1

        if len(edges) < limit:
            edges.append({
                "entity_id": other_id,
                "name": other_name,
                "type": other_type,
                "weight": edge["weight"],
                "document_ids": matched_document_ids,
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
    strict_type_filter: bool = True,
    document_term: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get a filtered view of the global network — top edges by weight.

    strict_type_filter semantics:
    - True: both endpoints must match type_filter.
    - False: either endpoint may match type_filter.
    - If document_term is provided, keep only edges whose document_ids contain
      at least one document matching that term across all fields.

    Returns
    -------
    dict
        {nodes, edges, stats}
    """
    edges_coll = _get_network_edges(db)
    if edges_coll is None:
        return {"nodes": [], "edges": [], "stats": {"total_edges": 0}}

    query: Dict[str, Any] = {"weight": {"$gte": min_weight}}

    # Type filter behavior:
    # strict=True  -> both endpoints must be selected types
    # strict=False -> either endpoint can match
    if type_filter:
        if strict_type_filter:
            query["source_type"] = {"$in": type_filter}
            query["target_type"] = {"$in": type_filter}
        else:
            query["$or"] = [
                {"source_type": {"$in": type_filter}},
                {"target_type": {"$in": type_filter}},
            ]

    cursor = edges_coll.find(query).sort("weight", -1).limit(limit)

    edges = []
    document_match_cache: Dict[str, bool] = {}
    node_ids: Set[str] = set()

    for edge in cursor:
        edge_document_ids = edge.get("document_ids", [])
        matched_document_ids = _filter_document_ids_by_term(
            db,
            edge_document_ids if isinstance(edge_document_ids, list) else [],
            document_term,
            document_match_cache,
        )
        if document_term and not matched_document_ids:
            continue

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
            "strict_type_filter": strict_type_filter,
        },
    }


def get_network_documents_for_viewer(
    db,
    entity_id: Optional[str] = None,
    type_filter: Optional[List[str]] = None,
    source_type: Optional[str] = None,
    target_type: Optional[str] = None,
    min_weight: int = 3,
    limit: int = 500,
    strict_type_filter: bool = True,
    pair_preset: str = "cross_type_only",
    rank_mode: str = "most_connected",
    document_term: Optional[str] = None,
    max_documents: int = 2000,
    max_docs_per_edge: int = 250,
) -> Dict[str, Any]:
    """
    Build a deterministic document list from the current network scope.

    The returned document ordering is designed for document viewer navigation
    (prev/next), not just display:
      1) matched_entity_count DESC
      2) network_score DESC
      3) matched_edge_count DESC
      4) filename ASC
      5) document_id ASC

    Parameters
    ----------
    entity_id : str, optional
        If provided, build from that entity's ego network.
        If omitted, build from filtered global network edges.
    type_filter : list[str], optional
        Entity type filter.
    source_type : str, optional
        Optional type-pair filter endpoint A.
    target_type : str, optional
        Optional type-pair filter endpoint B.
    min_weight : int
        Minimum edge weight.
    limit : int
        Max edges to scan.
    strict_type_filter : bool
        Global mode only:
          - True: both endpoints must match type_filter
          - False: either endpoint may match type_filter
    pair_preset : str
        Pair filtering preset:
          - cross_type_only
          - all_pairs
          - within_type_only
    rank_mode : str
        Edge ranking mode:
          - most_connected
          - most_cross_type
          - rare_but_strong
          - low_frequency_pairs
    max_documents : int
        Safety cap for unique documents returned.
    max_docs_per_edge : int
        Safety cap for document_ids consumed per edge.
    """
    edges_coll = _get_network_edges(db)
    if edges_coll is None:
        return {
            "mode": "entity" if entity_id else "global",
            "entity": None,
            "documents": [],
            "stats": {
                "edges_scanned": 0,
                "documents_returned": 0,
                "strict_type_filter": strict_type_filter,
            },
        }

    safe_limit = max(1, int(limit))
    safe_max_documents = max(1, int(max_documents))
    safe_docs_per_edge = max(1, int(max_docs_per_edge))
    rank_mode = (rank_mode or "most_connected").strip().lower()
    pair_preset = (pair_preset or "cross_type_only").strip().lower()
    normalized_types = [t for t in (type_filter or []) if t]
    source_type = (source_type or "").strip() or None
    target_type = (target_type or "").strip() or None

    entity_info = None
    mode = "entity" if entity_id else "global"
    if entity_id:
        entity_info = get_entity_info(db, entity_id)
        if entity_info is None:
            return {
                "mode": "entity",
                "entity": None,
                "documents": [],
                "stats": {
                    "edges_scanned": 0,
                    "documents_returned": 0,
                    "strict_type_filter": strict_type_filter,
                    "entity_not_found": True,
                },
            }

    projection = {
        "source_id": 1,
        "target_id": 1,
        "source_type": 1,
        "target_type": 1,
        "weight": 1,
        "document_ids": 1,
    }

    if entity_id:
        query: Dict[str, Any] = {
            "$or": [{"source_id": entity_id}, {"target_id": entity_id}],
            "weight": {"$gte": min_weight},
        }
    else:
        query = {"weight": {"$gte": min_weight}}
        if normalized_types:
            if strict_type_filter:
                query["source_type"] = {"$in": normalized_types}
                query["target_type"] = {"$in": normalized_types}
            else:
                query["$or"] = [
                    {"source_type": {"$in": normalized_types}},
                    {"target_type": {"$in": normalized_types}},
                ]

        if source_type and target_type:
            query["$and"] = query.get("$and", [])
            query["$and"].append(
                {
                    "$or": [
                        {"source_type": source_type, "target_type": target_type},
                        {"source_type": target_type, "target_type": source_type},
                    ]
                }
            )
        elif source_type:
            query["$and"] = query.get("$and", [])
            query["$and"].append({"$or": [{"source_type": source_type}, {"target_type": source_type}]})
        elif target_type:
            query["$and"] = query.get("$and", [])
            query["$and"].append({"$or": [{"source_type": target_type}, {"target_type": target_type}]})

    edges = list(edges_coll.find(query, projection).sort("weight", -1).limit(safe_limit))
    pair_counts: Counter = Counter()
    for edge in edges:
        pair_counts[(edge.get("source_type", "UNKNOWN"), edge.get("target_type", "UNKNOWN"))] += 1

    def _rank_key(edge: Dict[str, Any]):
        source_t = edge.get("source_type", "UNKNOWN")
        target_t = edge.get("target_type", "UNKNOWN")
        weight = int(edge.get("weight", 0) or 0)
        pair_count = pair_counts[(source_t, target_t)] or 1
        cross = source_t != target_t
        if rank_mode == "most_cross_type":
            return (0 if cross else 1, -weight, pair_count, source_t, target_t)
        if rank_mode == "rare_but_strong":
            rarity_score = float(weight) / float(pair_count)
            return (-rarity_score, -weight, pair_count, source_t, target_t)
        if rank_mode == "low_frequency_pairs":
            return (pair_count, -weight, source_t, target_t)
        return (-weight, pair_count, source_t, target_t)

    edges.sort(key=_rank_key)

    doc_scores: Counter = Counter()
    doc_edge_counts: Counter = Counter()
    doc_entities: Dict[str, Set[str]] = defaultdict(set)
    document_match_cache: Dict[str, bool] = {}
    edges_scanned = 0

    for edge in edges:
        edges_scanned += 1

        if entity_id and normalized_types:
            if edge.get("source_id") == entity_id:
                other_type = edge.get("target_type")
            else:
                other_type = edge.get("source_type")
            if other_type not in normalized_types:
                continue

        source_t = edge.get("source_type")
        target_t = edge.get("target_type")
        if pair_preset == "cross_type_only" and source_t == target_t:
            continue
        if pair_preset == "within_type_only" and source_t != target_t:
            continue

        if source_type and target_type:
            if {source_t, target_t} != {source_type, target_type}:
                continue
        elif source_type:
            if source_type not in (source_t, target_t):
                continue
        elif target_type:
            if target_type not in (source_t, target_t):
                continue

        edge_weight = int(edge.get("weight", 0) or 0)
        edge_docs = edge.get("document_ids", [])
        if not isinstance(edge_docs, list) or not edge_docs:
            continue

        matched_edge_docs = _filter_document_ids_by_term(
            db,
            edge_docs,
            document_term,
            document_match_cache,
        )
        if document_term and not matched_edge_docs:
            continue

        source_id = str(edge.get("source_id", ""))
        target_id = str(edge.get("target_id", ""))

        for doc_id in matched_edge_docs[:safe_docs_per_edge]:
            if doc_id not in doc_scores and len(doc_scores) >= safe_max_documents:
                continue

            doc_scores[doc_id] += edge_weight
            doc_edge_counts[doc_id] += 1
            if source_id:
                doc_entities[doc_id].add(source_id)
            if target_id:
                doc_entities[doc_id].add(target_id)

    if not doc_scores:
        return {
            "mode": mode,
            "entity": entity_info,
            "documents": [],
            "stats": {
                "edges_scanned": edges_scanned,
                "documents_returned": 0,
                "strict_type_filter": strict_type_filter,
            },
        }

    doc_meta: Dict[str, Dict[str, Any]] = {}
    valid_oids = []
    for doc_id in doc_scores.keys():
        if ObjectId.is_valid(doc_id):
            valid_oids.append(ObjectId(doc_id))

    if valid_oids:
        for doc in db["documents"].find(
            {"_id": {"$in": valid_oids}},
            {"filename": 1},
        ):
            doc_meta[str(doc["_id"])] = {
                "filename": doc.get("filename", "Unknown"),
            }

    def _sort_key(doc_id: str):
        filename = doc_meta.get(doc_id, {}).get("filename", "Unknown")
        return (
            -len(doc_entities.get(doc_id, set())),
            -doc_scores[doc_id],
            -doc_edge_counts[doc_id],
            str(filename).casefold(),
            doc_id,
        )

    ordered_doc_ids = sorted(doc_scores.keys(), key=_sort_key)

    documents_payload = []
    for doc_id in ordered_doc_ids:
        documents_payload.append(
            {
                "document_id": doc_id,
                "filename": doc_meta.get(doc_id, {}).get("filename", "Unknown"),
                "matched_entity_count": len(doc_entities.get(doc_id, set())),
                "matched_edge_count": int(doc_edge_counts[doc_id]),
                "network_score": int(doc_scores[doc_id]),
            }
        )

    return {
        "mode": mode,
        "entity": entity_info,
        "documents": documents_payload,
        "stats": {
            "edges_scanned": edges_scanned,
            "documents_returned": len(documents_payload),
            "strict_type_filter": strict_type_filter,
            "min_weight": min_weight,
            "limit": safe_limit,
            "type_filter": normalized_types,
            "source_type": source_type,
            "target_type": target_type,
            "pair_preset": pair_preset,
            "rank_mode": rank_mode,
            "document_term": (document_term or "").strip(),
            "max_documents": safe_max_documents,
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
        Each result includes:
        - shared_entity_count/shared_entities: direct overlap with source doc
        - network_connection_count/network_connector_entities: 1-hop network
          bridge signal used for discovery/ranking
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

    source_refs = {str(ref) for ref in source_doc.get("entity_refs", [])}
    if not source_refs:
        return []

    source_entity_map = get_entities_bulk(db, list(source_refs))
    person_folder = exclude_person_folder or source_doc.get("person_folder")

    # Find entities connected to our doc's entities
    edges_coll = _get_network_edges(db)
    if edges_coll is None:
        return []

    # Get all connected entity IDs (1-hop neighbors)
    connected_ids: Set[str] = set()
    for ref in source_refs:
        for edge in edges_coll.find(
            {"$or": [{"source_id": ref}, {"target_id": ref}]},
            {"source_id": 1, "target_id": 1},
        ):
            connected_ids.add(edge["source_id"])
            connected_ids.add(edge["target_id"])

    # Remove the source doc's own entities
    connected_ids -= source_refs

    if not connected_ids:
        return []

    # Find candidate documents through connected entities.
    # Use linked_entities.document_ids for efficiency.
    linked = _get_linked_entities(db)
    doc_network_counts: Counter = Counter()
    doc_connector_entities: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    for eid in list(connected_ids)[:100]:  # Cap to avoid huge queries
        try:
            entity = linked.find_one(
                {"_id": ObjectId(eid)},
                {"document_ids": 1, "canonical_name": 1, "type": 1},
            )
        except Exception:
            continue

        if not entity:
            continue

        connector_payload = {
            "entity_id": eid,
            "name": entity.get("canonical_name", "Unknown"),
            "type": entity.get("type", "UNKNOWN"),
        }
        for did in entity.get("document_ids", [])[:200]:  # Cap per entity
            did_str = str(did)
            if did_str != doc_id:
                doc_network_counts[did_str] += 1
                if len(doc_connector_entities[did_str]) < 6:
                    doc_connector_entities[did_str][eid] = connector_payload

    if not doc_network_counts:
        return []

    # Get top related docs by network connectivity signal.
    top_doc_ids = [did for did, _ in doc_network_counts.most_common(limit * 4)]

    # Fetch doc info and filter
    try:
        oids = [ObjectId(did) for did in top_doc_ids]
    except Exception:
        return []

    related = []
    for doc in documents.find(
        {"_id": {"$in": oids}},
        {"filename": 1, "person_folder": 1, "entity_refs": 1},
    ):
        did = str(doc["_id"])

        # Exclude same person folder
        if person_folder and doc.get("person_folder") == person_folder:
            continue

        # Directly shared entities = exact overlap of entity_refs.
        candidate_refs = {str(ref) for ref in doc.get("entity_refs", [])}
        shared_ids = [eid for eid in source_refs if eid in candidate_refs]
        shared_entities: List[Dict[str, Any]] = []
        for sid in shared_ids:
            info = source_entity_map.get(sid) or {
                "id": sid,
                "name": sid,
                "type": "UNKNOWN",
            }
            shared_entities.append(
                {
                    "entity_id": info["id"],
                    "name": info["name"],
                    "type": info["type"],
                }
            )

        connector_entities = list(doc_connector_entities.get(did, {}).values())

        related.append({
            "document_id": did,
            "filename": doc.get("filename", "Unknown"),
            "shared_entity_count": len(shared_entities),
            "shared_entities": shared_entities,
            "network_connection_count": doc_network_counts[did],
            "network_connector_entities": connector_entities,
        })

    # Prioritize documents that actually share entities directly, then network signal.
    related.sort(
        key=lambda x: (x["shared_entity_count"], x["network_connection_count"]),
        reverse=True,
    )
    return related[:limit]


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
