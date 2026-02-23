# app/network/network_routes.py
"""
Flask Blueprint routes for network analysis API.

All endpoints return JSON. No domain-specific logic — entity types, filtering,
and display are driven by configuration and the data itself.

Endpoints:
    GET /api/network/entity/<entity_id>     — Ego network
    GET /api/network/document/<doc_id>      — Document context graph
    GET /api/network/global                 — Filtered global network
    GET /api/network/metrics/<entity_id>    — Entity metrics
    GET /api/network/related/<doc_id>       — Related documents via network
    GET /api/network/stats                  — Network summary statistics
    GET /api/network/types                  — Available entity types
"""

import logging

from flask import jsonify, request

from database_setup import get_client, get_db
from network import network_bp
from network.config import NetworkConfig
from network.network_utils import (
    get_available_entity_types,
    get_document_network,
    get_ego_network,
    get_entity_metrics,
    get_global_network,
    get_network_stats,
    get_related_documents_via_network,
)

logger = logging.getLogger(__name__)
_client = get_client()
_db_instance = get_db(_client)


# ===========================================================================
# Helpers
# ===========================================================================

def _db():
    """Get a database handle. Follows existing project pattern."""
    return _db_instance


def _parse_type_filter() -> list | None:
    """
    Parse the 'type_filter' query parameter into a list of entity types.

    Accepts comma-separated values: ?type_filter=PERSON,GPE
    Returns None if not provided (meaning no filter).
    """
    raw = request.args.get("type_filter", "").strip()
    if not raw:
        return None
    return [t.strip() for t in raw.split(",") if t.strip()]


def _parse_int(param: str, default: int) -> int:
    """Safely parse an integer query parameter."""
    try:
        return int(request.args.get(param, default))
    except (TypeError, ValueError):
        return default


def _parse_bool(param: str, default: bool) -> bool:
    raw = request.args.get(param)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")


def _parse_document_term() -> str | None:
    """
    Parse optional free-text document term filter.

    This term is matched across all document fields.
    """
    raw = request.args.get("document_term", "")
    term = str(raw).strip()
    if not term:
        return None
    return term[:120]


def _parse_person_min_mentions(default: int = 3) -> int:
    """Parse minimum mention count required for PERSON endpoints."""
    value = _parse_int("person_min_mentions", default)
    return max(0, value)


def _network_enabled() -> bool:
    return NetworkConfig.from_env().enabled


# ===========================================================================
# Endpoints
# ===========================================================================

@network_bp.route("/entity/<entity_id>", methods=["GET"])
def ego_network_endpoint(entity_id: str):
    """
    GET /api/network/entity/<entity_id>

    Returns the ego network for a single entity — all its direct connections.

    Query parameters:
        type_filter  — comma-separated entity types (e.g., PERSON,GPE)
        min_weight   — minimum edge weight (default: 1)
        limit        — max edges to return (default: 50)
        document_term — optional term matched across all document fields
    """
    try:
        if not _network_enabled():
            return jsonify({"error": "Network analysis is disabled"}), 503

        db = _db()
        result = get_ego_network(
            db,
            entity_id=entity_id,
            type_filter=_parse_type_filter(),
            min_weight=_parse_int("min_weight", 1),
            limit=_parse_int("limit", 50),
            document_term=_parse_document_term(),
        )

        if result is None:
            return jsonify({"error": f"Entity {entity_id} not found"}), 404

        return jsonify(result)

    except Exception as e:
        logger.error("Error in ego_network_endpoint: %s", e, exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@network_bp.route("/document/<doc_id>", methods=["GET"])
def document_network_endpoint(doc_id: str):
    """
    GET /api/network/document/<doc_id>

    Returns all entities in this document and their cross-document connections.

    Query parameters:
        type_filter  — comma-separated entity types
        min_weight   — minimum edge weight (default: 1)
    """
    try:
        if not _network_enabled():
            return jsonify({"error": "Network analysis is disabled"}), 503

        db = _db()
        result = get_document_network(
            db,
            doc_id=doc_id,
            type_filter=_parse_type_filter(),
            min_weight=_parse_int("min_weight", 1),
        )

        if result is None:
            return jsonify({"error": f"Document {doc_id} not found"}), 404

        return jsonify(result)

    except Exception as e:
        logger.error("Error in document_network_endpoint: %s", e, exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@network_bp.route("/global", methods=["GET"])
def global_network_endpoint():
    """
    GET /api/network/global

    Returns a filtered view of the global co-occurrence network.

    Query parameters:
        type_filter  — comma-separated entity types
        min_weight   — minimum edge weight (default: 3)
        limit        — max edges to return (default: 500)
        strict_type_filter — when true, keep only edges where BOTH endpoint
                             types are in type_filter. when false, keep edges
                             where EITHER endpoint type is in type_filter.
                             default: true.
        person_min_mentions — minimum mention_count for PERSON endpoints
                              participating in returned edges (default: 3).
        document_term — optional term matched across all document fields
    """
    try:
        config = NetworkConfig.from_env()
        strict_type_filter = _parse_bool("strict_type_filter", True)
        if not config.enabled:
            return jsonify(
                {
                    "nodes": [],
                    "edges": [],
                    "stats": {
                        "total_edges_matching": 0,
                        "edges_returned": 0,
                        "nodes_returned": 0,
                        "strict_type_filter": strict_type_filter,
                        "disabled": True,
                    },
                }
            )

        db = _db()
        result = get_global_network(
            db,
            type_filter=_parse_type_filter(),
            min_weight=_parse_int("min_weight", config.default_min_weight),
            limit=_parse_int("limit", config.default_limit),
            strict_type_filter=strict_type_filter,
            person_min_mentions=_parse_person_min_mentions(3),
            document_term=_parse_document_term(),
        )

        return jsonify(result)

    except Exception as e:
        logger.error("Error in global_network_endpoint: %s", e, exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@network_bp.route("/metrics/<entity_id>", methods=["GET"])
def entity_metrics_endpoint(entity_id: str):
    """
    GET /api/network/metrics/<entity_id>

    Returns degree, top connections, and type distribution for an entity.
    Results are cached with a configurable TTL.
    """
    try:
        if not _network_enabled():
            return jsonify({"error": "Network analysis is disabled"}), 503

        db = _db()
        result = get_entity_metrics(db, entity_id)

        if result is None:
            return jsonify({"error": f"Entity {entity_id} not found"}), 404

        # Convert datetime for JSON serialization
        if "computed_at" in result:
            result["computed_at"] = result["computed_at"].isoformat()

        return jsonify(result)

    except Exception as e:
        logger.error("Error in entity_metrics_endpoint: %s", e, exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@network_bp.route("/related/<doc_id>", methods=["GET"])
def related_documents_endpoint(doc_id: str):
    """
    GET /api/network/related/<doc_id>

    Returns documents related through shared entity network connections,
    enabling cross-person-folder discovery.

    Query parameters:
        limit  — max related documents (default: 10)
    """
    try:
        if not _network_enabled():
            return jsonify({"error": "Network analysis is disabled"}), 503

        db = _db()
        results = get_related_documents_via_network(
            db,
            doc_id=doc_id,
            limit=_parse_int("limit", 10),
        )

        return jsonify({"document_id": doc_id, "related": results})

    except Exception as e:
        logger.error("Error in related_documents_endpoint: %s", e, exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@network_bp.route("/stats", methods=["GET"])
def network_stats_endpoint():
    """
    GET /api/network/stats

    Returns summary statistics about the network_edges collection.
    Useful for the explorer page header and diagnostics.
    """
    try:
        if not _network_enabled():
            return jsonify({"exists": False, "disabled": True, "total_edges": 0, "type_pairs": []})

        db = _db()
        stats = get_network_stats(db)
        return jsonify(stats)

    except Exception as e:
        logger.error("Error in network_stats_endpoint: %s", e, exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


@network_bp.route("/types", methods=["GET"])
def entity_types_endpoint():
    """
    GET /api/network/types

    Returns the list of distinct entity types present in linked_entities.
    Used to populate filter controls dynamically — no hard-coded types in the UI.
    """
    try:
        if not _network_enabled():
            return jsonify({"types": [], "disabled": True})

        db = _db()
        types = get_available_entity_types(db)
        return jsonify({"types": types})

    except Exception as e:
        logger.error("Error in entity_types_endpoint: %s", e, exc_info=True)
        return jsonify({"error": "Internal server error"}), 500
