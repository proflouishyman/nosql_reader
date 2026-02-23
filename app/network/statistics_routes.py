# app/network/statistics_routes.py
"""
API endpoints for network statistical analysis (ERGM-inspired).

Registers on the existing network_bp blueprint. All endpoints return JSON.
Expensive operations cache results in network_statistics_cache collection.

To register, add to app/network/__init__.py:
    from network import statistics_routes  # noqa: F401, E402

Import convention follows project standard:
    from database_setup import get_client, get_db
"""

import logging
import math
from datetime import datetime

from flask import jsonify, request

from database_setup import get_client, get_db
from network import network_bp
from network.config import NetworkConfig
from network.network_statistics import (
    build_networkx_graph,
    compare_to_random_graphs,
    compute_all_assortativity,
    compute_assortativity,
    compute_degree_distribution,
    compute_full_statistics,
    compute_gatekeepers,
    compute_graph_summary,
    compute_mixing_matrix,
    detect_communities,
    load_node_attributes_from_collection,
    _detect_categorical_attributes,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Helpers
# ===========================================================================

def _db():
    client = get_client()
    return get_db(client)


def _parse_type_filter():
    raw = request.args.get("type_filter", "").strip()
    if not raw:
        return None
    return [t.strip() for t in raw.split(",") if t.strip()]


def _parse_int(param, default):
    try:
        return int(request.args.get(param, default))
    except (TypeError, ValueError):
        return default


def _build_graph(db, type_filter=None, min_weight=3, max_edges=50000):
    """
    Build graph honoring the same defaults as the existing /global endpoint.
    Respects person_min_mentions from query parameter.
    """
    person_min = _parse_int("person_min_mentions", 3)
    G = build_networkx_graph(
        db,
        type_filter=type_filter,
        min_weight=min_weight,
        max_edges=max_edges,
        person_min_mentions=person_min,
    )
    extra_attrs = load_node_attributes_from_collection(db, G)
    return G, extra_attrs


def _get_cached(db, cache_key, ttl_seconds=86400):
    coll = db["network_statistics_cache"]
    doc = coll.find_one({"cache_key": cache_key})
    if not doc:
        return None
    computed_at = doc.get("computed_at")
    if computed_at and isinstance(computed_at, datetime):
        age = (datetime.utcnow() - computed_at).total_seconds()
        if age > ttl_seconds:
            return None
    doc.pop("_id", None)
    doc.pop("cache_key", None)
    return doc.get("result")


def _set_cached(db, cache_key, result):
    try:
        coll = db["network_statistics_cache"]
        coll.update_one(
            {"cache_key": cache_key},
            {"$set": {
                "cache_key": cache_key,
                "result": result,
                "computed_at": datetime.utcnow(),
            }},
            upsert=True,
        )
    except Exception as e:
        logger.warning("Failed to cache statistics for %s: %s", cache_key, e)


def _to_json_safe(value):
    """
    Recursively convert values to JSON/Mongo-safe primitives.

    Handles numpy scalars, datetime objects, non-finite numbers, and
    dict keys that are not strings.
    """
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()

    # numpy scalar compatibility without hard dependency on numpy types
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _to_json_safe(value.item())
        except Exception:
            pass

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, (str, bool, int)) or value is None:
        return value

    return str(value)


# ===========================================================================
# Endpoints — all register on existing network_bp (/api/network/*)
# ===========================================================================

@network_bp.route("/statistics/summary", methods=["GET"])
def statistics_summary_endpoint():
    """
    GET /api/network/statistics/summary

    Full ERGM-style statistical report. Cached.

    Query parameters:
        type_filter, min_weight, person_min_mentions (match /global semantics)
        n_permutations (default 1000), n_simulations (default 100)
        force=true to bypass cache
    """
    try:
        db = _db()
        config = NetworkConfig.from_env()
        tf = _parse_type_filter()
        mw = _parse_int("min_weight", config.default_min_weight)
        pmm = _parse_int("person_min_mentions", 3)
        force = request.args.get("force", "").lower() == "true"

        cache_key = f"stats_summary:{tf}:{mw}:{pmm}"
        if not force:
            cached = _get_cached(db, cache_key, config.metrics_cache_ttl)
            if cached:
                cached["from_cache"] = True
                return jsonify(cached)

        G, extra_attrs = _build_graph(db, type_filter=tf, min_weight=mw)
        if G.number_of_nodes() < 5:
            return jsonify({"error": "Network too small for statistical analysis.",
                            "nodes": G.number_of_nodes()})

        n_perm = _parse_int("n_permutations", 1000)
        n_sim = _parse_int("n_simulations", 100)
        result = compute_full_statistics(G, n_permutations=n_perm, n_simulations=n_sim)

        # Strip large node_assignments before caching
        if "communities" in result and "node_assignments" in result["communities"]:
            result["communities"]["node_assignments_count"] = len(result["communities"]["node_assignments"])
            del result["communities"]["node_assignments"]

        safe_result = _to_json_safe(result)
        _set_cached(db, cache_key, safe_result)
        safe_result["from_cache"] = False
        return jsonify(safe_result)

    except Exception as e:
        logger.error("Error in statistics_summary: %s", e, exc_info=True)
        return jsonify({"error": "Internal server error computing statistics."}), 500


@network_bp.route("/statistics/assortativity", methods=["GET"])
def assortativity_endpoint():
    """
    GET /api/network/statistics/assortativity?attribute=type

    Single-attribute or all-attribute assortativity test with permutation p-value.
    """
    try:
        db = _db()
        tf = _parse_type_filter()
        mw = _parse_int("min_weight", 3)
        attribute = request.args.get("attribute", "").strip() or None

        G, extra_attrs = _build_graph(db, type_filter=tf, min_weight=mw)
        if G.number_of_nodes() < 10:
            return jsonify({"error": "Too few nodes for assortativity analysis."})

        n_perm = _parse_int("n_permutations", 1000)

        if attribute:
            result = compute_assortativity(G, attribute, n_permutations=n_perm)
        else:
            result = compute_all_assortativity(G, n_permutations=n_perm)

        safe_result = result if isinstance(result, list) else [result]
        return jsonify({"results": _to_json_safe(safe_result)})

    except Exception as e:
        logger.error("Error in assortativity: %s", e, exc_info=True)
        return jsonify({"error": "Internal server error."}), 500


@network_bp.route("/statistics/mixing_matrix", methods=["GET"])
def mixing_matrix_endpoint():
    """
    GET /api/network/statistics/mixing_matrix?attribute=type

    Mixing matrix with standardized residuals.
    """
    try:
        attribute = request.args.get("attribute", "").strip()
        if not attribute:
            return jsonify({"error": "Missing required 'attribute' parameter."}), 400

        db = _db()
        tf = _parse_type_filter()
        mw = _parse_int("min_weight", 3)

        G, _ = _build_graph(db, type_filter=tf, min_weight=mw)
        result = compute_mixing_matrix(G, attribute)
        return jsonify(_to_json_safe(result))

    except Exception as e:
        logger.error("Error in mixing_matrix: %s", e, exc_info=True)
        return jsonify({"error": "Internal server error."}), 500


@network_bp.route("/statistics/degree_distribution", methods=["GET"])
def degree_distribution_endpoint():
    """GET /api/network/statistics/degree_distribution"""
    try:
        db = _db()
        tf = _parse_type_filter()
        mw = _parse_int("min_weight", 3)

        G, _ = _build_graph(db, type_filter=tf, min_weight=mw)
        result = compute_degree_distribution(G)
        return jsonify(_to_json_safe(result))

    except Exception as e:
        logger.error("Error in degree_distribution: %s", e, exc_info=True)
        return jsonify({"error": "Internal server error."}), 500


@network_bp.route("/statistics/communities", methods=["GET"])
def communities_endpoint():
    """GET /api/network/statistics/communities"""
    try:
        db = _db()
        tf = _parse_type_filter()
        mw = _parse_int("min_weight", 3)

        G, extra_attrs = _build_graph(db, type_filter=tf, min_weight=mw)
        result = detect_communities(G, attributes_to_compare=extra_attrs or None)

        if "node_assignments" in result:
            result["node_assignments_count"] = len(result["node_assignments"])
            del result["node_assignments"]

        return jsonify(_to_json_safe(result))

    except Exception as e:
        logger.error("Error in communities: %s", e, exc_info=True)
        return jsonify({"error": "Internal server error."}), 500


@network_bp.route("/statistics/gatekeepers", methods=["GET"])
def gatekeepers_endpoint():
    """GET /api/network/statistics/gatekeepers?limit=20"""
    try:
        db = _db()
        tf = _parse_type_filter()
        mw = _parse_int("min_weight", 3)
        limit = _parse_int("limit", 20)

        G, _ = _build_graph(db, type_filter=tf, min_weight=mw)
        result = compute_gatekeepers(G, limit=limit)
        return jsonify(_to_json_safe(result))

    except Exception as e:
        logger.error("Error in gatekeepers: %s", e, exc_info=True)
        return jsonify({"error": "Internal server error."}), 500


@network_bp.route("/statistics/comparison", methods=["GET"])
def comparison_endpoint():
    """GET /api/network/statistics/comparison"""
    try:
        db = _db()
        config = NetworkConfig.from_env()
        tf = _parse_type_filter()
        mw = _parse_int("min_weight", 3)
        pmm = _parse_int("person_min_mentions", 3)

        cache_key = f"stats_comparison:{tf}:{mw}:{pmm}"
        cached = _get_cached(db, cache_key, config.metrics_cache_ttl)
        if cached:
            cached["from_cache"] = True
            return jsonify(cached)

        n_sim = _parse_int("n_simulations", 100)
        G, _ = _build_graph(db, type_filter=tf, min_weight=mw)
        result = compare_to_random_graphs(G, n_simulations=n_sim)

        safe_result = _to_json_safe(result)
        _set_cached(db, cache_key, safe_result)
        safe_result["from_cache"] = False
        return jsonify(safe_result)

    except Exception as e:
        logger.error("Error in comparison: %s", e, exc_info=True)
        return jsonify({"error": "Internal server error."}), 500


@network_bp.route("/statistics/graph_summary", methods=["GET"])
def graph_summary_endpoint():
    """GET /api/network/statistics/graph_summary — lightweight, fast."""
    try:
        db = _db()
        tf = _parse_type_filter()
        mw = _parse_int("min_weight", 3)
        G, _ = _build_graph(db, type_filter=tf, min_weight=mw)
        result = compute_graph_summary(G)
        return jsonify(_to_json_safe(result))

    except Exception as e:
        logger.error("Error in graph_summary: %s", e, exc_info=True)
        return jsonify({"error": "Internal server error."}), 500


@network_bp.route("/statistics/available_attributes", methods=["GET"])
def available_attributes_endpoint():
    """GET /api/network/statistics/available_attributes"""
    try:
        db = _db()
        tf = _parse_type_filter()
        mw = _parse_int("min_weight", 3)
        G, extra_attrs = _build_graph(db, type_filter=tf, min_weight=mw, max_edges=5000)

        detected = _detect_categorical_attributes(G)
        return jsonify(_to_json_safe({
            "detected_attributes": detected,
            "enrichment_attributes": extra_attrs,
            "all_attributes": sorted(set(detected + extra_attrs)),
        }))

    except Exception as e:
        logger.error("Error in available_attributes: %s", e, exc_info=True)
        return jsonify({"error": "Internal server error."}), 500
