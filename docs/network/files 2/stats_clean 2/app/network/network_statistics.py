#!/usr/bin/env python3
# app/network/network_statistics.py
"""
Server-side statistical analysis of co-occurrence networks.

Provides ERGM-inspired hypothesis tests that go beyond the descriptive
client-side analytics already computed in network.js. While the browser
computes density, clustering, centrality rankings on the loaded graph slice,
this module answers: "Are these patterns statistically significant compared
to what random chance would produce?"

Methods implemented:
    1. Assortativity analysis with permutation testing (ERGM nodematch analog)
    2. Mixing matrix with standardized residuals (ERGM nodemix analog)
    3. Degree distribution analysis: Gini, power-law, hub detection
    4. Configuration model comparison (ERGM simulate analog)
    5. Community detection with NMI against known attributes
    6. Gatekeeper / bridge analysis (betweenness vs degree)

Dependencies: networkx, numpy, scipy (all optional-graceful)
Optional: python-louvain (community detection)

Import convention follows project standard:
    from database_setup import get_client, get_db
"""

import logging
import math
import random
import statistics as pystats
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. Graph construction from network_edges
# ===========================================================================

def build_networkx_graph(
    db,
    type_filter: Optional[List[str]] = None,
    min_weight: int = 1,
    max_edges: int = 50000,
    person_min_mentions: int = 0,
) -> nx.Graph:
    """
    Build a NetworkX graph from the network_edges collection.

    Respects the same filtering semantics as the existing API endpoints,
    including the person_min_mentions threshold.

    Parameters
    ----------
    db : pymongo database handle
    type_filter : list of str, optional
        Only include nodes of these entity types.
    min_weight : int
        Minimum edge weight to include.
    max_edges : int
        Safety cap on number of edges loaded.
    person_min_mentions : int
        For PERSON-type endpoints, require mention_count >= this value.
        Matches the existing /global endpoint behavior.
    """
    G = nx.Graph()

    edges_coll = db["network_edges"]
    linked_coll = db["linked_entities"]

    # Prefetch entity metadata for person_min_mentions filtering
    person_exclude: Set[str] = set()
    if person_min_mentions > 0:
        for ent in linked_coll.find(
            {"type": "PERSON", "mention_count": {"$lt": person_min_mentions}},
            {"_id": 1},
        ):
            person_exclude.add(str(ent["_id"]))

    # Build edge query
    query: Dict[str, Any] = {"weight": {"$gte": min_weight}}
    if type_filter:
        # strict type filter: both endpoints must match
        query["source_type"] = {"$in": type_filter}
        query["target_type"] = {"$in": type_filter}

    cursor = edges_coll.find(query).sort("weight", -1).limit(max_edges)

    node_ids: Set[str] = set()
    edge_count = 0

    for edge in cursor:
        src = edge["source_id"]
        tgt = edge["target_id"]

        # Apply person_min_mentions filter
        if person_min_mentions > 0:
            if (edge.get("source_type") == "PERSON" and src in person_exclude):
                continue
            if (edge.get("target_type") == "PERSON" and tgt in person_exclude):
                continue

        G.add_edge(src, tgt, weight=edge["weight"])
        node_ids.add(src)
        node_ids.add(tgt)
        edge_count += 1

    logger.info("Loaded %d edges, %d unique nodes", edge_count, len(node_ids))

    # Attach node attributes from linked_entities
    from bson import ObjectId

    for node_id in node_ids:
        try:
            entity = linked_coll.find_one(
                {"_id": ObjectId(node_id)},
                {"canonical_name": 1, "type": 1, "mention_count": 1},
            )
        except Exception:
            entity = None

        if entity:
            G.nodes[node_id]["name"] = entity.get("canonical_name", "Unknown")
            G.nodes[node_id]["type"] = entity.get("type", "UNKNOWN")
            G.nodes[node_id]["mention_count"] = entity.get("mention_count", 0)
        else:
            G.nodes[node_id]["name"] = "Unknown"
            G.nodes[node_id]["type"] = "UNKNOWN"
            G.nodes[node_id]["mention_count"] = 0

    return G


def load_node_attributes_from_collection(
    db,
    G: nx.Graph,
    collection_name: str = "network_node_attributes",
) -> List[str]:
    """
    Load enrichment attributes (occupation, department, national_origin, etc.)
    from a dedicated collection and attach them to graph nodes.

    This collection is populated separately from research notebook group
    indicators. Returns the list of attribute names loaded.

    Expected document schema:
    {
        "entity_id": "693c6dbe...",
        "attributes": {
            "occupation": "Brakeman",
            "department": "C.T. Department",
            "national_origin": "Italian"
        }
    }
    """
    if collection_name not in db.list_collection_names():
        logger.debug("Collection %s does not exist, skipping enrichment.", collection_name)
        return []

    coll = db[collection_name]
    attribute_names: Set[str] = set()
    loaded = 0

    for doc in coll.find({}):
        eid = doc.get("entity_id")
        if eid not in G.nodes:
            continue
        attrs = doc.get("attributes", {})
        for attr_name, attr_value in attrs.items():
            if attr_value is not None and attr_value != "":
                G.nodes[eid][attr_name] = str(attr_value)
                attribute_names.add(attr_name)
        loaded += 1

    logger.info("Loaded attributes for %d nodes. Attributes: %s", loaded, sorted(attribute_names))
    return sorted(attribute_names)


# ===========================================================================
# 2. Graph summary
# ===========================================================================

def compute_graph_summary(G: nx.Graph) -> Dict[str, Any]:
    """Basic structural summary of the graph."""
    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len) if components else set()

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": round(nx.density(G), 6) if G.number_of_nodes() > 1 else 0,
        "components": len(components),
        "largest_component_size": len(largest_cc),
        "largest_component_fraction": round(len(largest_cc) / max(G.number_of_nodes(), 1), 4),
        "isolates": nx.number_of_isolates(G),
    }


# ===========================================================================
# 3. Assortativity analysis with permutation testing
#    (ERGM nodematch analog — "Do entities sharing attribute X connect
#     more than random chance predicts?")
# ===========================================================================

def compute_assortativity(
    G: nx.Graph,
    attribute: str,
    n_permutations: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute attribute assortativity coefficient with a permutation test
    for statistical significance.

    This is the Python equivalent of ERGM's nodematch() term.
    The assortativity coefficient measures the tendency of nodes to connect
    to other nodes with the same attribute value. The permutation test
    establishes whether the observed value is beyond random expectation.

    Parameters
    ----------
    G : nx.Graph
        Graph with the attribute assigned to nodes.
    attribute : str
        Node attribute name to test (e.g., "type", "occupation").
    n_permutations : int
        Number of permutation trials for the null distribution.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with: attribute, observed, null_mean, null_std, p_value,
    significant, z_score, interpretation, categories.
    """
    rng = random.Random(seed)

    attr_dict = nx.get_node_attributes(G, attribute)
    nodes_with_attr = set(attr_dict.keys())

    if len(nodes_with_attr) < 10:
        return {
            "attribute": attribute,
            "error": f"Only {len(nodes_with_attr)} nodes have attribute '{attribute}'. Need >=10.",
            "n_nodes_with_attr": len(nodes_with_attr),
        }

    categories = sorted(set(attr_dict.values()))
    if len(categories) < 2:
        return {
            "attribute": attribute,
            "error": f"Only {len(categories)} category for '{attribute}'. Need >=2.",
            "n_nodes_with_attr": len(nodes_with_attr),
            "categories": categories,
        }

    H = G.subgraph(nodes_with_attr).copy()
    if H.number_of_edges() == 0:
        return {"attribute": attribute, "error": "No edges between nodes with this attribute."}

    try:
        observed = nx.attribute_assortativity_coefficient(H, attribute)
    except Exception as e:
        return {"attribute": attribute, "error": str(e)}

    # Permutation test: shuffle attribute labels, recompute
    node_list = list(H.nodes())
    attr_values = [H.nodes[n][attribute] for n in node_list]
    null_distribution = []
    temp_attr = f"_perm_{attribute}"

    for _ in range(n_permutations):
        shuffled = attr_values.copy()
        rng.shuffle(shuffled)
        for node, val in zip(node_list, shuffled):
            H.nodes[node][temp_attr] = val
        try:
            null_val = nx.attribute_assortativity_coefficient(H, temp_attr)
            null_distribution.append(null_val)
        except Exception:
            pass

    for node in node_list:
        H.nodes[node].pop(temp_attr, None)

    if not null_distribution:
        return {"attribute": attribute, "error": "Permutation test failed."}

    null_mean = pystats.mean(null_distribution)
    null_std = pystats.stdev(null_distribution) if len(null_distribution) > 1 else 0.001

    p_value_twotail = sum(1 for x in null_distribution if abs(x) >= abs(observed)) / len(null_distribution)
    z_score = (observed - null_mean) / max(null_std, 1e-10)

    if observed > 0 and p_value_twotail < 0.05:
        interpretation = "significant_homophily"
    elif observed < 0 and p_value_twotail < 0.05:
        interpretation = "significant_heterophily"
    else:
        interpretation = "not_significant"

    return {
        "attribute": attribute,
        "observed": round(observed, 4),
        "null_mean": round(null_mean, 4),
        "null_std": round(null_std, 4),
        "z_score": round(z_score, 2),
        "p_value": round(p_value_twotail, 4),
        "significant": p_value_twotail < 0.05,
        "interpretation": interpretation,
        "n_nodes_with_attr": len(nodes_with_attr),
        "n_categories": len(categories),
        "categories": categories,
        "n_permutations": len(null_distribution),
    }


def compute_all_assortativity(
    G: nx.Graph,
    attributes: Optional[List[str]] = None,
    n_permutations: int = 1000,
) -> List[Dict[str, Any]]:
    """Run assortativity tests on all available categorical attributes."""
    if attributes is None:
        attributes = _detect_categorical_attributes(G)

    results = []
    for attr in attributes:
        logger.info("Computing assortativity for '%s'...", attr)
        result = compute_assortativity(G, attr, n_permutations=n_permutations)
        results.append(result)

    results.sort(key=lambda r: abs(r.get("observed", 0)), reverse=True)
    return results


# ===========================================================================
# 4. Mixing matrix
#    (ERGM nodemix analog — "Which cross-category pairings are
#     over/under-represented?")
# ===========================================================================

def compute_mixing_matrix(
    G: nx.Graph,
    attribute: str,
    max_categories: int = 25,
) -> Dict[str, Any]:
    """
    Compute observed and expected mixing matrices with standardized residuals.

    This is the Python equivalent of ERGM's nodemix() term. The mixing
    matrix M[i][j] counts edges between nodes of category i and j.
    Standardized residuals identify significantly over/under-represented
    pairings (|residual| > 2 is notable).
    """
    attr_dict = nx.get_node_attributes(G, attribute)
    nodes_with_attr = set(attr_dict.keys())

    if len(nodes_with_attr) < 10:
        return {"attribute": attribute, "error": "Too few nodes with this attribute."}

    categories = sorted(set(attr_dict.values()))
    if len(categories) > max_categories:
        return {
            "attribute": attribute,
            "error": f"Too many categories ({len(categories)}). Max is {max_categories}.",
        }
    if len(categories) < 2:
        return {"attribute": attribute, "error": "Need at least 2 categories."}

    cat_index = {cat: i for i, cat in enumerate(categories)}
    n = len(categories)

    observed = np.zeros((n, n), dtype=float)
    for u, v in G.edges():
        if u in attr_dict and v in attr_dict:
            i = cat_index[attr_dict[u]]
            j = cat_index[attr_dict[v]]
            w = G.edges[u, v].get("weight", 1)
            observed[i][j] += w
            if i != j:
                observed[j][i] += w

    total_weight = observed.sum() / 2
    if total_weight == 0:
        return {"attribute": attribute, "error": "No edges between nodes with this attribute."}

    group_sizes = Counter(attr_dict.values())
    total_nodes = sum(group_sizes.values())

    expected = np.zeros((n, n), dtype=float)
    for i, cat_i in enumerate(categories):
        for j, cat_j in enumerate(categories):
            p_i = group_sizes[cat_i] / total_nodes
            p_j = group_sizes[cat_j] / total_nodes
            expected[i][j] = p_i * p_j * total_weight * 2

    with np.errstate(divide="ignore", invalid="ignore"):
        residuals = np.where(
            expected > 0,
            (observed - expected) / np.sqrt(np.maximum(expected, 1e-10)),
            0.0,
        )

    notable_pairs = []
    for i in range(n):
        for j in range(i, n):
            if abs(residuals[i][j]) > 2:
                notable_pairs.append({
                    "category_a": categories[i],
                    "category_b": categories[j],
                    "observed": float(observed[i][j]),
                    "expected": round(float(expected[i][j]), 1),
                    "residual": round(float(residuals[i][j]), 2),
                    "direction": "overrepresented" if residuals[i][j] > 0 else "underrepresented",
                })

    notable_pairs.sort(key=lambda p: abs(p["residual"]), reverse=True)

    return {
        "attribute": attribute,
        "categories": categories,
        "group_sizes": {cat: group_sizes[cat] for cat in categories},
        "observed": observed.tolist(),
        "expected": np.round(expected, 1).tolist(),
        "residuals": np.round(residuals, 2).tolist(),
        "notable_pairs": notable_pairs,
        "total_edges_counted": int(total_weight),
    }


# ===========================================================================
# 5. Degree distribution analysis
#    (ERGM degree() analog — "Is the network dominated by hubs?")
# ===========================================================================

def compute_degree_distribution(
    G: nx.Graph,
    hub_threshold_std: float = 2.0,
) -> Dict[str, Any]:
    """
    Analyze the degree distribution. Complements client-side centrality
    rankings with Gini coefficient, power-law fit, and hub identification.
    """
    if G.number_of_nodes() == 0:
        return {"error": "Empty graph."}

    degrees = [d for _, d in G.degree()]
    degree_mean = pystats.mean(degrees)
    degree_std = pystats.stdev(degrees) if len(degrees) > 1 else 0

    gini = _gini_coefficient(degrees)
    hist = dict(sorted(Counter(degrees).items()))

    hub_cutoff = degree_mean + hub_threshold_std * degree_std
    hubs = []
    for node, deg in sorted(G.degree(), key=lambda x: x[1], reverse=True):
        if deg >= hub_cutoff:
            hubs.append({
                "entity_id": node,
                "name": G.nodes[node].get("name", "Unknown"),
                "type": G.nodes[node].get("type", "UNKNOWN"),
                "degree": deg,
                "weighted_degree": dict(G.degree(weight="weight")).get(node, 0),
            })
        if len(hubs) >= 30:
            break

    powerlaw_result = _test_powerlaw_fit(degrees)

    sorted_degrees = sorted(degrees)
    n = len(sorted_degrees)

    return {
        "stats": {
            "min": min(degrees),
            "max": max(degrees),
            "mean": round(degree_mean, 2),
            "median": pystats.median(degrees),
            "std": round(degree_std, 2),
            "p90": sorted_degrees[int(n * 0.9)] if n > 10 else None,
            "p95": sorted_degrees[int(n * 0.95)] if n > 20 else None,
            "p99": sorted_degrees[int(n * 0.99)] if n > 100 else None,
        },
        "gini_coefficient": round(gini, 4),
        "gini_interpretation": (
            "high_inequality" if gini > 0.6 else
            "moderate_inequality" if gini > 0.4 else
            "low_inequality"
        ),
        "histogram": hist,
        "hubs": hubs,
        "hub_cutoff": round(hub_cutoff, 1),
        "powerlaw_fit": powerlaw_result,
    }


# ===========================================================================
# 6. Configuration model comparison
#    (ERGM simulate() analog — "Does the observed structure differ from
#     what the degree sequence alone produces?")
# ===========================================================================

def compare_to_random_graphs(
    G: nx.Graph,
    n_simulations: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compare observed network to random graphs with the same degree sequence.

    While the client-side panel already computes clustering and modularity
    for the active slice, this test establishes whether those values are
    SIGNIFICANTLY different from random — the key ERGM insight.
    """
    if G.number_of_nodes() < 20 or G.number_of_edges() < 20:
        return {"error": "Graph too small for meaningful comparison."}

    rng = np.random.RandomState(seed)

    observed_clustering = nx.average_clustering(G, weight=None)
    observed_modularity = _compute_modularity(G)

    largest_cc = max(nx.connected_components(G), key=len)
    H = G.subgraph(largest_cc)
    try:
        observed_avg_path = nx.average_shortest_path_length(H)
    except Exception:
        observed_avg_path = None

    degree_seq = [d for _, d in G.degree()]
    null_clustering = []
    null_modularity = []
    null_avg_path = []

    for i in range(n_simulations):
        try:
            R = nx.configuration_model(degree_seq, seed=rng.randint(0, 2**31))
            R = nx.Graph(R)
            R.remove_edges_from(nx.selfloop_edges(R))

            null_clustering.append(nx.average_clustering(R, weight=None))
            null_modularity.append(_compute_modularity(R))

            if observed_avg_path is not None:
                r_largest = max(nx.connected_components(R), key=len)
                RH = R.subgraph(r_largest)
                if RH.number_of_nodes() >= 3:
                    null_avg_path.append(nx.average_shortest_path_length(RH))
        except Exception as e:
            logger.debug("Random graph %d failed: %s", i, e)

    def _compare(observed_val, null_vals, label):
        if observed_val is None or not null_vals:
            return {"metric": label, "error": "Could not compute."}
        nm = pystats.mean(null_vals)
        ns = pystats.stdev(null_vals) if len(null_vals) > 1 else 0.001
        z = (observed_val - nm) / max(ns, 1e-10)
        return {
            "metric": label,
            "observed": round(observed_val, 4),
            "null_mean": round(nm, 4),
            "null_std": round(ns, 4),
            "z_score": round(z, 2),
            "significant": abs(z) > 1.96,
            "direction": "higher" if z > 0 else "lower",
            "n_simulations": len(null_vals),
        }

    return {
        "clustering": _compare(observed_clustering, null_clustering, "clustering_coefficient"),
        "modularity": _compare(observed_modularity, null_modularity, "modularity"),
        "avg_path_length": _compare(observed_avg_path, null_avg_path, "average_path_length"),
    }


# ===========================================================================
# 7. Community detection
# ===========================================================================

def detect_communities(
    G: nx.Graph,
    attributes_to_compare: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Detect communities and measure how well known attributes predict them.

    NMI (Normalized Mutual Information) scores reveal whether detected
    communities align with entity type, occupation, department, etc.
    """
    if G.number_of_nodes() < 5:
        return {"error": "Graph too small for community detection."}

    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G, random_state=42)
        method = "louvain"
    except ImportError:
        try:
            communities_gen = nx.community.greedy_modularity_communities(G)
            partition = {}
            for comm_id, comm_nodes in enumerate(communities_gen):
                for node in comm_nodes:
                    partition[node] = comm_id
            method = "greedy_modularity"
        except Exception as e:
            return {"error": f"Community detection failed: {e}"}

    community_groups: Dict[int, List[str]] = defaultdict(list)
    for node, comm_id in partition.items():
        community_groups[comm_id].append(node)

    community_sets = [set(nodes) for nodes in community_groups.values()]
    try:
        modularity = nx.community.modularity(G, community_sets)
    except Exception:
        modularity = _compute_modularity(G)

    community_summaries = []
    for comm_id in sorted(community_groups.keys()):
        nodes = community_groups[comm_id]
        subgraph = G.subgraph(nodes)

        top_nodes = sorted(
            [(n, subgraph.degree(n)) for n in nodes],
            key=lambda x: x[1], reverse=True
        )[:5]

        summary = {
            "community_id": comm_id,
            "size": len(nodes),
            "internal_edges": subgraph.number_of_edges(),
            "top_members": [
                {"entity_id": n, "name": G.nodes[n].get("name", "?"), "degree": d}
                for n, d in top_nodes
            ],
        }

        if attributes_to_compare:
            composition = {}
            for attr in attributes_to_compare:
                vals = [G.nodes[n].get(attr) for n in nodes if G.nodes[n].get(attr)]
                if vals:
                    composition[attr] = dict(Counter(vals).most_common(10))
            summary["composition"] = composition

        community_summaries.append(summary)

    # NMI: how well each attribute predicts community membership
    nmi_scores = {}
    if attributes_to_compare:
        comm_labels = [partition.get(n, -1) for n in G.nodes()]
        for attr in attributes_to_compare:
            attr_labels = [G.nodes[n].get(attr, "__none__") for n in G.nodes()]
            nmi = _normalized_mutual_information(comm_labels, attr_labels)
            nmi_scores[attr] = round(nmi, 4)

    community_summaries.sort(key=lambda c: c["size"], reverse=True)

    return {
        "method": method,
        "n_communities": len(community_groups),
        "modularity": round(modularity, 4),
        "communities": community_summaries,
        "nmi_scores": nmi_scores,
        "node_assignments": {n: c for n, c in partition.items()},
    }


# ===========================================================================
# 8. Gatekeeper / bridge analysis
# ===========================================================================

def compute_gatekeepers(
    G: nx.Graph,
    limit: int = 30,
) -> Dict[str, Any]:
    """
    Identify gatekeeper entities that bridge otherwise separate communities.

    Bridge score = betweenness_centrality / log(degree + 1)

    This complements the client-side betweenness ranking by normalizing
    for degree — revealing true structural bridges, not just hubs.
    """
    if G.number_of_nodes() < 10:
        return {"error": "Graph too small for gatekeeper analysis."}

    k = min(G.number_of_nodes(), 500)
    betweenness = nx.betweenness_centrality(G, k=k, seed=42)
    degree_dict = dict(G.degree())

    gatekeepers = []
    for node in G.nodes():
        bc = betweenness.get(node, 0)
        deg = degree_dict.get(node, 0)
        if deg == 0:
            continue

        bridge_score = bc / math.log(deg + 1)
        gatekeepers.append({
            "entity_id": node,
            "name": G.nodes[node].get("name", "Unknown"),
            "type": G.nodes[node].get("type", "UNKNOWN"),
            "degree": deg,
            "betweenness_centrality": round(bc, 6),
            "bridge_score": round(bridge_score, 6),
        })

    gatekeepers.sort(key=lambda g: g["bridge_score"], reverse=True)

    return {
        "gatekeepers": gatekeepers[:limit],
        "total_nodes_analyzed": G.number_of_nodes(),
        "betweenness_approximation_k": k,
    }


# ===========================================================================
# 9. Full summary
# ===========================================================================

def compute_full_statistics(
    G: nx.Graph,
    attributes: Optional[List[str]] = None,
    n_permutations: int = 1000,
    n_simulations: int = 100,
) -> Dict[str, Any]:
    """Run all statistical analyses and return a comprehensive report."""
    if attributes is None:
        attributes = _detect_categorical_attributes(G)

    logger.info("Running full statistics: %d nodes, %d edges, attributes=%s",
                G.number_of_nodes(), G.number_of_edges(), attributes)

    results: Dict[str, Any] = {
        "computed_at": datetime.utcnow().isoformat(),
        "graph_summary": compute_graph_summary(G),
    }

    logger.info("Computing assortativity...")
    results["assortativity"] = compute_all_assortativity(G, attributes, n_permutations)

    logger.info("Computing degree distribution...")
    results["degree_distribution"] = compute_degree_distribution(G)

    logger.info("Detecting communities...")
    results["communities"] = detect_communities(G, attributes)

    logger.info("Computing gatekeepers...")
    results["gatekeepers"] = compute_gatekeepers(G)

    logger.info("Comparing to random graphs (%d simulations)...", n_simulations)
    results["comparison_to_random"] = compare_to_random_graphs(G, n_simulations)

    logger.info("Full statistics complete.")
    return results


# ===========================================================================
# Internal helpers
# ===========================================================================

def _detect_categorical_attributes(G: nx.Graph, max_categories: int = 50) -> List[str]:
    """Auto-detect categorical string attributes on nodes."""
    if G.number_of_nodes() == 0:
        return []

    sample_node = next(iter(G.nodes()))
    all_attrs = set(G.nodes[sample_node].keys())
    exclude = {"name", "mention_count", "entity_id"}
    candidate_attrs = all_attrs - exclude

    categorical = []
    for attr in sorted(candidate_attrs):
        values = [G.nodes[n].get(attr) for n in G.nodes() if G.nodes[n].get(attr) is not None]
        if not values or not isinstance(values[0], str):
            continue
        unique_count = len(set(values))
        if 2 <= unique_count <= max_categories:
            categorical.append(attr)

    return categorical


def _gini_coefficient(values: List[int]) -> float:
    if not values or all(v == 0 for v in values):
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    cumsum = sum((2 * i - n + 1) * v for i, v in enumerate(sorted_vals))
    return cumsum / (n * sum(sorted_vals)) if sum(sorted_vals) > 0 else 0.0


def _test_powerlaw_fit(degrees: List[int]) -> Dict[str, Any]:
    try:
        from scipy import stats as sp_stats
        nonzero = [d for d in degrees if d >= 1]
        if len(nonzero) < 20:
            return {"test": "skipped", "reason": "Too few non-zero degrees."}

        xmin = min(nonzero)
        alpha = 1 + len(nonzero) / sum(np.log(np.array(nonzero) / xmin))
        fitted_samples = (np.random.pareto(alpha - 1, len(nonzero)) + 1) * xmin
        ks_stat, ks_p = sp_stats.ks_2samp(nonzero, fitted_samples)

        return {
            "test": "kolmogorov_smirnov",
            "alpha_estimate": round(alpha, 3),
            "ks_statistic": round(ks_stat, 4),
            "ks_p_value": round(ks_p, 4),
            "plausible_powerlaw": ks_p > 0.1,
        }
    except ImportError:
        return {"test": "skipped", "reason": "scipy not available."}
    except Exception as e:
        return {"test": "error", "reason": str(e)}


def _compute_modularity(G: nx.Graph) -> float:
    try:
        communities = nx.community.greedy_modularity_communities(G)
        return nx.community.modularity(G, communities)
    except Exception:
        return 0.0


def _normalized_mutual_information(labels_a: list, labels_b: list) -> float:
    if len(labels_a) != len(labels_b) or not labels_a:
        return 0.0
    n = len(labels_a)
    contingency: Dict[tuple, int] = Counter(zip(labels_a, labels_b))
    row_counts = Counter(labels_a)
    col_counts = Counter(labels_b)

    mi = 0.0
    for (a, b), count in contingency.items():
        if count == 0:
            continue
        p_ab = count / n
        p_a = row_counts[a] / n
        p_b = col_counts[b] / n
        mi += p_ab * math.log(p_ab / (p_a * p_b))

    h_a = -sum((c / n) * math.log(c / n) for c in row_counts.values() if c > 0)
    h_b = -sum((c / n) * math.log(c / n) for c in col_counts.values() if c > 0)

    if h_a + h_b == 0:
        return 0.0
    return 2 * mi / (h_a + h_b)
