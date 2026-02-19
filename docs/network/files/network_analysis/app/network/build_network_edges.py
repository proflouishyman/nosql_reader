#!/usr/bin/env python3
# app/network/build_network_edges.py
"""
Build the network_edges collection from existing entity data.

This script is idempotent: it drops and rebuilds the collection on each run.
It reads from documents.entity_refs and linked_entities to construct pairwise
co-occurrence edges between entities that appear in the same document.

Domain-agnostic: works with any entity types present in linked_entities.
Entity type filtering, frequency thresholds, and minimum edge weights are
all configurable.

Usage:
    # Inside Docker container
    python -m app.network.build_network_edges

    # With overrides
    python -m app.network.build_network_edges \
        --entity-types PERSON,GPE \
        --min-weight 3 \
        --max-mentions 1000

    # Dry run (compute stats without writing)
    python -m app.network.build_network_edges --dry-run
"""

import argparse
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING, UpdateOne

# ---------------------------------------------------------------------------
# Allow execution from project root:  python -m app.network.build_network_edges
# ---------------------------------------------------------------------------
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.database_setup import get_client, get_db
from app.network.config import NetworkConfig

# ===========================================================================
# Logging
# ===========================================================================
logger = logging.getLogger("network.build_edges")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    _console = logging.StreamHandler()
    _console.setLevel(logging.INFO)
    _file = logging.FileHandler("build_network_edges.log", mode="a")
    _file.setLevel(logging.DEBUG)
    _fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    _console.setFormatter(_fmt)
    _file.setFormatter(_fmt)
    logger.addHandler(_console)
    logger.addHandler(_file)


# ===========================================================================
# Index creation
# ===========================================================================

def create_indexes(collection) -> None:
    """Create all required indexes on network_edges."""
    logger.info("Creating indexes on network_edges...")

    collection.create_index(
        [("source_id", ASCENDING), ("target_id", ASCENDING), ("edge_type", ASCENDING)],
        unique=True,
        name="unique_edge",
    )
    collection.create_index("source_id", name="idx_source")
    collection.create_index("target_id", name="idx_target")
    collection.create_index("source_type", name="idx_source_type")
    collection.create_index("target_type", name="idx_target_type")
    collection.create_index("weight", name="idx_weight")
    collection.create_index(
        [("source_type", ASCENDING), ("target_type", ASCENDING), ("weight", DESCENDING)],
        name="idx_type_pair_weight",
    )

    logger.info("Indexes created.")


# ===========================================================================
# Entity lookup builder
# ===========================================================================

def build_entity_lookup(
    db,
    entity_types: List[str],
    max_mention_count: int,
) -> Dict[str, Dict[str, Any]]:
    """
    Build an in-memory lookup from linked_entities._id (as string) to entity
    metadata. Filters by entity type and mention frequency.

    Returns
    -------
    dict
        Mapping of str(linked_entities._id) → {canonical_name, type, mention_count}
    """
    linked = db["linked_entities"]
    query: Dict[str, Any] = {}

    # Filter by type if configured (empty list means include all)
    if entity_types:
        query["type"] = {"$in": entity_types}

    # Filter by mention count
    if max_mention_count > 0:
        query["mention_count"] = {"$lte": max_mention_count}

    cursor = linked.find(
        query,
        {"canonical_name": 1, "type": 1, "mention_count": 1},
    )

    lookup = {}
    skipped_types = 0
    skipped_freq = 0

    for entity in cursor:
        eid = str(entity["_id"])
        lookup[eid] = {
            "canonical_name": entity.get("canonical_name", "Unknown"),
            "type": entity.get("type", "UNKNOWN"),
            "mention_count": entity.get("mention_count", 0),
        }

    # Also count what we skipped for logging
    total = linked.count_documents({})
    skipped = total - len(lookup)

    logger.info(
        "Entity lookup built: %d included, %d excluded (of %d total). "
        "Types: %s, max_mentions: %d",
        len(lookup), skipped, total,
        entity_types or "ALL", max_mention_count,
    )

    return lookup


# ===========================================================================
# Pair generation
# ===========================================================================

def generate_pairs(
    entity_ids: List[str],
) -> List[Tuple[str, str]]:
    """
    Generate unique ordered pairs from a list of entity IDs.

    Ordering convention: source_id < target_id (lexicographic on the hex string).
    This prevents duplicate edges (A→B and B→A).
    """
    unique_ids = sorted(set(entity_ids))
    if len(unique_ids) < 2:
        return []
    return list(combinations(unique_ids, 2))


# ===========================================================================
# Main build logic
# ===========================================================================

def build_edges(
    config: NetworkConfig,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Build the network_edges collection.

    Parameters
    ----------
    config : NetworkConfig
        All thresholds and settings.
    dry_run : bool
        If True, compute statistics but don't write to the database.

    Returns
    -------
    dict
        Summary statistics of the build.
    """
    client = get_client()
    db = get_db(client)

    stats = {
        "started_at": datetime.utcnow().isoformat(),
        "config": {
            "entity_types": config.entity_types,
            "max_mention_count": config.max_mention_count,
            "min_edge_weight": config.min_edge_weight,
            "build_batch_size": config.build_batch_size,
        },
        "documents_processed": 0,
        "documents_skipped_no_refs": 0,
        "pairs_generated": 0,
        "edges_written": 0,
        "edges_below_min_weight": 0,
        "dry_run": dry_run,
    }

    # -----------------------------------------------------------------------
    # Step 1: Build entity lookup
    # -----------------------------------------------------------------------
    lookup = build_entity_lookup(
        db,
        entity_types=config.entity_types,
        max_mention_count=config.max_mention_count,
    )

    if not lookup:
        logger.warning("Entity lookup is empty — no edges to build. Check NETWORK_ENTITY_TYPES and NETWORK_MAX_MENTION_COUNT.")
        return stats

    valid_entity_ids: Set[str] = set(lookup.keys())

    # -----------------------------------------------------------------------
    # Step 2: Prepare collection
    # -----------------------------------------------------------------------
    network_edges = db["network_edges"]

    if not dry_run:
        logger.info("Dropping existing network_edges collection...")
        network_edges.drop()
        create_indexes(network_edges)

    # -----------------------------------------------------------------------
    # Step 3: Accumulate edges in memory
    # -----------------------------------------------------------------------
    # Key: (source_id, target_id) → {weight, document_ids}
    edge_accumulator: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
        lambda: {"weight": 0, "document_ids": []}
    )

    documents = db["documents"]
    total_docs = documents.count_documents({})
    logger.info("Processing %d documents in batches of %d...", total_docs, config.build_batch_size)

    cursor = documents.find(
        {"entity_refs": {"$exists": True, "$ne": []}},
        {"_id": 1, "entity_refs": 1},
    ).batch_size(config.build_batch_size)

    t_start = time.time()

    for doc in cursor:
        doc_id = str(doc["_id"])
        raw_refs = doc.get("entity_refs", [])

        # Filter refs to those in our lookup (correct type + frequency)
        valid_refs = [str(ref) for ref in raw_refs if str(ref) in valid_entity_ids]

        if len(valid_refs) < 2:
            stats["documents_skipped_no_refs"] += 1
            stats["documents_processed"] += 1
            continue

        pairs = generate_pairs(valid_refs)
        stats["pairs_generated"] += len(pairs)

        for source_id, target_id in pairs:
            edge_accumulator[(source_id, target_id)]["weight"] += 1
            edge_accumulator[(source_id, target_id)]["document_ids"].append(doc_id)

        stats["documents_processed"] += 1

        if stats["documents_processed"] % 2000 == 0:
            elapsed = time.time() - t_start
            logger.info(
                "  Processed %d/%d docs (%.1fs, %d unique edges so far)",
                stats["documents_processed"],
                total_docs,
                elapsed,
                len(edge_accumulator),
            )

    elapsed = time.time() - t_start
    logger.info(
        "Document scan complete: %d docs in %.1fs. %d raw unique edges accumulated.",
        stats["documents_processed"],
        elapsed,
        len(edge_accumulator),
    )

    # -----------------------------------------------------------------------
    # Step 4: Filter by min_weight and write to DB
    # -----------------------------------------------------------------------
    now = datetime.utcnow()
    operations: List[Any] = []
    batch_size = 1000

    for (source_id, target_id), edge_data in edge_accumulator.items():
        weight = edge_data["weight"]

        if weight < config.min_edge_weight:
            stats["edges_below_min_weight"] += 1
            continue

        source_info = lookup[source_id]
        target_info = lookup[target_id]

        edge_doc = {
            "source_id": source_id,
            "target_id": target_id,
            "source_name": source_info["canonical_name"],
            "target_name": target_info["canonical_name"],
            "source_type": source_info["type"],
            "target_type": target_info["type"],
            "edge_type": "co_occurrence",
            "weight": weight,
            "document_ids": edge_data["document_ids"],
            "created_at": now,
            "updated_at": now,
        }

        if not dry_run:
            operations.append(
                UpdateOne(
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "edge_type": "co_occurrence",
                    },
                    {"$set": edge_doc},
                    upsert=True,
                )
            )

            if len(operations) >= batch_size:
                result = network_edges.bulk_write(operations, ordered=False)
                logger.debug(
                    "  Wrote batch: %d upserted, %d modified",
                    result.upserted_count,
                    result.modified_count,
                )
                operations = []

        stats["edges_written"] += 1

    # Flush remaining operations
    if operations and not dry_run:
        result = network_edges.bulk_write(operations, ordered=False)
        logger.debug(
            "  Wrote final batch: %d upserted, %d modified",
            result.upserted_count,
            result.modified_count,
        )

    stats["completed_at"] = datetime.utcnow().isoformat()
    stats["total_elapsed_seconds"] = round(time.time() - t_start, 2)

    # -----------------------------------------------------------------------
    # Step 5: Log summary
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info("  Documents processed: %d", stats["documents_processed"])
    logger.info("  Documents skipped (< 2 valid entities): %d", stats["documents_skipped_no_refs"])
    logger.info("  Pairs generated: %d", stats["pairs_generated"])
    logger.info("  Edges written: %d", stats["edges_written"])
    logger.info("  Edges filtered (below min_weight=%d): %d", config.min_edge_weight, stats["edges_below_min_weight"])
    logger.info("  Total time: %.1fs", stats["total_elapsed_seconds"])
    if dry_run:
        logger.info("  ** DRY RUN — no data written **")
    logger.info("=" * 60)

    return stats


# ===========================================================================
# CLI entry point
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the network_edges collection from entity co-occurrences.",
    )
    parser.add_argument(
        "--entity-types",
        type=str,
        default=None,
        help="Comma-separated entity types to include (overrides NETWORK_ENTITY_TYPES).",
    )
    parser.add_argument(
        "--min-weight",
        type=int,
        default=None,
        help="Minimum edge weight to persist (overrides NETWORK_MIN_EDGE_WEIGHT).",
    )
    parser.add_argument(
        "--max-mentions",
        type=int,
        default=None,
        help="Exclude entities with mention_count above this (overrides NETWORK_MAX_MENTION_COUNT).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute statistics without writing to the database.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    overrides = {}
    if args.entity_types is not None:
        overrides["entity_types"] = [t.strip() for t in args.entity_types.split(",")]
    if args.min_weight is not None:
        overrides["min_edge_weight"] = args.min_weight
    if args.max_mentions is not None:
        overrides["max_mention_count"] = args.max_mentions

    config = NetworkConfig.from_env(overrides)

    logger.info("Starting network edge build...")
    logger.info("Config: types=%s max_mentions=%d min_weight=%d",
                config.entity_types, config.max_mention_count, config.min_edge_weight)

    stats = build_edges(config, dry_run=args.dry_run)
    return stats


if __name__ == "__main__":
    main()
