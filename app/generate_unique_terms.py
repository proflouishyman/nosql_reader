#!/usr/bin/env python3
"""
Generate unique words and phrases for the search-terms explorer.

This is the canonical term-generation entrypoint for the app.
"""

import argparse
import logging
import os
import re
from collections import Counter, defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Tuple

from pymongo import ASCENDING, DESCENDING, UpdateOne

from database_setup import get_client, get_db

# Keep tokenization compatible with the existing search-terms behavior.
WORD_PATTERN = re.compile(r"\w+")


logger = logging.getLogger("GenerateUniqueTerms")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)


def extract_terms(text: str) -> Tuple[Counter, Counter]:
    """Return word and bigram counters for a text value."""
    words = WORD_PATTERN.findall(text.lower())
    phrases = [" ".join(pair) for pair in zip(words, words[1:])]
    return Counter(words), Counter(phrases)


def text_from_field_value(value: Any) -> str:
    """
    Convert supported field values into text.

    Contract-preserving behavior:
    - string fields are indexed directly
    - list[str] fields are joined and indexed
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return " ".join(value)
    return ""


def collect_document_terms(document: Dict[str, Any]) -> DefaultDict[str, Dict[str, Counter]]:
    """Collect per-field term counts from one MongoDB document."""
    collected: DefaultDict[str, Dict[str, Counter]] = defaultdict(
        lambda: {"word": Counter(), "phrase": Counter()}
    )

    for field, value in document.items():
        if field == "_id":
            continue

        text = text_from_field_value(value)
        if not text:
            continue

        word_counts, phrase_counts = extract_terms(text)
        if word_counts:
            collected[field]["word"].update(word_counts)
        if phrase_counts:
            collected[field]["phrase"].update(phrase_counts)

    return collected


def merge_field_terms(
    target: DefaultDict[str, Dict[str, Counter]],
    source: DefaultDict[str, Dict[str, Counter]],
) -> None:
    """Merge one term tree into another."""
    for field, typed_counters in source.items():
        target[field]["word"].update(typed_counters["word"])
        target[field]["phrase"].update(typed_counters["phrase"])


def chunked(items: Iterable[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    """Yield a cursor/iterable in fixed-size chunks."""
    batch: List[Dict[str, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_bulk_operations(
    batch_terms: DefaultDict[str, Dict[str, Counter]],
) -> List[UpdateOne]:
    """Convert aggregated batch counts into MongoDB upsert operations."""
    operations: List[UpdateOne] = []

    for field, typed_counters in batch_terms.items():
        if not field:
            continue

        for term_type in ("word", "phrase"):
            for term, frequency in typed_counters[term_type].items():
                if not term or frequency <= 0:
                    continue

                operations.append(
                    UpdateOne(
                        {"term": term, "field": field, "type": term_type},
                        {"$inc": {"frequency": int(frequency)}},
                        upsert=True,
                    )
                )

    return operations


def ensure_unique_terms_indexes(db) -> None:
    """Create expected indexes for the unique_terms collection."""
    unique_terms = db["unique_terms"]
    unique_terms.create_index(
        [("term", ASCENDING), ("field", ASCENDING), ("type", ASCENDING)],
        unique=True,
        name="unique_term_field_type",
    )
    unique_terms.create_index(
        [("field", ASCENDING), ("type", ASCENDING), ("frequency", DESCENDING)],
        name="field_type_frequency_idx",
    )


def generate_unique_terms(db, batch_size: int) -> None:
    """
    Rebuild the unique_terms collection from documents.

    Rebuild semantics intentionally avoid stale/incremental drift.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    documents_collection = db["documents"]
    unique_terms_collection = db["unique_terms"]

    total_documents = documents_collection.count_documents({})
    logger.info("Found %s documents to index.", total_documents)
    if total_documents == 0:
        logger.info("No documents found. Skipping unique term generation.")
        return

    ensure_unique_terms_indexes(db)

    # Full rebuild keeps counts deterministic and prevents double-counting.
    deleted = unique_terms_collection.delete_many({})
    logger.info("Cleared %s existing unique_terms documents.", deleted.deleted_count)

    cursor = documents_collection.find({}, no_cursor_timeout=True)
    processed_count = 0

    try:
        for batch_docs in chunked(cursor, batch_size):
            batch_terms: DefaultDict[str, Dict[str, Counter]] = defaultdict(
                lambda: {"word": Counter(), "phrase": Counter()}
            )

            # Aggregate in Python once per batch to avoid one DB write per term token.
            for doc in batch_docs:
                merge_field_terms(batch_terms, collect_document_terms(doc))

            operations = build_bulk_operations(batch_terms)
            if operations:
                unique_terms_collection.bulk_write(operations, ordered=False)

            processed_count += len(batch_docs)
            logger.info(
                "Processed %s/%s documents (%s operations in batch).",
                processed_count,
                total_documents,
                len(operations),
            )
    finally:
        cursor.close()

    total_terms = unique_terms_collection.count_documents({})
    logger.info("Unique term generation complete. Stored %s term records.", total_terms)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate unique words and phrases.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("BATCH_SIZE", "250")),
        help="Number of documents per indexing batch (default: env BATCH_SIZE or 250).",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    client = get_client()
    try:
        db = get_db(client)
        generate_unique_terms(db, batch_size=args.batch_size)
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
