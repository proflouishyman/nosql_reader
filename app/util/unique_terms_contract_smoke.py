#!/usr/bin/env python3
"""
Unique terms explorer contract smoke test.

This module can run standalone or be imported by the master E2E suite.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Added explicit app-root insertion so standalone execution resolves local modules reliably.
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from database_setup import get_client, get_db
from routes import app


def _require(condition: bool, message: str) -> None:
    """Raise AssertionError with a consistent message when a contract fails."""
    if not condition:
        raise AssertionError(message)


def _pick_valid_field(db: Any) -> str:
    """Select a field with indexed terms to use for endpoint contract checks."""
    unique_terms_collection = db["unique_terms"]

    # Added deterministic preference for known text fields used by the UI.
    for candidate in ("ocr_text", "summary", "filename"):
        if unique_terms_collection.count_documents({"field": candidate, "type": "word"}) > 0:
            return candidate

    fallback = unique_terms_collection.find_one({"type": "word"}, {"field": 1})
    _require(fallback is not None and isinstance(fallback.get("field"), str), "no usable unique_terms field found")
    return str(fallback["field"])


def run_unique_terms_contract_check(client: Any, db: Any) -> str:
    """
    Validate DB and API contracts for the unique terms explorer.

    Returns a concise detail string on success.
    """
    unique_terms_collection = db["unique_terms"]
    total_terms = unique_terms_collection.count_documents({})
    _require(total_terms > 0, "unique_terms collection is empty")

    # Added schema invariants so regressions are caught before UI checks.
    invalid_type_count = unique_terms_collection.count_documents({"type": {"$nin": ["word", "phrase"]}})
    _require(invalid_type_count == 0, "unique_terms contains invalid type values")

    null_term_count = unique_terms_collection.count_documents({"$or": [{"term": None}, {"term": ""}]})
    _require(null_term_count == 0, "unique_terms contains null/empty term values")

    null_field_count = unique_terms_collection.count_documents({"$or": [{"field": None}, {"field": ""}]})
    _require(null_field_count == 0, "unique_terms contains null/empty field values")

    missing_freq_count = unique_terms_collection.count_documents({"frequency": {"$exists": False}})
    _require(missing_freq_count == 0, "unique_terms contains rows without frequency")

    negative_freq_count = unique_terms_collection.count_documents({"frequency": {"$lt": 0}})
    _require(negative_freq_count == 0, "unique_terms contains negative frequencies")

    index_info = unique_terms_collection.index_information()
    _require("unique_term_field_type" in index_info, "missing unique_term_field_type index")
    _require("field_type_frequency_idx" in index_info, "missing field_type_frequency_idx index")
    _require(bool(index_info["unique_term_field_type"].get("unique")), "unique_term_field_type index must be unique")

    field = _pick_valid_field(db)
    response = client.get(
        "/search-terms",
        query_string={"field": field, "page": 1, "per_page": 5},
        headers={"X-Requested-With": "XMLHttpRequest"},
    )
    _require(response.status_code == 200, f"/search-terms AJAX returned {response.status_code}")

    payload = response.get_json(silent=True)
    _require(isinstance(payload, dict), "/search-terms AJAX payload is not a JSON object")

    required_keys = {"words", "phrases", "unique_words", "unique_phrases", "total_records"}
    _require(required_keys.issubset(payload.keys()), "search-terms payload missing required keys")

    words = payload["words"]
    phrases = payload["phrases"]
    unique_words = payload["unique_words"]
    unique_phrases = payload["unique_phrases"]
    total_records = payload["total_records"]

    _require(isinstance(words, list), "words must be a list")
    _require(isinstance(phrases, list), "phrases must be a list")
    _require(isinstance(unique_words, int) and unique_words >= 0, "unique_words must be a non-negative integer")
    _require(isinstance(unique_phrases, int) and unique_phrases >= 0, "unique_phrases must be a non-negative integer")
    _require(isinstance(total_records, int) and total_records >= 0, "total_records must be a non-negative integer")
    _require(total_records == unique_words + unique_phrases, "total_records must equal unique_words + unique_phrases")
    _require(total_records > 0, "search-terms returned zero total records")

    if words:
        first_word = words[0]
        _require(isinstance(first_word, dict), "words entries must be objects")
        _require({"word", "count"}.issubset(first_word.keys()), "word entry missing word/count keys")

    if phrases:
        first_phrase = phrases[0]
        _require(isinstance(first_phrase, dict), "phrases entries must be objects")
        _require({"phrase", "count"}.issubset(first_phrase.keys()), "phrase entry missing phrase/count keys")

    return (
        f"field={field} total_terms={total_terms} "
        f"unique_words={unique_words} unique_phrases={unique_phrases}"
    )


def main() -> int:
    """CLI entrypoint for local or container smoke checks."""
    db_client = get_client()
    try:
        db = get_db(db_client)
        with app.test_client() as client:
            detail = run_unique_terms_contract_check(client, db)
            print(f"[PASS] Unique terms contract smoke :: {detail}")
        return 0
    except AssertionError as exc:
        print(f"[FAIL] Unique terms contract smoke :: {exc}")
        return 1
    finally:
        db_client.close()


if __name__ == "__main__":
    raise SystemExit(main())
