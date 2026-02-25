#!/usr/bin/env python3
"""
normalize_occupations.py
Fuzzy-matches raw occupation_primary strings against a curated canonical list
and writes occupation_normalized + occupation_norm_score + occupation_norm_method
back to the demographics SQLite database.

Raw field (occupation_primary) is NEVER modified.

Usage:
    python /app/setup/normalize_occupations.py                    # normalize all unmatched
    python /app/setup/normalize_occupations.py --force            # re-normalize everything
    python /app/setup/normalize_occupations.py --threshold 75     # tune match threshold
    python /app/setup/normalize_occupations.py --preview          # print mapping, don't write
    python /app/setup/normalize_occupations.py --show-unmatched   # list strings below threshold
"""

import argparse
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz
from rapidfuzz import process as rf_process

# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent.parent  # /app
DB_PATH = APP_DIR / "data" / "demographics.db"

# ---------------------------------------------------------------------------
# Canonical occupation list
# Curated for B&O Railroad early 20th century records.
# ---------------------------------------------------------------------------
CANONICAL_OCCUPATIONS: List[str] = [
    # Operations — train crew
    "Brakeman",
    "Conductor",
    "Engineer",
    "Fireman",
    "Flagman",
    "Switchman",
    "Yardmaster",
    "Yard Foreman",
    "Trainmaster",

    # Operations — track & maintenance
    "Section Foreman",
    "Section Hand",
    "Track Laborer",
    "Track Walker",
    "Trackman",
    "Gang Foreman",
    "Bridge Carpenter",
    "Bridge Foreman",
    "Bridge and Building Carpenter",
    "Maintenance of Way Laborer",

    # Shops & mechanical
    "Machinist",
    "Boilermaker",
    "Blacksmith",
    "Carpenter",
    "Car Inspector",
    "Car Repairer",
    "Painter",
    "Pipefitter",
    "Sheet Metal Worker",
    "Electrician",
    "Apprentice Machinist",
    "Helper",

    # Freight & warehousing
    "Freight Handler",
    "Freight Agent",
    "Agent",
    "Clerk",
    "Station Agent",
    "Baggageman",
    "Express Messenger",

    # Administrative & clerical
    "Stenographer",
    "Bookkeeper",
    "Cashier",
    "Typewriter Operator",
    "Office Clerk",
    "Messenger",

    # Supervisory
    "Foreman",
    "Supervisor",
    "Inspector",
    "Master Mechanic",
    "Roadmaster",
    "Division Superintendent",
    "General Foreman",

    # Labor (generic)
    "Laborer",
    "Extra Gang Laborer",

    # Signals & telegraph
    "Telegraph Operator",
    "Signal Maintainer",
    "Lineman",

    # Other railroad
    "Hostler",
    "Car Cleaner",
    "Watchman",
    "Crossing Watchman",
    "Gateman",
    "Porter",
    "Janitor",
    "Pumper",
    "Wiper",
]

# ---------------------------------------------------------------------------
# Additional manual overrides — applied BEFORE fuzzy matching.
# ---------------------------------------------------------------------------
MANUAL_OVERRIDES: Dict[str, str] = {
    "b&b carpenter": "Bridge and Building Carpenter",
    "b & b carpenter": "Bridge and Building Carpenter",
    "b.&b. carpenter": "Bridge and Building Carpenter",
    "mow laborer": "Maintenance of Way Laborer",
    "m.o.w. laborer": "Maintenance of Way Laborer",
    "extra gang": "Extra Gang Laborer",
    "e.g. laborer": "Extra Gang Laborer",
    "tel. operator": "Telegraph Operator",
    "tel operator": "Telegraph Operator",
}


def normalize_for_matching(raw_value: str) -> str:
    """Lowercase and strip punctuation noise for more consistent matching."""
    import re

    normalized = raw_value.lower().strip()
    normalized = re.sub(r"[.\-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _build_override_map() -> Dict[str, str]:
    """Precompute normalized manual overrides once per run for deterministic matching."""
    return {normalize_for_matching(key): value for key, value in MANUAL_OVERRIDES.items()}


def match_occupation(
    raw_value: str,
    canonicals: List[str],
    threshold: int,
    normalized_overrides: Dict[str, str],
    normalized_canonicals: List[str],
) -> Tuple[Optional[str], float, str]:
    """
    Returns (canonical_label, score, method).
    method is one of: 'manual', 'exact', 'fuzzy', 'unmatched'
    """
    if not raw_value or not raw_value.strip():
        return None, 0.0, "unmatched"

    normalized_raw = normalize_for_matching(raw_value)

    # Manual override for known noisy strings.
    if normalized_raw in normalized_overrides:
        return normalized_overrides[normalized_raw], 1.0, "manual"

    # Exact normalized match against canonical list.
    for index, canonical in enumerate(canonicals):
        if normalized_raw == normalized_canonicals[index]:
            return canonical, 1.0, "exact"

    # Fuzzy matching handles order/spacing differences in OCR-heavy strings.
    result = rf_process.extractOne(
        normalized_raw,
        normalized_canonicals,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=threshold,
    )
    if result is None:
        return None, 0.0, "unmatched"

    _, score, matched_idx = result
    return canonicals[matched_idx], round(score / 100.0, 4), "fuzzy"


def get_conn(read_only: bool = False) -> sqlite3.Connection:
    """Open demographics DB in read-only or read-write mode."""
    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        print("Run build_demographics_db.py first.")
        sys.exit(1)

    uri = f"file:{DB_PATH}{'?mode=ro' if read_only else ''}"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def ensure_columns(conn: sqlite3.Connection) -> None:
    """Add normalization columns/index if missing for old databases."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(people)").fetchall()}
    migrations = [
        ("occupation_normalized", "TEXT"),
        ("occupation_norm_score", "REAL"),
        ("occupation_norm_method", "TEXT"),
    ]

    for column_name, column_type in migrations:
        if column_name not in existing:
            conn.execute(f"ALTER TABLE people ADD COLUMN {column_name} {column_type}")
            print(f"  Added column: {column_name}")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_occupation_norm ON people(occupation_normalized)")
    conn.commit()


def run(
    threshold: int = 82,
    force: bool = False,
    preview: bool = False,
    show_unmatched: bool = False,
) -> None:
    """Normalize occupations with optional preview and threshold tuning."""
    print("=" * 70)
    print("OCCUPATION NORMALIZER")
    print("=" * 70)
    print(f"DB:         {DB_PATH}")
    print(f"Threshold:  {threshold} (below threshold => unmatched)")
    print(f"Canonicals: {len(CANONICAL_OCCUPATIONS)} labels")
    print(f"Overrides:  {len(MANUAL_OVERRIDES)} manual rules")
    print(f"Mode:       {'PREVIEW (no writes)' if preview else 'LIVE'}")
    print(f"Force:      {force}")
    print("=" * 70)

    conn = get_conn(read_only=preview)
    if not preview:
        ensure_columns(conn)

    if force or preview:
        rows = conn.execute(
            "SELECT DISTINCT occupation_primary FROM people WHERE occupation_primary IS NOT NULL"
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT DISTINCT occupation_primary FROM people
            WHERE occupation_primary IS NOT NULL
              AND (occupation_normalized IS NULL OR occupation_norm_method IS NULL)
            """
        ).fetchall()

    raw_strings = [row["occupation_primary"] for row in rows]
    print(f"\nDistinct raw occupations to process: {len(raw_strings)}")

    if not raw_strings:
        print("Nothing to do — all occupations are already normalized. Use --force to redo.")
        conn.close()
        return

    normalized_overrides = _build_override_map()
    normalized_canonicals = [normalize_for_matching(label) for label in CANONICAL_OCCUPATIONS]

    mapping: Dict[str, Tuple[Optional[str], float, str]] = {}
    unmatched = []

    for raw_value in raw_strings:
        canonical, score, method = match_occupation(
            raw_value,
            CANONICAL_OCCUPATIONS,
            threshold,
            normalized_overrides,
            normalized_canonicals,
        )
        mapping[raw_value] = (canonical, score, method)
        if canonical is None:
            unmatched.append(raw_value)

    matched = len(raw_strings) - len(unmatched)
    print(f"\nResults: {matched}/{len(raw_strings)} matched ({(matched / len(raw_strings) * 100):.1f}%)")
    print(f"         {len(unmatched)} unmatched")

    print("\nSample mappings (first 30):")
    print(f"  {'RAW':<40} {'CANONICAL':<35} {'SCORE':>8}  METHOD")
    print("  " + "-" * 94)
    for raw_value in raw_strings[:30]:
        canonical, score, method = mapping[raw_value]
        score_display = f"{score:.2f}" if score else "—"
        print(f"  {raw_value:<40} {(canonical or '(unmatched)'):<35} {score_display:>8}  {method}")

    if len(raw_strings) > 30:
        print(f"  ... and {len(raw_strings) - 30} more")

    if show_unmatched or preview:
        print(f"\nUnmatched strings ({len(unmatched)}):")
        for raw_value in sorted(unmatched):
            print(f"  '{raw_value}'")

    if preview:
        print("\n[PREVIEW MODE] No writes performed.")
        projected = Counter(label for (label, _, _) in mapping.values() if label)
        print("\nProjected canonical distribution (top 20):")
        for canonical, count in projected.most_common(20):
            print(f"  {canonical:<40} {count:>4}")
        conn.close()
        return

    print("\nWriting to database...")
    for raw_value, (canonical, score, method) in mapping.items():
        conn.execute(
            """
            UPDATE people
               SET occupation_normalized = ?,
                   occupation_norm_score = ?,
                   occupation_norm_method = ?
             WHERE occupation_primary = ?
            """,
            (canonical, score if canonical else None, method, raw_value),
        )

    conn.commit()

    total_people = conn.execute("SELECT COUNT(*) as n FROM people").fetchone()["n"]
    normalized_people = conn.execute(
        "SELECT COUNT(*) as n FROM people WHERE occupation_normalized IS NOT NULL"
    ).fetchone()["n"]

    print(f"\n✅ Done. Updated {len(mapping)} distinct raw occupation strings.")
    print(f"   {normalized_people}/{total_people} persons now have a normalized occupation.")

    print("\nFinal canonical occupation distribution (top 20):")
    distribution = conn.execute(
        """
        SELECT occupation_normalized, COUNT(*) as n
          FROM people
         WHERE occupation_normalized IS NOT NULL
         GROUP BY occupation_normalized
         ORDER BY n DESC
         LIMIT 20
        """
    ).fetchall()
    for row in distribution:
        print(f"  {row['occupation_normalized']:<40} {row['n']:>4} persons")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuzzy-normalize occupation strings in demographics.db")
    parser.add_argument(
        "--threshold",
        type=int,
        default=82,
        help="Minimum fuzzy score 0-100 to accept a match (default: 82)",
    )
    parser.add_argument("--force", action="store_true", help="Re-normalize all rows, not just unprocessed ones")
    parser.add_argument("--preview", action="store_true", help="Print the mapping without writing")
    parser.add_argument("--show-unmatched", action="store_true", help="Print all values below threshold")
    cli_args = parser.parse_args()

    run(
        threshold=cli_args.threshold,
        force=cli_args.force,
        preview=cli_args.preview,
        show_unmatched=cli_args.show_unmatched,
    )
