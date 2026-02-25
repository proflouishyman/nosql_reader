#!/usr/bin/env python3
"""
build_demographics_db.py
Builds /app/data/demographics.db from person_syntheses MongoDB collection.
One row per person_folder — no double-counting.

Usage:
    python /app/setup/build_demographics_db.py           # skip already processed
    python /app/setup/build_demographics_db.py --force   # reprocess everything
    python /app/setup/build_demographics_db.py --limit N # cap at N persons (testing)
    python /app/setup/build_demographics_db.py --dry-run # print extraction, no write
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Path setup — allow running from any working directory inside the container
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent.parent  # /app
sys.path.insert(0, str(APP_DIR))  # Keep import contract consistent with other setup scripts.

from database_setup import get_client, get_db

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_PATH = APP_DIR / "data" / "demographics.db"
MONGO_DB_NAME = os.environ.get("DB_NAME", "railroad_documents")
SYNTHESES_COLLECTION = "person_syntheses"

# LLM settings — reads from environment, same pattern as rest of codebase
LLM_PROVIDER = os.environ.get("HISTORIAN_AGENT_MODEL_PROVIDER", "ollama")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:32b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS people (
    person_folder        TEXT PRIMARY KEY,
    person_id            TEXT,
    canonical_name       TEXT,
    birth_year           INTEGER,

    -- Demographics (LLM-extracted)
    gender               TEXT DEFAULT 'unknown',
    gender_confidence    TEXT DEFAULT 'low',
    race                 TEXT DEFAULT 'unknown',
    race_confidence      TEXT DEFAULT 'low',
    nationality          TEXT DEFAULT 'unknown',
    nationality_confidence TEXT DEFAULT 'low',

    -- Employment (from synthesis.employment_timeline)
    occupation_primary   TEXT,
    occupations_all      TEXT,   -- JSON array of all unique positions
    occupation_normalized TEXT,  -- Fuzzy-normalized occupation label (raw occupation remains untouched).
    occupation_norm_score REAL,  -- Match score in [0,1] from normalize_occupations.py.
    occupation_norm_method TEXT, -- exact/fuzzy/manual/unmatched classification from normalizer.
    division             TEXT,
    department           TEXT,
    hire_year            INTEGER,
    separation_year      INTEGER,
    separation_type      TEXT,   -- resignation/furlough/death/unknown

    -- Injury stats (from synthesis.injury_history)
    num_injuries         INTEGER DEFAULT 0,
    total_benefit_days   INTEGER DEFAULT 0,
    total_benefit_amount REAL DEFAULT 0.0,
    injury_types         TEXT,   -- JSON array

    -- Document metadata
    num_documents        INTEGER DEFAULT 0,
    doc_date_earliest    TEXT,
    doc_date_latest      TEXT,

    -- Processing metadata
    synthesis_version    TEXT,
    processed_at         TEXT,
    llm_extraction_used  INTEGER DEFAULT 0   -- 1 if LLM was called for this row
);
"""

CREATE_INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_occupation ON people(occupation_primary);",
    "CREATE INDEX IF NOT EXISTS idx_gender ON people(gender);",
    "CREATE INDEX IF NOT EXISTS idx_race ON people(race);",
    "CREATE INDEX IF NOT EXISTS idx_division ON people(division);",
    "CREATE INDEX IF NOT EXISTS idx_hire_year ON people(hire_year);",
    "CREATE INDEX IF NOT EXISTS idx_occupation_norm ON people(occupation_normalized);",
]

# ---------------------------------------------------------------------------
# LLM client — minimal, self-contained, matches codebase patterns
# ---------------------------------------------------------------------------


def call_llm(prompt: str) -> str:
    """
    Call the configured LLM and return response text.
    Mirrors person_synthesis.py style where possible.
    Gracefully returns empty string on failure.
    """
    try:
        if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
            import openai

            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )
            return resp.choices[0].message.content or ""

        # Default/fallback provider: Ollama chat endpoint
        import requests

        payload = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 512},
        }
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")
    except Exception as exc:
        print(f"    [LLM ERROR] {exc}")
        return ""


def parse_llm_json(text: str, default: Dict[str, Any]) -> Dict[str, Any]:
    """Strip markdown fences and parse JSON. Returns default on failure."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else default
    except json.JSONDecodeError:
        return default


# ---------------------------------------------------------------------------
# Extraction — direct from synthesis JSON (no LLM)
# ---------------------------------------------------------------------------


def extract_employment_fields(synthesis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pull occupation, division, hire/separation year from employment_timeline.
    Returns dict of fields.
    """
    timeline = synthesis.get("employment_timeline", [])
    if not isinstance(timeline, list):
        timeline = []

    positions = []
    divisions = []
    departments = []
    hire_year = None
    separation_year = None
    separation_type = "unknown"

    for event in timeline:
        if not isinstance(event, dict):
            continue

        # Collect positions once per unique label for stable aggregation.
        pos = event.get("position") or event.get("job_title")
        if pos and pos not in positions:
            positions.append(pos)

        # Collect divisions.
        div = event.get("division")
        if div and div not in divisions:
            divisions.append(div)

        dept = event.get("department")
        if dept and dept not in departments:
            departments.append(dept)

        # Hire year: earliest "hire" or "application" event.
        event_type = str(event.get("event_type") or "").lower()
        date_str = str(event.get("date") or "")
        year = _parse_year(date_str)

        if year:
            if event_type in ("hire", "application", "start") and (hire_year is None or year < hire_year):
                hire_year = year
            if event_type in ("resignation", "furlough", "death", "separation", "termination"):
                if separation_year is None or year > separation_year:
                    separation_year = year
                    separation_type = event_type

    return {
        "occupation_primary": positions[0] if positions else None,
        "occupations_all": json.dumps(positions) if positions else "[]",
        "division": divisions[0] if divisions else None,
        "department": departments[0] if departments else None,
        "hire_year": hire_year,
        "separation_year": separation_year,
        "separation_type": separation_type,
    }



def extract_injury_fields(synthesis: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate injury stats from synthesis.injury_history."""
    injuries = synthesis.get("injury_history", [])
    if not isinstance(injuries, list):
        injuries = []

    total_days = 0
    total_amount = 0.0
    injury_types = []

    for injury in injuries:
        if not isinstance(injury, dict):
            continue

        itype = injury.get("injury_type")
        if itype and itype not in injury_types:
            injury_types.append(itype)

        for payment in injury.get("payments", []) or []:
            if not isinstance(payment, dict):
                continue
            days = payment.get("days") or 0
            amount = payment.get("amount") or 0.0
            try:
                total_days += int(days)
                total_amount += float(amount)
            except (TypeError, ValueError):
                pass

    return {
        "num_injuries": len(injuries),
        "total_benefit_days": total_days,
        "total_benefit_amount": round(total_amount, 2),
        "injury_types": json.dumps(injury_types) if injury_types else "[]",
    }



def extract_doc_dates(synthesis: Dict[str, Any]) -> Dict[str, Any]:
    """Pull earliest/latest document dates from document_analysis."""
    analysis = synthesis.get("document_analysis", {}) or {}
    date_range = analysis.get("date_range", {}) or {}
    return {
        "doc_date_earliest": date_range.get("earliest"),
        "doc_date_latest": date_range.get("latest"),
    }



def _parse_year(date_str: str) -> Optional[int]:
    """Extract 4-digit year from any date string format."""
    if not date_str:
        return None
    match = re.search(r"\b(18|19|20)\d{2}\b", str(date_str))
    return int(match.group()) if match else None


# ---------------------------------------------------------------------------
# Extraction — LLM-assisted (for gender, race, nationality)
# ---------------------------------------------------------------------------

LLM_EXTRACTION_PROMPT = """You are extracting demographic attributes from a historical railroad employment record synthesis.
Your job is careful, evidence-based extraction. Do NOT infer or assume.

BIOGRAPHICAL NARRATIVE:
{narrative}

PERSON NAME: {name}
BIRTH YEAR: {birth_year}

Extract the following. Return ONLY valid JSON, no explanation:

{{
  "gender": "male" | "female" | "unknown",
  "gender_confidence": "high" | "medium" | "low",
  "gender_evidence": "brief explanation of how you determined this",

  "race": "white" | "black" | "colored" | "unknown" | "<explicit label from document>",
  "race_confidence": "high" | "medium" | "low",
  "race_evidence": "direct quote or phrase from the text, or 'not stated'",

  "nationality": "<country or region of origin>" | "unknown",
  "nationality_confidence": "high" | "medium" | "low",
  "nationality_evidence": "direct quote or phrase, or 'not stated'"
}}

RULES:
- For gender: 'male' if pronouns he/him/his or exclusively male job titles of the era (fireman, brakeman, etc.) AND no contrary evidence. 'female' only if explicitly stated or female pronouns appear.
- For race: ONLY mark non-unknown if a racial label is EXPLICITLY stated in the records. Do not infer from surnames.
- For nationality: Only mark if a specific country/region of origin is explicitly mentioned.
- All three fields default to 'unknown' with confidence 'low' if evidence is absent.
- The historical documents may use terms like 'colored', 'white', 'negro' — transcribe those terms faithfully as the race value if present.
"""



def extract_demographics_via_llm(synthesis: Dict[str, Any], person_name: str) -> Dict[str, Any]:
    """
    Use LLM to extract gender, race, nationality from the biographical narrative.
    Falls back to all-unknown on any error.
    """
    fallback = {
        "gender": "unknown",
        "gender_confidence": "low",
        "race": "unknown",
        "race_confidence": "low",
        "nationality": "unknown",
        "nationality_confidence": "low",
        "llm_extraction_used": 1,
    }

    narrative = synthesis.get("biographical_narrative", "")
    identity = synthesis.get("person_identity", {}) or {}
    birth_year = identity.get("birth_year", "unknown")

    if not narrative:
        fallback["llm_extraction_used"] = 0
        return fallback

    prompt = LLM_EXTRACTION_PROMPT.format(
        narrative=str(narrative)[:3000],  # Cap to avoid token overflow for long syntheses.
        name=person_name,
        birth_year=birth_year,
    )

    raw = call_llm(prompt)
    result = parse_llm_json(raw, fallback)

    # Validate expected keys to keep DB row contract stable.
    for key in (
        "gender",
        "gender_confidence",
        "race",
        "race_confidence",
        "nationality",
        "nationality_confidence",
    ):
        if key not in result:
            result[key] = fallback[key]

    result["llm_extraction_used"] = 1
    return result


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(CREATE_TABLE_SQL)
    # Backfill newer normalization columns for pre-existing databases without dropping data.
    ensure_occupation_norm_columns(conn)
    for idx_sql in CREATE_INDEX_SQL:
        conn.execute(idx_sql)
    conn.commit()
    return conn


def ensure_occupation_norm_columns(conn: sqlite3.Connection) -> None:
    """Ensure fuzzy-normalization columns exist for databases created before Section 11."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(people)").fetchall()}
    migrations = [
        ("occupation_normalized", "TEXT"),
        ("occupation_norm_score", "REAL"),
        ("occupation_norm_method", "TEXT"),
    ]
    for column_name, column_type in migrations:
        if column_name not in existing:
            conn.execute(f"ALTER TABLE people ADD COLUMN {column_name} {column_type}")
            print(f"  [MIGRATION] Added column: {column_name}")



def already_processed(conn: sqlite3.Connection, person_folder: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM people WHERE person_folder = ?", (person_folder,)
    ).fetchone()
    return row is not None



def upsert_person(conn: sqlite3.Connection, row: Dict[str, Any], dry_run: bool = False) -> None:
    if dry_run:
        print(f"    [DRY RUN] Would write: {json.dumps({k: v for k, v in row.items() if k != 'occupations_all'}, indent=2)}")
        return

    columns = ", ".join(row.keys())
    placeholders = ", ".join("?" for _ in row)
    conflict_updates = ", ".join(f"{k}=excluded.{k}" for k in row if k != "person_folder")

    sql = f"""
        INSERT INTO people ({columns}) VALUES ({placeholders})
        ON CONFLICT(person_folder) DO UPDATE SET {conflict_updates}
    """
    conn.execute(sql, list(row.values()))
    conn.commit()


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------


def process_all(force: bool = False, limit: Optional[int] = None, dry_run: bool = False) -> None:
    print("=" * 70)
    print("DEMOGRAPHICS DATABASE BUILDER")
    print("=" * 70)
    print(f"Output:   {DB_PATH}")
    print(f"Mode:     {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"Force:    {force}")
    print(f"LLM:      {LLM_PROVIDER} / {LLM_MODEL}")
    print(f"DB Name:  {MONGO_DB_NAME}")
    print("=" * 70)

    # Connect MongoDB
    client = get_client()
    db = get_db(client)
    syntheses_col = db[SYNTHESES_COLLECTION]

    total_in_mongo = syntheses_col.count_documents({})
    print(f"\nFound {total_in_mongo} person_syntheses in MongoDB")

    # Connect SQLite
    conn = init_db(DB_PATH)

    # Fetch all synthesis records
    cursor = syntheses_col.find(
        {},
        {
            "person_folder": 1,
            "person_id": 1,
            "person_name": 1,
            "num_documents": 1,
            "synthesis": 1,
            "version": 1,
            "generated_date": 1,
        },
    )

    folders = list(cursor)
    if limit:
        folders = folders[:limit]
        print(f"Limited to {limit} persons for this run")

    stats = {"total": len(folders), "written": 0, "skipped": 0, "errors": 0}
    start = time.time()

    for i, record in enumerate(folders, 1):
        person_folder = str(record.get("person_folder", "")).strip()
        person_name = str(record.get("person_name", "Unknown"))
        synthesis = record.get("synthesis", {}) or {}

        print(f"\n[{i}/{stats['total']}] {person_folder or '<missing-folder>'} ({person_name})")

        # Skip malformed rows to preserve one-row-per-folder contract.
        if not person_folder:
            print("  ⚠️  Missing person_folder — skipping invalid record")
            stats["skipped"] += 1
            continue

        if not force and already_processed(conn, person_folder):
            print("  ⏭️  Already processed — skipping (use --force to reprocess)")
            stats["skipped"] += 1
            continue

        try:
            # --- Direct extraction (no LLM) ---
            identity = synthesis.get("person_identity", {}) or {}
            employment = extract_employment_fields(synthesis)
            injuries = extract_injury_fields(synthesis)
            dates = extract_doc_dates(synthesis)

            # --- LLM extraction for demographics ---
            print("  🤖 LLM extraction for demographics...")
            demographics = extract_demographics_via_llm(synthesis, person_name)
            print(
                f"     gender={demographics['gender']} ({demographics['gender_confidence']}), "
                f"race={demographics['race']} ({demographics['race_confidence']})"
            )

            # --- Assemble row ---
            row = {
                "person_folder": person_folder,
                "person_id": record.get("person_id"),
                "canonical_name": identity.get("canonical_name") or person_name,
                "birth_year": identity.get("birth_year"),
                "gender": demographics["gender"],
                "gender_confidence": demographics["gender_confidence"],
                "race": demographics["race"],
                "race_confidence": demographics["race_confidence"],
                "nationality": demographics["nationality"],
                "nationality_confidence": demographics["nationality_confidence"],
                **employment,
                **injuries,
                **dates,
                "num_documents": record.get("num_documents", 0),
                "synthesis_version": record.get("version", "1.0"),
                "processed_at": datetime.now().isoformat(),
                "llm_extraction_used": demographics.get("llm_extraction_used", 0),
            }

            upsert_person(conn, row, dry_run=dry_run)
            stats["written"] += 1
            print("  ✅ Written to SQLite")

        except Exception as exc:
            print(f"  ❌ ERROR: {exc}")
            stats["errors"] += 1

        # Brief rate limiting between LLM calls.
        if i < stats["total"]:
            time.sleep(0.5)

    # Summary
    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Total:   {stats['total']}")
    print(f"Written: {stats['written']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Errors:  {stats['errors']}")
    print(f"Time:    {elapsed / 60:.1f} minutes")
    print(f"DB:      {DB_PATH}")

    if not dry_run and stats["written"] > 0:
        # Quick summary query for operator confidence after a build.
        summary = conn.execute(
            """
            SELECT
                COUNT(*) as total,
                COUNT(CASE WHEN gender='male' THEN 1 END) as male,
                COUNT(CASE WHEN gender='female' THEN 1 END) as female,
                COUNT(CASE WHEN race != 'unknown' THEN 1 END) as race_known,
                COUNT(CASE WHEN occupation_primary IS NOT NULL THEN 1 END) as has_occupation
            FROM people
            """
        ).fetchone()
        print(f"\nDatabase summary: {summary['total']} total persons")
        print(f"  Gender known: {summary['male']} male, {summary['female']} female")
        print(f"  Race explicitly stated: {summary['race_known']}")
        print(f"  Occupation identified: {summary['has_occupation']}")

    conn.close()
    client.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build demographics SQLite DB from person_syntheses")
    parser.add_argument("--force", action="store_true", help="Reprocess all folders (overwrite existing rows)")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N persons (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Print extracted data without writing to SQLite")
    args = parser.parse_args()

    process_all(force=args.force, limit=args.limit, dry_run=args.dry_run)
