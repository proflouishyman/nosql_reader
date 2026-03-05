# app/historian_agent/demographic_orientation.py
# Purpose: Optional demographic orientation summary for corpus exploration prompts.

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _table_columns(conn: sqlite3.Connection) -> set[str]:
    """Return column names from people table for backward-compatible querying."""
    rows = conn.execute("PRAGMA table_info(people)").fetchall()
    return {str(row[1]) for row in rows}


def _top_counts(
    conn: sqlite3.Connection,
    label_expr: str,
    limit: int,
) -> List[Tuple[str, int]]:
    """Return top non-empty labels for a fixed SQL label expression."""
    sql = f"""
        SELECT {label_expr} AS label, COUNT(*) AS n
        FROM people
        GROUP BY label
        HAVING label IS NOT NULL AND TRIM(CAST(label AS TEXT)) != ''
        ORDER BY n DESC
        LIMIT ?
    """
    rows = conn.execute(sql, (max(1, int(limit)),)).fetchall()
    out: List[Tuple[str, int]] = []
    for row in rows:
        label = str(row[0]).strip()
        if not label:
            continue
        out.append((label, int(row[1])))
    return out


def _format_top(items: List[Tuple[str, int]]) -> str:
    if not items:
        return "(none)"
    return ", ".join(f"{label} ({count})" for label, count in items)


def build_demographic_orientation(
    db_path: str,
    top_n: int = 5,
) -> Tuple[str, Dict[str, Any]]:
    """
    Build a compact orientation summary from demographics.db.

    Returns:
      (summary_text, metadata)
    """
    path = Path(db_path)
    meta: Dict[str, Any] = {
        "enabled": False,
        "source": str(path),
        "reason": "",
    }
    if not path.exists():
        meta["reason"] = "db_missing"
        return "", meta

    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        columns = _table_columns(conn)

        total = int(conn.execute("SELECT COUNT(*) AS n FROM people").fetchone()["n"])
        gender = _top_counts(conn, "gender", top_n) if "gender" in columns else []
        race = _top_counts(conn, "race", top_n) if "race" in columns else []
        ethnicity = _top_counts(conn, "ethnicity", top_n) if "ethnicity" in columns else []
        national_origin = _top_counts(conn, "national_origin", top_n) if "national_origin" in columns else []
        occupation = _top_counts(
            conn,
            "COALESCE(occupation_normalized, occupation_primary)",
            top_n,
        ) if ("occupation_primary" in columns or "occupation_normalized" in columns) else []
        division = _top_counts(conn, "division", top_n) if "division" in columns else []

        hire_span = ""
        if "hire_year" in columns:
            row = conn.execute(
                """
                SELECT MIN(hire_year) AS min_year, MAX(hire_year) AS max_year
                FROM people
                WHERE hire_year IS NOT NULL
                """
            ).fetchone()
            min_year = row["min_year"] if row else None
            max_year = row["max_year"] if row else None
            if min_year is not None and max_year is not None:
                hire_span = f"{min_year}-{max_year}"

        lines = [
            f"- People records: {total}",
            f"- Top occupations: {_format_top(occupation)}",
            f"- Top divisions: {_format_top(division)}",
            f"- Race distribution (top): {_format_top(race)}",
            f"- Ethnicity distribution (top): {_format_top(ethnicity)}",
            f"- National origin (top): {_format_top(national_origin)}",
            f"- Gender distribution (top): {_format_top(gender)}",
        ]
        if hire_span:
            lines.append(f"- Hire-year span in people table: {hire_span}")

        meta.update({
            "enabled": True,
            "reason": "ok",
            "people_total": total,
            "top_n": int(top_n),
            "has_ethnicity": "ethnicity" in columns,
            "has_national_origin": "national_origin" in columns,
            "has_hire_year": "hire_year" in columns,
        })
        return "\n".join(lines), meta
    except Exception as exc:
        meta["reason"] = f"error:{exc.__class__.__name__}"
        return "", meta
    finally:
        if conn is not None:
            conn.close()
