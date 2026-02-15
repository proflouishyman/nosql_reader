#!/usr/bin/env python3
"""
One-time metadata enrichment for Tier 0 stratification.

Extracts `year` and `document_type` from existing OCR text and section fields.
No LLM calls; this is regex-only and safe to run repeatedly.

Usage:
    docker compose exec app python scripts/enrich_metadata.py --dry-run
    docker compose exec app python scripts/enrich_metadata.py
"""

import os
import re
import sys
from collections import Counter
from typing import Any, Dict, Optional

from pymongo import MongoClient, UpdateOne


MONGO_URI = os.environ.get("APP_MONGO_URI", "mongodb://admin:secret@mongodb:27017/admin")
DB_NAME = os.environ.get("DB_NAME", "railroad_documents")


YEAR_PATTERNS_LABELED = [
    r"Date\s*[:;\-]\s*(?:[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+)?(18\d{2}|19\d{2})",
    r"Dated\s*[:;\-]?\s*(?:[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+)?(18\d{2}|19\d{2})",
    r"Date of (?:Injury|Accident|Examination|Employment|Birth|Death|Application|Disablement)\s*[:;\-]?\s*(?:\w+\.?\s+\d{1,2},?\s+)?(18\d{2}|19\d{2})",
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+(18\d{2}|19\d{2})",
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+(18\d{2}|19\d{2})",
]

YEAR_PATTERN_HEADER = r"\b(18[5-9]\d|19[0-5]\d)\b"


# Added fixed, explainable form matching to boost genre stratification coverage.
FORM_TYPE_PATTERNS = {
    "injury_report": [
        r"REPORT OF PERSONAL INJURY",
        r"SURGEON'S FIRST REPORT OF ACCIDENT",
        r"SURGEON'S REPORT",
        r"REPORT OF ACCIDENT",
        r"How\s+(?:was\s+)?(?:he|she|the employee)\s+injured",
        r"Nature of Injury",
        r"cause of injury",
    ],
    "disability_certificate": [
        r"CERTIFICATE OF (?:DIS)?ABILITY",
        r"CERTIFICATE OF DISABLEMENT",
        r"MEDICAL EXAMINER.S REPORT OF DISABLEMENT",
        r"Notice of Disablement",
        r"totally disabled",
    ],
    "membership_application": [
        r"Application for (?:Full )?[Mm]embership",
        r"APPLICATION FOR MEMBERSHIP IN THE RELIEF",
        r"QUESTIONS TO BE ASKED THE APPLICANT",
        r"Application for membership in the Relief Feature",
    ],
    "death_benefit_application": [
        r"Application for (?:Natural )?Death Benefit",
        r"PROOF OF DEATH",
        r"Certificate of Death",
        r"Natural Death Benefit",
    ],
    "return_to_duty": [
        r"CERTIFICATE OF ABILITY",
        r"able to (?:return|resume) (?:to )?(?:duty|work)",
        r"Return to Duty",
        r"resumed duty",
    ],
    "correspondence": [
        r"Dear (?:Sir|Mr|Mrs|Madam)",
        r"Yours (?:truly|respectfully|very truly)",
        r"(?:Superintendent|Dear)\s+(?:Sir|Mr)",
        r"I (?:beg|have) to (?:advise|inform)",
    ],
    "wage_record": [
        r"WAGE\b",
        r"Rate of [Pp]ay",
        r"Monthly (?:Wages|Pay|Salary)",
        r"SCHEDULE OF WAGES",
    ],
    "employment_record": [
        r"Employment and Record Bureau",
        r"Date (?:of )?(?:first )?[Ee]mployment",
        r"SERVICE RECORD",
        r"RECORD OF EMPLOYE",
    ],
    "medical_exam": [
        r"MEDICAL EXAMINER.S (?:REPORT|CERTIFICATE)",
        r"Physical Examination",
        r"Examination of Applicant",
        r"examined .{0,30} and find",
    ],
}


def extract_year(doc: Dict[str, Any]) -> Optional[int]:
    """Extract the most reliable year available from a document."""
    text = doc.get("ocr_text") or ""
    if not text:
        return None

    for pattern in YEAR_PATTERNS_LABELED:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                year = int(match.group(1))
                if 1850 <= year <= 1960:
                    return year
            except (ValueError, IndexError):
                continue

    for section in doc.get("sections") or []:
        if not isinstance(section, dict):
            continue
        for field_obj in section.get("fields") or []:
            if not isinstance(field_obj, dict):
                continue
            field_name = (field_obj.get("field_name") or "").lower()
            if "date" not in field_name:
                continue
            value = str(field_obj.get("value") or "")
            match = re.search(r"(18\d{2}|19\d{2})", value)
            if not match:
                continue
            try:
                year = int(match.group(1))
                if 1850 <= year <= 1960:
                    return year
            except ValueError:
                continue

    header = text[:500]
    match = re.search(YEAR_PATTERN_HEADER, header)
    if match:
        try:
            year = int(match.group(1))
            if 1860 <= year <= 1955:
                return year
        except ValueError:
            pass

    # Added conservative filename/path fallback to recover dates encoded in archive naming.
    path_text = " ".join(
        [
            str(doc.get("relative_path") or ""),
            str(doc.get("file_path") or ""),
            str(doc.get("filename") or ""),
        ]
    )
    for match in re.finditer(r"(18\d{2}|19\d{2})", path_text):
        try:
            year = int(match.group(1))
            if 1850 <= year <= 1955:
                return year
        except ValueError:
            continue

    return None


def classify_document_type(doc: Dict[str, Any]) -> Optional[str]:
    """Classify document form type from OCR text and summary fallback."""
    text = doc.get("ocr_text") or ""
    if not text:
        return None

    scores: Dict[str, int] = {}
    for doc_type, patterns in FORM_TYPE_PATTERNS.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 1
        if score > 0:
            scores[doc_type] = score

    if not scores:
        summary = doc.get("summary") or ""
        for doc_type, patterns in FORM_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, summary, re.IGNORECASE):
                    return doc_type
        return None

    return max(scores, key=scores.get)


def run_enrichment(dry_run: bool = False) -> None:
    """Run metadata enrichment across all documents."""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    docs_coll = db["documents"]

    bulk_ops = []
    stats = {
        "total": 0,
        "year_added": 0,
        "type_added": 0,
        "year_already": 0,
        "type_already": 0,
        "year_types": Counter(),
        "doc_types": Counter(),
    }

    projection = {
        "_id": 1,
        "ocr_text": 1,
        "sections": 1,
        "summary": 1,
        "file_path": 1,
        "relative_path": 1,
        "filename": 1,
        "year": 1,
        "document_type": 1,
    }

    cursor = docs_coll.find({}, projection)

    for doc in cursor:
        stats["total"] += 1
        updates: Dict[str, Any] = {}

        existing_year = doc.get("year")
        if existing_year:
            stats["year_already"] += 1
        else:
            year = extract_year(doc)
            if year:
                updates["year"] = year
                stats["year_added"] += 1
                stats["year_types"][str(year)] += 1

        existing_type = doc.get("document_type")
        if existing_type:
            stats["type_already"] += 1
        else:
            doc_type = classify_document_type(doc)
            if doc_type:
                updates["document_type"] = doc_type
                stats["type_added"] += 1
                stats["doc_types"][doc_type] += 1

        if updates and not dry_run:
            bulk_ops.append(UpdateOne({"_id": doc["_id"]}, {"$set": updates}))

        if len(bulk_ops) >= 500:
            docs_coll.bulk_write(bulk_ops)
            bulk_ops = []

        if stats["total"] % 1000 == 0:
            print(f"  Processed {stats['total']}...", flush=True)

    if bulk_ops:
        docs_coll.bulk_write(bulk_ops)

    client.close()

    mode = "DRY RUN" if dry_run else "ENRICHMENT COMPLETE"
    print(f"\n{'=' * 50}")
    print(f"  {mode}")
    print(f"{'=' * 50}")
    print(f"  Total documents scanned:  {stats['total']}")
    print(f"  Years already populated:  {stats['year_already']}")
    print(f"  Years newly extracted:    {stats['year_added']}")
    print(f"  Types already populated:  {stats['type_already']}")
    print(f"  Types newly classified:   {stats['type_added']}")

    if stats["doc_types"]:
        print("\n  Document type distribution (new):")
        for dtype, count in stats["doc_types"].most_common():
            print(f"    {dtype}: {count}")

    total_year = stats["year_already"] + stats["year_added"]
    total_type = stats["type_already"] + stats["type_added"]
    print("\n  Final coverage:")
    print(f"    year:          {total_year}/{stats['total']} ({100 * total_year // max(1, stats['total'])}%)")
    print(f"    document_type: {total_type}/{stats['total']} ({100 * total_type // max(1, stats['total'])}%)")


if __name__ == "__main__":
    is_dry_run = "--dry-run" in sys.argv
    if is_dry_run:
        print("Running in DRY RUN mode (no writes)...")
    run_enrichment(dry_run=is_dry_run)
