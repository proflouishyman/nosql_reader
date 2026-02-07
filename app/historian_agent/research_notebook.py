# app/historian_agent/research_notebook.py
# Created: 2026-02-05
# Purpose: Persistent research notebook for Tier 0 corpus exploration

"""
Research Notebook - Cumulative state for systematic corpus reading.

Internal storage uses dataclasses and lists.
External APIs return dictionaries for JSON serialization.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Entity:
    """Tracked entity (person, organization, place)."""
    name: str
    entity_type: str
    first_seen: str
    last_seen: str
    document_count: int = 1
    contexts: List[str] = field(default_factory=list)
    first_seen_block: Optional[str] = None
    last_seen_block: Optional[str] = None

    def merge(self, other: "Entity") -> None:
        self.document_count += other.document_count
        self.contexts.extend(other.contexts[:3])
        self.contexts = self.contexts[:10]

        if other.first_seen_block and not self.first_seen_block:
            self.first_seen_block = other.first_seen_block
        if other.last_seen_block:
            self.last_seen_block = other.last_seen_block


@dataclass
class Pattern:
    """Identified pattern across documents."""
    pattern_text: str
    pattern_type: str
    evidence_doc_ids: List[str] = field(default_factory=list)
    evidence_block_ids: List[str] = field(default_factory=list)
    confidence: str = "low"
    time_range: Optional[str] = None
    first_noticed: str = ""

    def add_evidence(self, doc_ids: List[str], block_ids: Optional[List[str]] = None) -> None:
        self.evidence_doc_ids.extend(doc_ids)
        self.evidence_doc_ids = list(set(self.evidence_doc_ids))

        if block_ids:
            self.evidence_block_ids.extend(block_ids)
            self.evidence_block_ids = list(set(self.evidence_block_ids))

        if len(self.evidence_doc_ids) >= 20:
            self.confidence = "high"
        elif len(self.evidence_doc_ids) >= 10:
            self.confidence = "medium"


@dataclass
class Contradiction:
    """Noticed contradiction between sources."""
    claim_a: str
    claim_b: str
    source_a: str
    source_b: str
    context: str
    noticed_in_batch: str
    contradiction_type: str = "unknown"
    confidence: str = "low"


@dataclass
class ResearchQuestion:
    """Emerging research question."""
    question: str
    why_interesting: str
    evidence_needed: str
    related_entities: List[str] = field(default_factory=list)
    time_window: Optional[str] = None
    noticed_in_batch: str = ""


# ============================================================================
# Research Notebook
# ============================================================================

class ResearchNotebook:
    """Persistent state that accumulates across batches."""

    def __init__(self) -> None:
        self.entities: Dict[str, Entity] = {}
        self.patterns: Dict[str, Pattern] = {}
        self.contradictions: List[Contradiction] = []
        self.questions: List[ResearchQuestion] = []
        self.temporal_map: Dict[str, List[str]] = defaultdict(list)
        self.corpus_map = {
            "total_documents_read": 0,
            "by_year": defaultdict(int),
            "by_collection": defaultdict(int),
            "by_document_type": defaultdict(int),
            "by_person": defaultdict(int),
            "batches_processed": 0,
            "time_coverage": {"start": None, "end": None},
            "gaps_identified": [],
        }
        self.created_at = datetime.now().isoformat()
        self.last_updated = self.created_at
        self.batches_log: List[Dict[str, Any]] = []

    def integrate_batch_findings(self, findings: Dict[str, Any], batch_label: str) -> None:
        for entity_dict in findings.get("entities", []):
            self._add_entity(entity_dict)

        for pattern_dict in findings.get("patterns", []):
            self._add_pattern(pattern_dict, batch_label)

        for contra_dict in findings.get("contradictions", []):
            contra_dict["noticed_in_batch"] = batch_label
            contra_type, confidence = self._classify_contradiction(contra_dict)
            contra_dict["contradiction_type"] = contra_type
            contra_dict["confidence"] = confidence
            self.contradictions.append(Contradiction(**contra_dict))

        for question_dict in findings.get("questions", []):
            question_dict["noticed_in_batch"] = batch_label
            self.questions.append(ResearchQuestion(**question_dict))

        for year, events in findings.get("temporal_events", {}).items():
            self.temporal_map[year].extend(events)

        self._update_corpus_map(findings.get("stats", {}))

        self.batches_log.append({
            "batch_label": batch_label,
            "processed_at": datetime.now().isoformat(),
            "entities_added": len(findings.get("entities", [])),
            "patterns_added": len(findings.get("patterns", [])),
            "questions_added": len(findings.get("questions", [])),
        })

        self.corpus_map["batches_processed"] += 1
        self.last_updated = datetime.now().isoformat()

    def _add_entity(self, entity_dict: Dict[str, Any]) -> None:
        name = entity_dict.get("name")
        if not name:
            return

        normalized_name = str(name).lower().strip()
        entity_type = entity_dict.get("type") or entity_dict.get("entity_type") or "unknown"
        first_seen = entity_dict.get("first_seen") or "unknown"
        last_seen = entity_dict.get("last_seen") or first_seen
        context = entity_dict.get("context")
        contexts = entity_dict.get("contexts") or ([] if context is None else [context])
        if not isinstance(contexts, list):
            contexts = [str(contexts)]

        new_entity = Entity(
            name=str(name),
            entity_type=str(entity_type),
            first_seen=str(first_seen),
            last_seen=str(last_seen),
            contexts=[str(c) for c in contexts if c],
            first_seen_block=entity_dict.get("first_seen_block"),
            last_seen_block=entity_dict.get("last_seen_block"),
        )

        if normalized_name in self.entities:
            self.entities[normalized_name].merge(new_entity)
        else:
            self.entities[normalized_name] = new_entity

    def _add_pattern(self, pattern_dict: Dict[str, Any], batch_label: str) -> None:
        pattern_text = pattern_dict.get("pattern") or pattern_dict.get("pattern_text")
        if not pattern_text:
            return

        raw_evidence = pattern_dict.get("evidence") or []
        evidence_blocks = pattern_dict.get("evidence_blocks") or []

        doc_ids: List[str] = []
        block_ids: List[str] = []

        for item in raw_evidence:
            if isinstance(item, str) and "::" in item:
                block_ids.append(item)
                doc_ids.append(item.split("::")[0])
            elif item:
                doc_ids.append(str(item))

        for item in evidence_blocks:
            if isinstance(item, str):
                block_ids.append(item)
                doc_ids.append(item.split("::")[0] if "::" in item else item)

        doc_ids = list(set(doc_ids))
        block_ids = list(set(block_ids))

        if pattern_text in self.patterns:
            existing = self.patterns[pattern_text]
            existing.add_evidence(doc_ids, block_ids)
        else:
            self.patterns[pattern_text] = Pattern(
                pattern_text=str(pattern_text),
                pattern_type=str(pattern_dict.get("type") or pattern_dict.get("pattern_type") or "unknown"),
                evidence_doc_ids=doc_ids,
                evidence_block_ids=block_ids,
                confidence=str(pattern_dict.get("confidence") or "low"),
                time_range=pattern_dict.get("time_range"),
                first_noticed=batch_label,
            )

    def _classify_contradiction(self, contra_dict: Dict[str, Any]) -> tuple[str, str]:
        claim_a = str(contra_dict.get("claim_a", "")).lower()
        claim_b = str(contra_dict.get("claim_b", "")).lower()

        # Heuristic 1: name variant / OCR error (high string similarity)
        def _normalize(text: str) -> str:
            return "".join(ch for ch in text if ch.isalnum() or ch.isspace()).strip()

        norm_a = _normalize(claim_a)
        norm_b = _normalize(claim_b)
        if norm_a and norm_b:
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, norm_a, norm_b).ratio()
            if similarity >= 0.85:
                return "name_variant_or_ocr", "medium"

        # Heuristic 2: certificate or ID number conflict
        numbers_a = set(re.findall(r"\\b\\d{3,}\\b", claim_a))
        numbers_b = set(re.findall(r"\\b\\d{3,}\\b", claim_b))
        if numbers_a and numbers_b and numbers_a != numbers_b:
            if "certificate" in claim_a or "certificate" in claim_b:
                return "certificate_number_conflict", "medium"

        # Heuristic 3: date conflict
        years_a = set(re.findall(r"(18\\d{2}|19\\d{2}|20\\d{2})", claim_a))
        years_b = set(re.findall(r"(18\\d{2}|19\\d{2}|20\\d{2})", claim_b))
        if years_a and years_b and years_a != years_b:
            return "date_conflict", "low"

        # Default
        return "true_conflict", "low"

    def _update_corpus_map(self, stats: Dict[str, Any]) -> None:
        self.corpus_map["total_documents_read"] += stats.get("docs_in_batch", 0)

        for year, count in stats.get("by_year", {}).items():
            self.corpus_map["by_year"][year] += count

        for collection, count in stats.get("by_collection", {}).items():
            self.corpus_map["by_collection"][collection] += count

        for doc_type, count in stats.get("by_document_type", {}).items():
            self.corpus_map["by_document_type"][doc_type] += count

        for person, count in stats.get("by_person", {}).items():
            self.corpus_map["by_person"][person] += count

        if stats.get("earliest_year"):
            if self.corpus_map["time_coverage"]["start"] is None:
                self.corpus_map["time_coverage"]["start"] = stats["earliest_year"]
            else:
                self.corpus_map["time_coverage"]["start"] = min(
                    self.corpus_map["time_coverage"]["start"],
                    stats["earliest_year"],
                )

        if stats.get("latest_year"):
            if self.corpus_map["time_coverage"]["end"] is None:
                self.corpus_map["time_coverage"]["end"] = stats["latest_year"]
            else:
                self.corpus_map["time_coverage"]["end"] = max(
                    self.corpus_map["time_coverage"]["end"],
                    stats["latest_year"],
                )

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_entities": len(self.entities),
            "top_entities": sorted(
                [(e.name, e.document_count) for e in self.entities.values()],
                key=lambda x: x[1],
                reverse=True,
            )[:20],
            "total_patterns": len(self.patterns),
            "high_confidence_patterns": [
                p.pattern_text for p in self.patterns.values() if p.confidence == "high"
            ],
            "total_questions": len(self.questions),
            "total_contradictions": len(self.contradictions),
            "documents_read": self.corpus_map["total_documents_read"],
            "time_coverage": self.corpus_map["time_coverage"],
            "batches_processed": self.corpus_map["batches_processed"],
        }

    def format_for_llm_context(self) -> str:
        summary = self.get_summary()

        text = f"PRIOR KNOWLEDGE (from {summary['documents_read']} documents read):\n\n"
        text += f"Entities Found ({summary['total_entities']} total):\n"
        for name, count in summary["top_entities"][:10]:
            text += f"  - {name} ({count} documents)\n"

        text += f"\nHigh-Confidence Patterns ({len(summary['high_confidence_patterns'])}):\n"
        for pattern in summary["high_confidence_patterns"][:5]:
            text += f"  - {pattern}\n"

        text += f"\nResearch Questions ({summary['total_questions']}):\n"
        for q in self.questions[-5:]:
            text += f"  - {q.question}\n"

        text += f"\nTime Coverage: {summary['time_coverage']['start']} - {summary['time_coverage']['end']}\n"
        text += f"Batches Processed: {summary['batches_processed']}\n"
        return text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": {k: asdict(v) for k, v in self.entities.items()},
            "patterns": {k: asdict(v) for k, v in self.patterns.items()},
            "contradictions": [asdict(c) for c in self.contradictions],
            "questions": [asdict(q) for q in self.questions],
            "temporal_map": dict(self.temporal_map),
            "corpus_map": dict(self.corpus_map),
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "batches_log": self.batches_log,
        }

    def save(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "ResearchNotebook":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        notebook = cls()
        notebook.entities = {k: Entity(**v) for k, v in data.get("entities", {}).items()}
        notebook.patterns = {k: Pattern(**v) for k, v in data.get("patterns", {}).items()}
        notebook.contradictions = [Contradiction(**c) for c in data.get("contradictions", [])]
        notebook.questions = [ResearchQuestion(**q) for q in data.get("questions", [])]
        notebook.temporal_map = defaultdict(list, data.get("temporal_map", {}))
        notebook.corpus_map = data.get("corpus_map", notebook.corpus_map)
        notebook.created_at = data.get("created_at", notebook.created_at)
        notebook.last_updated = data.get("last_updated", notebook.last_updated)
        notebook.batches_log = data.get("batches_log", [])

        return notebook
