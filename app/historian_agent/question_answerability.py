# app/historian_agent/question_answerability.py
# Created: 2026-02-06
# Purpose: Lightweight answerability precheck for research questions

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any

from config import APP_CONFIG
from rag_base import DocumentStore, debug_print


_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "were", "was", "are",
    "how", "what", "when", "where", "why", "did", "does", "who", "into", "over",
    "between", "within", "their", "which", "these", "those", "have", "has",
    "about", "into", "over", "than", "then", "such", "also", "would", "could",
    "should", "during", "across", "under", "after", "before", "based",
}


@dataclass
class AnswerabilityResult:
    """Result of answerability precheck."""
    doc_count: int
    status: str
    sample_doc_ids: List[str]


class AnswerabilityChecker:
    """Check whether a question is likely answerable using lightweight retrieval."""

    def __init__(self) -> None:
        self.doc_store = DocumentStore()
        self.fields = ["content", "ocr_text", "text", "structured_data"]

    def check(self, question_text: str) -> AnswerabilityResult:
        tokens = self._extract_tokens(question_text)
        if not tokens:
            return AnswerabilityResult(doc_count=0, status="too_few", sample_doc_ids=[])

        filters = []
        for token in tokens:
            regex = {"$regex": re.escape(token), "$options": "i"}
            for field in self.fields:
                filters.append({field: regex})

        query = {"$or": filters} if filters else {}
        cursor = self.doc_store.chunks_coll.find(query).limit(APP_CONFIG.tier0.answerability_top_k)

        doc_ids = []
        for record in cursor:
            doc_id = record.get("parent_doc_id") or record.get("_id")
            if doc_id is not None:
                doc_ids.append(str(doc_id))

        unique_doc_ids = list(dict.fromkeys(doc_ids))
        doc_count = len(unique_doc_ids)

        if doc_count < APP_CONFIG.tier0.answerability_min_docs:
            status = "too_few"
        elif doc_count > APP_CONFIG.tier0.answerability_max_docs:
            status = "too_many"
        else:
            status = "ok"

        return AnswerabilityResult(
            doc_count=doc_count,
            status=status,
            sample_doc_ids=unique_doc_ids[:5],
        )

    def _extract_tokens(self, text: str, max_terms: int = 6) -> List[str]:
        words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text.lower())
        candidates = [w for w in words if w not in _STOPWORDS]
        candidates = sorted(candidates, key=len, reverse=True)
        return candidates[:max_terms]
