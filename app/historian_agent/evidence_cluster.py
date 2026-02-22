# app/historian_agent/evidence_cluster.py
# Created: 2026-02-15
# Purpose: Build richer evidence clusters for Tier 0 leaf-question synthesis.

"""
Evidence Cluster Builder

Builds a richer evidence set per question by combining:
1. Seed document IDs already attached to the question
2. Retrieval expansion from the hybrid retriever when available
3. Deterministic keyword fallback and deduplication
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from config import APP_CONFIG
from rag_base import DocumentStore, debug_print


class EvidenceClusterBuilder:
    """Build rich evidence clusters for essay/leaf question generation."""

    def __init__(self, doc_store: Optional[DocumentStore] = None) -> None:
        self.doc_store = doc_store or DocumentStore()
        self._hybrid_retriever = None
        self._init_hybrid_retriever()

    def build_cluster(
        self,
        question_text: str,
        seed_doc_ids: List[str],
        max_seed: int = 5,
        retrieval_top_k: int = 10,
        max_total: int = 12,
        snippet_max_chars: int = 2000,
    ) -> List[Dict[str, Any]]:
        """Build evidence items with text snippets and provenance metadata."""
        evidence: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()

        for doc_id in (seed_doc_ids or [])[:max_seed]:
            normalized_id = self._normalize_doc_id(doc_id)
            if not normalized_id or normalized_id in seen_ids:
                continue
            text = self._fetch_doc_text(normalized_id, snippet_max_chars)
            if not text:
                continue
            evidence.append(
                {
                    "doc_id": normalized_id,
                    "text": text,
                    "source": "seed",
                    "score": 1.0,
                }
            )
            seen_ids.add(normalized_id)

        retrieval_hits = self._retrieve_doc_candidates(question_text, retrieval_top_k)
        for doc_id, score in retrieval_hits:
            normalized_id = self._normalize_doc_id(doc_id)
            if not normalized_id or normalized_id in seen_ids:
                continue
            text = self._fetch_doc_text(normalized_id, snippet_max_chars)
            if not text:
                continue
            evidence.append(
                {
                    "doc_id": normalized_id,
                    "text": text,
                    "source": "retrieval",
                    "score": float(score),
                }
            )
            seen_ids.add(normalized_id)
            if len(evidence) >= max_total:
                break

        evidence.sort(key=lambda item: (0 if item.get("source") == "seed" else 1, -float(item.get("score", 0.0))))
        return evidence[:max_total]

    def build_clusters_for_questions(
        self,
        questions: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Build question_text -> evidence cluster mapping for multiple question payloads."""
        clusters: Dict[str, List[Dict[str, Any]]] = {}
        for question in questions or []:
            if not isinstance(question, dict):
                continue
            question_text = str(question.get("question") or question.get("question_text") or "").strip()
            if not question_text:
                continue
            seed_ids = (
                question.get("evidence_doc_ids")
                or question.get("evidence_sample")
                or question.get("supporting_documents")
                or []
            )
            if isinstance(seed_ids, str):
                seed_ids = [part.strip() for part in seed_ids.split(",") if part.strip()]
            cluster = self.build_cluster(question_text, list(seed_ids), **kwargs)
            clusters[question_text] = cluster
        return clusters

    def _retrieve_doc_candidates(self, question_text: str, top_k: int) -> List[Tuple[str, float]]:
        if not question_text.strip() or top_k <= 0:
            return []

        candidates = self._retrieve_hybrid(question_text, top_k)
        if candidates:
            return candidates

        return self._retrieve_keyword_fallback(question_text, top_k)

    def _retrieve_hybrid(self, question_text: str, top_k: int) -> List[Tuple[str, float]]:
        if self._hybrid_retriever is None:
            return []

        try:
            docs = self._hybrid_retriever.get_relevant_documents(question_text)
        except Exception as exc:
            debug_print(f"Evidence cluster hybrid retrieval failed: {exc}")
            return []

        scored: Dict[str, float] = {}
        for rank, doc in enumerate(docs, 1):
            metadata = getattr(doc, "metadata", {}) or {}
            doc_id = (
                metadata.get("document_id")
                or metadata.get("parent_doc_id")
                or metadata.get("doc_id")
                or metadata.get("_id")
            )
            doc_id = self._normalize_doc_id(doc_id)
            if not doc_id:
                continue
            score = float(metadata.get("score") or metadata.get("rrf_score") or (1.0 / rank))
            existing = scored.get(doc_id)
            if existing is None or score > existing:
                scored[doc_id] = score

        ordered = sorted(scored.items(), key=lambda item: item[1], reverse=True)
        return ordered[:top_k]

    def _init_hybrid_retriever(self) -> None:
        # Added package-local hybrid retriever init to avoid fragile import paths in legacy handler wrappers.
        try:
            from historian_agent.embeddings import EmbeddingService
            from historian_agent.vector_store import get_vector_store
            from historian_agent.retrievers import HybridRetriever, VectorRetriever, KeywordRetriever

            retrieval_pool = int(getattr(APP_CONFIG.retriever, "retrieval_pool_size", 40))
            top_k = max(10, retrieval_pool)

            embedding_service = EmbeddingService(
                provider=APP_CONFIG.embedding.provider,
                model=APP_CONFIG.embedding.model,
            )
            vector_store = get_vector_store(store_type="chroma")
            vector_retriever = VectorRetriever(
                vector_store=vector_store,
                embedding_service=embedding_service,
                mongo_collection=self.doc_store.chunks_coll,
                top_k=top_k,
            )

            class _ConfigShim:
                context_fields = ["text", "ocr_text"]

            keyword_retriever = KeywordRetriever(
                mongo_collection=self.doc_store.chunks_coll,
                config=_ConfigShim(),
                top_k=top_k,
            )
            self._hybrid_retriever = HybridRetriever(
                vector_retriever=vector_retriever,
                keyword_retriever=keyword_retriever,
                top_k=top_k,
            )
        except Exception as exc:
            debug_print(f"Evidence cluster hybrid retriever unavailable, using fallback only: {exc}")
            self._hybrid_retriever = None

    def _retrieve_keyword_fallback(self, question_text: str, top_k: int) -> List[Tuple[str, float]]:
        tokens = [
            token
            for token in re.findall(r"[a-z0-9]{4,}", question_text.lower())
            if token not in {"what", "when", "where", "which", "were", "from", "with", "that", "this", "about"}
        ][:6]
        if not tokens:
            return []

        or_clauses: List[Dict[str, Any]] = []
        for token in tokens:
            or_clauses.append({"text": {"$regex": token, "$options": "i"}})
            or_clauses.append({"ocr_text": {"$regex": token, "$options": "i"}})

        query = {"$or": or_clauses}
        projection = {"document_id": 1, "parent_doc_id": 1, "text": 1, "ocr_text": 1}
        cursor = self.doc_store.chunks_coll.find(query, projection).limit(max(20, top_k * 8))

        scores: Dict[str, float] = {}
        for row in cursor:
            doc_id = row.get("document_id") or row.get("parent_doc_id")
            doc_id = self._normalize_doc_id(doc_id)
            if not doc_id:
                continue
            text = (row.get("text") or row.get("ocr_text") or "").lower()
            if not text:
                continue
            score = float(sum(1 for token in tokens if token in text))
            if score <= 0:
                continue
            if doc_id not in scores or score > scores[doc_id]:
                scores[doc_id] = score

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ordered[:top_k]

    def _fetch_doc_text(self, doc_id: str, max_chars: int) -> Optional[str]:
        try:
            query: Dict[str, Any]
            try:
                from bson import ObjectId

                query = {"_id": ObjectId(doc_id)}
            except Exception:
                query = {"_id": doc_id}

            doc = self.doc_store.documents_coll.find_one(
                query,
                {"ocr_text": 1, "content": 1, "text": 1, "summary": 1},
            )
            if not doc:
                return None
            text = doc.get("ocr_text") or doc.get("content") or doc.get("text") or doc.get("summary") or ""
            if not text:
                return None
            return str(text)[:max_chars]
        except Exception as exc:
            debug_print(f"Failed to fetch evidence text for {doc_id}: {exc}")
            return None

    def _normalize_doc_id(self, value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if not text:
            return ""
        if "::" in text:
            text = text.split("::", 1)[0]
        return text
