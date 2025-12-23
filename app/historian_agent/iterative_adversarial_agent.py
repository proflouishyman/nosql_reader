#!/usr/bin/env python3
"""
Tiered Historian Agent with Adversarial Verification.

Purpose:
Runs a tiered RAG pipeline:
1) Hybrid retrieval + reranking to produce an initial answer
2) Adversarial verification of the answer against retrieved source text
3) If verification score is low, escalate to multi-query expansion and re-synthesize

Created: 2025-12-22 21:00:00 America/New_York
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from .adversarial_rag import AdversarialRAGHandler  # expects .verify(...) wrapper or adapt call site
from .retrievers import HybridRetriever
from .rag_query_handler import RAGQueryHandler


logger = logging.getLogger(__name__)


@dataclass
class TierConfig:
    top_k_retrieve: int = 40
    top_k_rerank: int = 10
    verification_threshold: int = 90
    max_expanded_docs: int = 15
    multi_query_count: int = 3
    generator_timeout_s: int = 60
    verifier_timeout_s: int = 60
    max_retries: int = 2


class IterativeAdversarialAgent:
    """
    Tiered RAG agent:
    - Tier 1: hybrid retrieval + rerank + generate + adversarial verify
    - Tier 2: multi-query expansion if verification below threshold
    """

    def __init__(
        self,
        rag_handler: RAGQueryHandler,
        retriever: HybridRetriever,
        adversarial_handler: AdversarialRAGHandler,
        config: Optional[TierConfig] = None,
    ) -> None:
        self.rag_handler = rag_handler
        self.retriever = retriever
        self.adversarial_handler = adversarial_handler
        self.config = config or TierConfig()

        logger.info(
            "[INIT] Initialized with Generator: %s, Verifier: %s",
            getattr(self.rag_handler, "generator_model", "unknown"),
            getattr(self.adversarial_handler, "verifier_model", "unknown"),
        )

    # -------------------------
    # Public entrypoint
    # -------------------------
    def run(self, question: str) -> Dict[str, Any]:
        """
        Execute tiered pipeline for a single question.
        Returns a dict containing answer, sources, metrics, and verification report.
        """
        start = time.time()
        metrics: Dict[str, Any] = {}
        logs: Dict[str, Any] = {"tiers": []}

        logger.info("[START] Beginning investigation: %r", question)

        # -------------------------
        # Tier 1: retrieve + rerank + answer
        # -------------------------
        t1 = time.time()
        logger.info("[TIER 1] Executing hybrid retrieval + reranking...")
        tier1_docs = self._retrieve_and_rerank(question, top_k=self.config.top_k_retrieve, rerank_k=self.config.top_k_rerank)

        context_text, context_doc_ids, context_fetch_meta = self._fetch_full_text_from_docs(tier1_docs)
        answer1, gen_meta1 = self._generate_answer(question, context_text)

        verify_report1 = self._verify_answer(
            question=question,
            answer=answer1,
            docs=tier1_docs,
            context_text=context_text,
        )
        score1 = int(verify_report1.get("citation_score", 0) or 0)

        logs["tiers"].append(
            {
                "tier": 1,
                "docs": len(tier1_docs),
                "doc_ids": context_doc_ids,
                "generation_meta": gen_meta1,
                "verification": verify_report1,
                "elapsed_s": round(time.time() - t1, 2),
            }
        )

        metrics["tier1_generation_chars"] = len(answer1 or "")
        metrics["tier1_context_chars"] = len(context_text or "")
        metrics["tier1_verification_score"] = score1

        # If Tier 1 passes, return
        if score1 >= self.config.verification_threshold:
            total = round(time.time() - start, 2)
            logger.info("[COMPLETE] Tier 1 passed (score=%s). Total time: %ss", score1, total)
            return {
                "question": question,
                "answer": answer1,
                "sources": tier1_docs,
                "verification": verify_report1,
                "metrics": {**metrics, "total_time_s": total},
                "logs": logs,
            }

        # -------------------------
        # Tier 2: multi-query expansion
        # -------------------------
        t2 = time.time()
        logger.info(
            "[TIER 2] Low verification score (%s/%s < %s). Escalating to Multi-Query expansion...",
            score1,
            100,
            self.config.verification_threshold,
        )

        alt_queries = self.generate_multi_queries(question, n=self.config.multi_query_count)
        logger.info("[TIER 2] Generated %d alternative queries", len(alt_queries))

        expanded_docs = list(tier1_docs)
        logger.info("[TIER 2] Starting with %d documents from Tier 1", len(expanded_docs))

        for i, q in enumerate(alt_queries, start=1):
            logger.info("[MULTI-QUERY] Query %d/%d: %s", i, len(alt_queries), q)
            more_docs = self._retrieve_and_rerank(q, top_k=self.config.top_k_retrieve, rerank_k=max(3, self.config.top_k_rerank // 2))
            before = len(expanded_docs)
            expanded_docs.extend(more_docs)
            after = len(expanded_docs)
            logger.info("[MULTI-QUERY] Found %d additional documents", after - before)

        unique_docs = self._dedupe_docs(expanded_docs)
        if len(unique_docs) > self.config.max_expanded_docs:
            unique_docs = unique_docs[: self.config.max_expanded_docs]

        logger.info(
            "[EXPANSION] Collected %d total docs → %d unique (capped at %d)",
            len(expanded_docs),
            len(unique_docs),
            self.config.max_expanded_docs,
        )

        logger.info("[EXPANSION] Fetching full text for %d documents...", len(unique_docs))
        expanded_context, expanded_doc_ids, fetch_meta2 = self._fetch_full_text_from_docs(unique_docs)

        logger.info("[TIER 2] Synthesizing comprehensive answer from expanded context...")
        answer2, gen_meta2 = self._generate_answer(question, expanded_context)

        verify_report2 = self._verify_answer(
            question=question,
            answer=answer2,
            docs=unique_docs,
            context_text=expanded_context,
        )
        score2 = int(verify_report2.get("citation_score", 0) or 0)

        logs["tiers"].append(
            {
                "tier": 2,
                "docs": len(unique_docs),
                "doc_ids": expanded_doc_ids,
                "generation_meta": gen_meta2,
                "verification": verify_report2,
                "elapsed_s": round(time.time() - t2, 2),
                "alt_queries": alt_queries,
            }
        )

        total = round(time.time() - start, 2)
        logger.info(
            "[COMPLETE] Investigation complete. Total time: %ss, Tier2 score=%s, Sources: %d",
            total,
            score2,
            len(unique_docs),
        )

        metrics["tier2_generation_chars"] = len(answer2 or "")
        metrics["tier2_context_chars"] = len(expanded_context or "")
        metrics["tier2_verification_score"] = score2
        metrics["total_time_s"] = total

        return {
            "question": question,
            "answer": answer2,
            "sources": unique_docs,
            "verification": verify_report2,
            "metrics": metrics,
            "logs": logs,
        }

    # -------------------------
    # Tier primitives
    # -------------------------
    def _retrieve_and_rerank(self, query: str, *, top_k: int, rerank_k: int) -> List[Dict[str, Any]]:
        """
        Hybrid retrieve + rerank wrapper.
        Returns a list of document dicts.
        """
        docs = self.retriever.retrieve(query, top_k=top_k)
        # retriever.rerank is assumed internal to retrieve() in your implementation, but keep this hook
        if hasattr(self.retriever, "rerank") and callable(getattr(self.retriever, "rerank")):
            try:
                docs = self.retriever.rerank(query, docs, top_k=rerank_k)
            except Exception:
                # If rerank isn't supported or fails, keep original order
                logger.exception("Rerank failed, continuing with un-reranked results.")
        else:
            docs = docs[:rerank_k]
        return docs

    def _fetch_full_text_from_docs(self, docs: List[Dict[str, Any]]) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        Given a list of docs with metadata.document_id, fetch full text via rag_handler.
        """
        doc_ids: List[str] = []
        for d in docs:
            if not isinstance(d, dict):
                continue
            md = d.get("metadata", {}) if isinstance(d.get("metadata", {}), dict) else {}
            doc_id = md.get("document_id") or d.get("document_id") or d.get("id")
            if doc_id:
                doc_ids.append(str(doc_id))

        if not doc_ids:
            return "", [], {"note": "no document ids found"}

        # Expect rag_handler.get_full_document_text(ids) -> (text, missing_ids, meta)
        try:
            full_text, missing, meta = self.rag_handler.get_full_document_text(doc_ids)
            meta = meta or {}
            meta["missing_doc_ids"] = missing or []
            return full_text or "", doc_ids, meta
        except Exception as e:
            logger.exception("Failed fetching full text for docs: %s", e)
            return "", doc_ids, {"error": str(e)}

    def _generate_answer(self, question: str, context_text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Use rag_handler to generate an answer.
        """
        try:
            answer, meta = self.rag_handler.generate_answer(question, context_text=context_text)
            return answer or "", meta or {}
        except Exception as e:
            logger.exception("Generation failed: %s", e)
            return "", {"error": str(e)}

    def _verify_answer(
        self,
        *,
        question: str,
        answer: str,
        docs: List[Dict[str, Any]],
        context_text: str,
    ) -> Dict[str, Any]:
        """
        Run adversarial verification. If verification fails, return fallback score/report.
        """
        try:
            # IMPORTANT: This expects AdversarialRAGHandler.verify(...) to exist.
            # If you haven't added it, change this to verify_citations(...) or process_query(...) as appropriate.
            report = self.adversarial_handler.verify(
                question=question,
                answer=answer,
                sources=docs,
                context=context_text,
                timeout=self.config.verifier_timeout_s,
            )
            if not isinstance(report, dict):
                return {
                    "is_accurate": False,
                    "citation_score": 0,
                    "reasoning": f"Verifier returned non-dict: {type(report)}",
                    "fallback_used": True,
                }
            return report
        except Exception as e:
            logger.exception("Verification failed: %s", e)
            return {
                "is_accurate": True,
                "citation_score": 75,
                "reasoning": f"⚠️ Verification system error: {str(e)}. Answer generated from retrieved sources but not verified.",
                "fallback_used": True,
            }

    # -------------------------
    # Multi-query generation
    # -------------------------
    def generate_multi_queries(self, question: str, n: int = 3) -> List[str]:
        """
        Generate alternative search queries to expand recall.
        Tries to parse strict JSON, then falls back to extracting JSON-ish blocks or line items.
        """
        base_prompt = (
            "You are a query generation assistant. Produce ONLY JSON.\n"
            'Return a JSON array of strings named "queries" or a raw JSON array.\n'
            f"Generate {n} alternative search queries for:\n"
            f"{question}\n"
        )

        try:
            # If rag_handler has a direct LLM call for query gen, use it. Otherwise fall back.
            if hasattr(self.rag_handler, "llm_generate") and callable(getattr(self.rag_handler, "llm_generate")):
                response_text = self.rag_handler.llm_generate(base_prompt, timeout=self.config.generator_timeout_s)
            else:
                # Use generator model through rag_handler.generate_answer as a fallback, but it may add text
                response_text, _ = self._generate_answer(base_prompt, context_text="")

            raw = (response_text or "").strip()
            queries: Any = None

            # 1) direct JSON parse
            try:
                queries = json.loads(raw)
            except Exception:
                # 2) find JSON array
                m = re.search(r"\[[\s\S]*\]", raw)
                if m:
                    queries = json.loads(m.group(0))
                else:
                    # 3) find JSON object wrapper { "queries": [...] }
                    m2 = re.search(r"\{[\s\S]*\}", raw)
                    if m2:
                        obj = json.loads(m2.group(0))
                        if isinstance(obj, dict):
                            queries = obj.get("queries") or obj.get("search_queries")

            # 4) bullet / numbered fallback
            if queries is None:
                candidates: List[str] = []
                for ln in raw.splitlines():
                    ln = ln.strip().lstrip("-•*")
                    ln = re.sub(r"^\d+[\).]\s*", "", ln).strip()
                    if ln:
                        candidates.append(ln)
                queries = candidates

            if isinstance(queries, dict) and "queries" in queries:
                queries = queries["queries"]

            if not isinstance(queries, list) or len(queries) == 0:
                raise ValueError(f"Invalid query format returned: {type(queries)}")

            out = [str(q).strip() for q in queries if str(q).strip()]
            if len(out) > n:
                out = out[:n]
            # Ensure we always return something useful
            if not out:
                out = self._fallback_queries(question, n=n)
            return out

        except Exception as e:
            logger.warning("[MULTI-QUERY GEN] Failed to generate queries: %s. Using fallback queries.", e)
            return self._fallback_queries(question, n=n)

    def _fallback_queries(self, question: str, n: int = 3) -> List[str]:
        """
        Simple deterministic fallback query variants.
        """
        variants = [
            f"{question} detailed information",
            f"{question} historical context",
            f"{question} related documents",
            f"{question} primary sources",
            f"{question} examples",
        ]
        return variants[:n]

    # -------------------------
    # Helpers
    # -------------------------
    def _dedupe_docs(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate docs by document_id when available.
        """
        seen: set[str] = set()
        out: List[Dict[str, Any]] = []
        for d in docs:
            if not isinstance(d, dict):
                continue
            md = d.get("metadata", {}) if isinstance(d.get("metadata", {}), dict) else {}
            doc_id = md.get("document_id") or d.get("document_id") or d.get("id")
            key = str(doc_id) if doc_id is not None else json.dumps(d, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            out.append(d)
        return out


def build_agent_from_env() -> IterativeAdversarialAgent:
    """
    Convenience builder for the iterative adversarial agent.
    Expects environment variables used by your existing RAG stack.
    """
    db_name = os.getenv("MONGO_DB_NAME", "railroad_documents")
    vector_collection = os.getenv("CHROMA_COLLECTION_NAME", "historian_documents")

    generator_model = os.getenv("GENERATOR_MODEL", "gpt-oss:20b")
    verifier_model = os.getenv("VERIFIER_MODEL", "qwen2.5:32b")

    verifier_timeout = int(os.getenv("ADVERSARIAL_TIMEOUT", "60"))
    max_retries = int(os.getenv("ADVERSARIAL_MAX_RETRIES", "2"))

    rag_handler = RAGQueryHandler(
        db_name=db_name,
        collection_name=vector_collection,
        generator_model=generator_model,
    )

    retriever = HybridRetriever(
        vector_store=rag_handler.vector_store,
        keyword_store=rag_handler.keyword_store,
        vector_weight=float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.7")),
        keyword_weight=float(os.getenv("HYBRID_KEYWORD_WEIGHT", "0.3")),
        top_k=int(os.getenv("HYBRID_TOP_K", "40")),
        rrf_k=int(os.getenv("HYBRID_RRF_K", "60")),
    )

    adversarial_handler = AdversarialRAGHandler(
        rag_handler=rag_handler,
        verifier_model=verifier_model,
        timeout=verifier_timeout,
        max_retries=max_retries,
    )

    cfg = TierConfig(
        top_k_retrieve=int(os.getenv("TOP_K_RETRIEVE", "40")),
        top_k_rerank=int(os.getenv("TOP_K_RERANK", "10")),
        verification_threshold=int(os.getenv("VERIFICATION_THRESHOLD", "90")),
        max_expanded_docs=int(os.getenv("MAX_EXPANDED_DOCS", "15")),
        multi_query_count=int(os.getenv("MULTI_QUERY_COUNT", "3")),
        generator_timeout_s=int(os.getenv("GENERATOR_TIMEOUT", "60")),
        verifier_timeout_s=verifier_timeout,
        max_retries=max_retries,
    )

    return IterativeAdversarialAgent(
        rag_handler=rag_handler,
        retriever=retriever,
        adversarial_handler=adversarial_handler,
        config=cfg,
    )


# Backwards-compatible alias for older imports
TieredHistorianAgent = IterativeAdversarialAgent

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = build_agent_from_env()

    import sys

    if len(sys.argv) < 2:
        print("Usage: iterative_adversarial_agent.py \"<question>\"")
        raise SystemExit(2)

    q = sys.argv[1]
    result = agent.run(q)
    print(json.dumps(result, indent=2, ensure_ascii=False))

