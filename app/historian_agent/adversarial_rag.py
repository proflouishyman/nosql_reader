#!/usr/bin/env python3
"""
Adversarial RAG - Consistent & Configurable Version
Syncs defaults with rag_query_handler.py while allowing dynamic overrides.
"""

import sys
import os
import logging
import re
import json
import time
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Import synchronized constants and base handler
from rag_query_handler import (
    RAGQueryHandler, 
    OLLAMA_BASE_URL, 
    LLM_MODEL, 
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    SYSTEM_PROMPT,
    CONTENT_FIELDS,
    TITLE_FIELDS,
    VECTOR_WEIGHT,   # Default from Env/Handler
    KEYWORD_WEIGHT,  # Default from Env/Handler
    TOP_K            # Default from Env/Handler
)
from reranking import DocumentReranker

# Adversarial Defaults
DEFAULT_RERANK_K = 20
DEFAULT_FINAL_K = 10

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CritiqueResult:
    confidence_score: float
    issues_found: List[str]
    recommendation: str 
    revised_answer: Optional[str] = None

# ============================================================================
# ENHANCED ADVERSARIAL HANDLER
# ============================================================================

class AdversarialRAGHandler:
    def __init__(self):
        # Initialize base RAG components
        self.rag_handler = RAGQueryHandler()
        
        # Initialize Reranker with system defaults
        self.reranker = DocumentReranker(
            cross_encoder_weight=0.85,
            temporal_weight=0.10,
            entity_weight=0.05
        )
        logger.info("✓ Adversarial Handler Initialized with System Defaults")

    def process_query(self, question: str, overrides: Dict[str, Any] = None):
        """
        Runs the full adversarial pipeline with optional parameter overrides.
        
        Supported Overrides:
            - initial_k: int (Initial retrieval depth)
            - final_k: int (Count of docs sent to LLM)
            - vector_w: float (Hybrid retrieval vector weight)
            - keyword_w: float (Hybrid retrieval keyword weight)
            - temperature: float (LLM creativity)
        """
        # Set runtime parameters (Priority: Overrides > Env/Constants)
        overrides = overrides or {}
        run_params = {
            "initial_k": overrides.get("initial_k", TOP_K),
            "final_k": overrides.get("final_k", DEFAULT_FINAL_K),
            "vector_w": overrides.get("vector_w", VECTOR_WEIGHT),
            "keyword_w": overrides.get("keyword_w", KEYWORD_WEIGHT),
            "temperature": overrides.get("temperature", LLM_TEMPERATURE)
        }

        start_time = time.time()
        logger.info(f"Query Process Started | Parameters: {run_params}")

        # 1. HYBRID RETRIEVAL (Intercepted for dynamic weights)
        # We manually call the retrievers to respect the dynamic weights
        self.rag_handler.hybrid_retriever.vector_weight = run_params["vector_w"]
        self.rag_handler.hybrid_retriever.keyword_weight = run_params["keyword_w"]
        self.rag_handler.hybrid_retriever.top_k = run_params["initial_k"]
        
        raw_chunks = self.rag_handler.hybrid_retriever.get_relevant_documents(question)
        
        # 2. RERANKING (Top initial_k -> final_k)
        reranked_results = self.reranker.rerank(
            query=question,
            documents=raw_chunks,
            top_k=run_params["final_k"]
        )
        reranked_docs = [r.document for r in reranked_results]

        # 3. METADATA HYDRATION
        parent_ids = list(set([d.metadata.get("document_id") for d in reranked_docs]))
        parent_meta_map = self.rag_handler._hydrate_parent_metadata(parent_ids)

        # 4. CONTEXT ASSEMBLY
        context_parts = []
        sources = []
        for doc in reranked_docs:
            p_id = str(doc.metadata.get("document_id"))
            parent_doc = parent_meta_map.get(p_id, {})
            
            fname = self.rag_handler._get_best_field(parent_doc, TITLE_FIELDS, f"Doc-{p_id[:8]}")
            text = doc.page_content or self.rag_handler._get_best_field(doc.metadata, CONTENT_FIELDS)
            
            context_parts.append(f"--- SOURCE: {fname} ---\nTEXT: {text}\n")
            sources.append({"filename": fname, "id": p_id})

        full_context = "".join(context_parts)

        # 5. INITIAL GENERATION
        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{full_context}\n\nQuestion: {question}\n\nAnswer:"
        initial_answer = self._call_ollama(prompt, run_params["temperature"])

        # 6. ADVERSARIAL CRITIQUE
        critique = self._critique_answer(question, initial_answer, full_context)

        total_time = time.time() - start_time
        return {
            "answer": critique.revised_answer if critique.revised_answer else initial_answer,
            "sources": sources,
            "critique": critique,
            "latency": f"{total_time:.2f}s"
        }

    def _critique_answer(self, question: str, answer: str, context: str) -> CritiqueResult:
        critique_prompt = f"""
        Analyze the following answer based strictly on the provided context.
        Identify any claims not found in the text or incorrect citations.
        
        CONTEXT:
        {context[:10000]}
        
        QUESTION: {question}
        ANSWER: {answer}
        
        Output Format:
        CONFIDENCE: <0.0 - 1.0>
        ISSUES: <list issues>
        REVISION: <corrected version or 'NONE'>
        """
        
        # Critique always uses a fixed, low temperature for consistency
        response_text = self._call_ollama(critique_prompt, temperature=0.1)
        
        confidence = 0.8
        conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response_text)
        if conf_match: confidence = float(conf_match.group(1))

        rev_match = re.search(r'REVISION:\s*(.*)', response_text, re.DOTALL)
        revised = None
        if rev_match and "NONE" not in rev_match.group(1).upper():
            revised = rev_match.group(1).strip()

        return CritiqueResult(confidence_score=confidence, issues_found=[], recommendation="accept" if confidence > 0.7 else "revise", revised_answer=revised)

    def _call_ollama(self, prompt: str, temperature: float):
        payload = {
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": LLM_MAX_TOKENS}
        }
        try:
            r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=300)
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception as e:
            return f"LLM Error: {e}"

    def close(self):
        self.rag_handler.close()

# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python adversarial_rag.py \"Question\"")
        sys.exit(1)

    query = sys.argv[1]
    handler = AdversarialRAGHandler()
    
    try:
        # Example of using an override for a high-accuracy run
        result = handler.process_query(query, overrides={"final_k": 15, "temperature": 0.1})
        
        print("\n" + "="*80)
        print(f"ADVERSARIAL ANSWER (Confidence: {result['critique'].confidence_score})")
        print("="*80)
        print(f"\n{result['answer']}\n")
        print("="*80)
        print("SOURCES (FILENAME\tID):")
        unique_sources = {s['filename']: s['id'] for s in result['sources']}
        for fname, fid in sorted(unique_sources.items()):
            print(f"{fname}\t{fid}")
    finally:
        handler.close()