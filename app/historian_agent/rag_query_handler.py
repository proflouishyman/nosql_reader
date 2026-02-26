# app/historian_agent/rag_query_handler.py
# UPDATED: 2025-12-28 - Integrated with llm_abstraction layer

"""
RAG Query Handler - Basic retrieval + generation pipeline.

CHANGES FROM ORIGINAL:
- Removed scattered env reads (LLM_MODEL, OLLAMA_BASE_URL, etc.)
- Uses LLMClient from llm_abstraction instead of raw requests
- Uses DocumentStore from rag_base for database operations
- Preserves EXACT retrieval pipeline (hybrid, reranking, expansion)
- Backward compatible with existing routes
- Fixed all imports and missing definitions
"""

import sys
import os
import time
from typing import Dict, List, Any, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

# Shared base components - FIXED: Import path
import sys
sys.path.insert(0, '/app')  # Ensure app is in path

from rag_base import DocumentStore, count_tokens, debug_print
from llm_abstraction import LLMClient, LLMResponse
from config import APP_CONFIG

# Your existing components (unchanged)
from embeddings import EmbeddingService
from vector_store import get_vector_store
from retrievers import HybridRetriever, VectorRetriever, KeywordRetriever
from reranking import DocumentReranker


# Configuration - FIXED: Use APP_CONFIG instead of direct env reads
TOP_K = APP_CONFIG.retriever.top_k
RETRIEVAL_POOL_SIZE = APP_CONFIG.retriever.retrieval_pool_size

# System prompt for generation
SYSTEM_PROMPT = """
Avoid repetition: do not restate the same point, do not reuse the same opening phrases,
do not repeat yourself. If you have nothing new to add, stop.
You are an expert historian committed to accuracy.
False positives are much worse than false negatives.

You are a research assistant specialising in the
Baltimore & Ohio Railroad Relief Department archive (1880-1945).

## Citation format
Reference sources inline as [Source N] where N is the source number.
Every factual claim must carry a citation.
Use only this exact citation token format.
Do not write "(Source N: ...)", "(Sources N-M)", or include document IDs inside citation markers.

## Uncertainty
If a claim rests on a single source, note it explicitly:
"This is drawn from a single record and should be corroborated."
If the evidence is ambiguous or partial, say so in plain prose.
Do not hedge with percentages or confidence scores.

## Evidence gaps
After your answer, add a brief paragraph beginning "What the record does not show:"
describing what you looked for but could not find in the retrieved sources.
If the corpus appears to have no relevant material, say so directly.

## Where to look next
End every answer with 1-3 specific follow-up leads in a section beginning
"To investigate further:". These should be concrete names, boxes, dates,
or record types that appear related to the question but were not retrieved.
"""


class RAGQueryHandler:
    """
    Basic RAG query handler.
    
    Pipeline (YOUR EXISTING LOGIC - unchanged):
        1. Hybrid retrieval (vector + keyword) → Pool of 40 chunks
        2. Reranking → Top 10 chunks
        3. De-duplication → Unique parent document IDs
        4. Full document expansion → Complete text
        5. LLM generation → Answer
    """
    
    def __init__(self):
        """Initialize RAG query handler."""
        debug_print("Initializing RAGQueryHandler", f"Top-K: {TOP_K}")
        
        # FIXED: Initialize shared base components
        self.doc_store = DocumentStore()
        self.llm = LLMClient()
        
        # Your existing retrieval setup (UNCHANGED!)
        self.embedding_service = EmbeddingService(
            provider=APP_CONFIG.embedding.provider,
            model=APP_CONFIG.embedding.model
        )
        self.vector_store = get_vector_store(store_type="chroma")
        self.reranker = DocumentReranker()
        
        # Build hybrid retriever (YOUR EXISTING LOGIC)
        v_ret = VectorRetriever(
            self.vector_store,
            self.embedding_service,
            self.doc_store.chunks_coll,
            top_k=RETRIEVAL_POOL_SIZE
        )
        
        class ConfigShim:
            """Temporary shim for keyword retriever config."""
            context_fields = ["text", "ocr_text"]
        
        k_ret = KeywordRetriever(
            self.doc_store.chunks_coll,
            ConfigShim(),
            top_k=RETRIEVAL_POOL_SIZE
        )
        
        self.hybrid_retriever = HybridRetriever(
            v_ret,
            k_ret,
            top_k=RETRIEVAL_POOL_SIZE
        )
        
        debug_print("Initialization complete")
    
    def hydrate_parent_metadata(self, doc_ids: List[str]) -> Dict[str, Dict]:
        """
        Fetch parent document metadata (delegates to DocumentStore).
        
        Kept for backward compatibility - just wraps DocumentStore method.
        """
        return self.doc_store.hydrate_parent_metadata(doc_ids)
    
    def get_full_document_text(
        self,
        doc_ids: List[str]
    ) -> Tuple[str, Dict[str, str], float]:
        """
        Fetch complete text for documents (delegates to DocumentStore).
        
        Kept for backward compatibility - just wraps DocumentStore method.
        """
        return self.doc_store.get_full_document_text(doc_ids)
    
    def process_query(
        self,
        question: str,
        context: str = "",
        label: str = "LLM"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Process a RAG query.
        
        YOUR EXISTING PIPELINE - preserved exactly!
        Only change: LLM call uses llm_abstraction instead of raw requests.
        
        Args:
            question: User's research question
            context: Optional pre-assembled context (bypasses retrieval)
            label: Label for logging
        
        Returns:
            Tuple of (answer, metrics)
            
        Raises:
            Exception: If LLM generation fails critically
        """
        # FIXED: Initialize metrics properly
        metrics = {
            "retrieval_time": 0.0,
            "llm_time": 0.0,
            "tokens": 0,
            "doc_count": 0,
            "sources": {},
            "context": ""
        }
        
        t_start = time.time()
        
        # YOUR EXISTING RETRIEVAL LOGIC - unchanged!
        if not context:
            r_start = time.time()
            
            # 1. Broad Retrieval (Pool of 40)
            raw_chunks = self.hybrid_retriever.get_relevant_documents(question)
            
            # 2. Reranking (Top 10)
            reranked = self.reranker.rerank(question, raw_chunks, top_k=10)
            
            # 3. De-duplication (Unique Parents)
            unique_ids = []
            seen = set()
            for c in reranked:
                d_id = c.metadata.get("document_id")
                if d_id and d_id not in seen:
                    unique_ids.append(d_id)
                    seen.add(d_id)
            
            # 4. Cap to TOP_K
            unique_ids = unique_ids[:TOP_K]
            
            debug_print(
                f"Retrieval: {len(raw_chunks)} -> "
                f"Rerank: {len(reranked)} -> "
                f"Expansion: {len(unique_ids)}"
            )
            
            # 5. Small-to-Big Expansion (via shared DocumentStore)
            # FIXED: Properly capture mapping
            context, sources_list, _ = self.doc_store.get_full_document_text(unique_ids)
            
            metrics["retrieval_time"] = time.time() - r_start
            metrics["doc_count"] = len(unique_ids)
            metrics["sources"] = sources_list  # Already a list!
            metrics["context"] = context
            # # Normalize sources list for API compatibility, need to add mapping if to work
            # if isinstance(mapping, dict):
            #     sources_list = [
            #         {
            #             "label": label,
            #             "id": info["id"],
            #             "display_name": info.get("display_name", label),
            #         }
            #         for label, info in mapping.items()
            #     ]
            # else:
            #     sources_list = []

            # metrics["sources"] = sources_list
        
        # Build prompt (YOUR EXISTING FORMAT)
        prompt = f"TASK: {question}\n\nCONTEXT:\n{context}"
        metrics["tokens"] = count_tokens(prompt)
        
        # LLM generation (UPDATED: uses llm_abstraction)
        try:
            l_start = time.time()
            
            response = self.llm.generate_simple(
                prompt=prompt,
                system=SYSTEM_PROMPT,
                profile="quality",  # Uses config from llm_profiles
                temperature=0.1,
                num_predict=4000,
            )
            
            # FIXED: Proper error handling
            if not response.success:
                # Return error message with metrics rather than raising
                error_msg = f"LLM generation failed: {str(response.error)}"
                debug_print(f"ERROR: {error_msg}")
                return error_msg, metrics
            
            answer = response.content
            metrics["llm_time"] = time.time() - l_start
            
            # FIXED: Add LLM metrics
            metrics["llm_tokens"] = response.tokens
            metrics["llm_model"] = response.model_name
            metrics["llm_provider"] = response.provider_used
        
        except Exception as e:
            # FIXED: Return tuple consistently
            error_msg = f"Exception during LLM generation: {str(e)}"
            debug_print(f"ERROR: {error_msg}")
            return error_msg, metrics
        
        metrics["total_time"] = time.time() - t_start
        
        # FIXED: Always return tuple of (answer, metrics)
        return answer, metrics
    
    def close(self):
        """Clean up resources."""
        # MongoDB client closed by factory singleton
        pass


# For backward compatibility and CLI testing
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rag_query_handler.py 'your question'")
        sys.exit(1)
    
    question = " ".join(sys.argv[1:])
    handler = RAGQueryHandler()
    
    try:
        answer, metrics = handler.process_query(question)
        print(answer)
        print(f"\nTime: {metrics['total_time']:.1f}s, Sources: {metrics['doc_count']}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
