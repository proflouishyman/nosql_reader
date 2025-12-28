# app/historian_agent/rag_base.py
# Created: 2025-12-24
# Purpose: Shared components for RAG pipeline - database access, LLM calls, utilities

"""
RAG Base Components - Shared infrastructure for all three RAG handlers.

Design Principles:
1. Keep your working retrieval pipeline intact
2. Don't abstract away complexity you need  
3. Config management without forced frameworks
4. Single source of truth for shared operations

What's Shared:
- Database connections (MongoDB)
- Document metadata hydration
- Full document text expansion (chunks → complete docs)
- LLM HTTP calls (direct Ollama, no LangChain)
- Utility functions (token counting, debug logging)

What's NOT Shared:
- Retrieval strategies (each handler has custom needs)
- Answer generation logic (different prompts/flows)
- Verification logic (only adversarial needs this)
"""

import sys
import os
import time
import requests
from typing import Dict, List, Any, Tuple, Optional
from pymongo import MongoClient
from bson import ObjectId
from dataclasses import dataclass

from config import APP_CONFIG, merge_config
from factories import DatabaseFactory


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RAGConfig:
    """Runtime configuration for RAG components."""
    llm_model: str
    llm_base_url: str
    llm_temperature: float
    llm_timeout: float
    llm_num_predict: int
    db_name: str
    top_k: int
    parent_retrieval_cap: int
    debug_mode: bool
    
    @classmethod
    def from_config(
        cls,
        llm_overrides: Optional[Dict[str, Any]] = None,
        retriever_overrides: Optional[Dict[str, Any]] = None
    ) -> "RAGConfig":
        """Create RAGConfig from APP_CONFIG with optional overrides."""
        llm_cfg = APP_CONFIG.llm_generator
        if llm_overrides:
            llm_cfg = merge_config(llm_cfg, llm_overrides)
        
        ret_cfg = APP_CONFIG.retriever
        if retriever_overrides:
            ret_cfg = merge_config(ret_cfg, retriever_overrides)
        
        return cls(
            llm_model=llm_cfg.model,
            llm_base_url=llm_cfg.base_url,
            llm_temperature=llm_cfg.temperature,
            llm_timeout=llm_cfg.timeout_s,
            llm_num_predict=llm_cfg.num_predict,
            db_name=APP_CONFIG.database.db_name,
            top_k=ret_cfg.parent_retrieval_cap,
            parent_retrieval_cap=ret_cfg.parent_retrieval_cap,
            debug_mode=APP_CONFIG.debug_mode,
        )


# ============================================================================
# Utilities
# ============================================================================

def count_tokens(text: str) -> int:
    """Rough token count: 4 chars ≈ 1 token."""
    return len(text) // 4


def debug_print(msg: str, detail: str = None, tokens: int = 0, level: str = "INFO"):
    """Print debug message if debug mode enabled."""
    if not APP_CONFIG.debug_mode:
        return
    
    timestamp = time.strftime("%H:%M:%S")
    token_str = f" | 🔋 {tokens} tokens" if tokens > 0 else ""
    icon = "🔍" if level == "INFO" else "⚠️" if level == "WARN" else "❌"
    
    sys.stderr.write(f"{icon} [{timestamp}] {msg}{token_str}\n")
    if detail:
        sys.stderr.write(f"   | {detail}\n")
    sys.stderr.flush()


# ============================================================================
# Database Operations
# ============================================================================

class DocumentStore:
    """Handles MongoDB document operations."""
    
    def __init__(self):
        """Initialize database connections via factory."""
        collections = DatabaseFactory.get_collections()
        self.docs_coll = collections['documents']
        self.chunks_coll = collections['chunks']
        debug_print("DocumentStore initialized", f"DB: {APP_CONFIG.database.db_name}")
    
    def hydrate_parent_metadata(self, doc_ids: List[str]) -> Dict[str, Dict]:
        """Fetch parent document metadata for given document IDs."""
        query_ids = [ObjectId(d) if ObjectId.is_valid(d) else d for d in doc_ids]
        docs = self.docs_coll.find({"_id": {"$in": query_ids}})
        return {str(doc["_id"]): doc for doc in docs}
    
    def get_full_document_text(
        self,
        doc_ids: List[str]
    ) -> Tuple[str, Dict[str, str], float]:
        """
        Fetch complete text for multiple documents (small-to-big expansion).
        
        Returns: (full_text, filename_mapping, latency)
        """
        t_start = time.time()
        meta = self.hydrate_parent_metadata(doc_ids)
        
        full_text_block = ""
        mapping = {}
        
        for d_id in doc_ids:
            fname = meta.get(d_id, {}).get("filename", f"Doc-{d_id[:8]}")
            mapping[fname] = d_id
            
            chunks = list(
                self.chunks_coll.find({"document_id": d_id}).sort("chunk_index", 1)
            )
            
            full_text_block += f"\n--- FULL DOCUMENT: {fname} (ID: {d_id}) ---\n"
            for c in chunks:
                full_text_block += c.get("text", "") + "\n"
        
        latency = time.time() - t_start
        debug_print(
            "Document Expansion",
            f"Fetched {len(doc_ids)} full documents",
            tokens=count_tokens(full_text_block)
        )
        
        return full_text_block, mapping, latency


# ============================================================================
# LLM Operations
# ============================================================================

class LLMClient:
    """Direct HTTP calls to Ollama (no LangChain)."""
    
    def __init__(self, config: RAGConfig):
        """Initialize LLM client with config."""
        self.config = config
        debug_print(
            "LLM Client initialized",
            f"Model: {config.llm_model}, URL: {config.llm_base_url}"
        )
    
    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
        num_predict: Optional[int] = None,
    ) -> str:
        """Generate text using Ollama."""
        temp = temperature if temperature is not None else self.config.llm_temperature
        tout = timeout if timeout is not None else self.config.llm_timeout
        npred = num_predict if num_predict is not None else self.config.llm_num_predict
        
        debug_print(
            "LLM Generate",
            f"Model: {self.config.llm_model}, Temp: {temp}",
            tokens=count_tokens(prompt)
        )
        
        t_start = time.time()
        resp = requests.post(
            f"{self.config.llm_base_url}/api/generate",
            json={
                "model": self.config.llm_model,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "options": {
                    "temperature": temp,
                    "num_predict": npred,
                    "num_ctx": 131072,
                    "repeat_penalty": 1.15,
                }
            },
            timeout=tout
        )
        
        resp.raise_for_status()
        answer = resp.json().get("response", "")
        
        elapsed = time.time() - t_start
        debug_print("LLM Complete", f"{count_tokens(answer)} tokens in {elapsed:.1f}s")
        
        return answer
    
    def generate_with_retry(
        self,
        prompt: str,
        system: str = "",
        max_retries: int = 3,
        **kwargs
    ) -> str:
        """Generate with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                return self.generate(prompt, system, **kwargs)
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    debug_print(
                        "LLM Retry",
                        f"Attempt {attempt + 1} failed, waiting {wait_time}s",
                        level="WARN"
                    )
                    time.sleep(wait_time)
                else:
                    debug_print("LLM Failed", f"All {max_retries} attempts exhausted", level="ERROR")
                    raise


__all__ = ["RAGConfig", "count_tokens", "debug_print", "DocumentStore", "LLMClient"]
