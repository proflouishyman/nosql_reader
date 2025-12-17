#!/usr/bin/env python3
import sys
import os
import requests
import time
from typing import Dict, List, Any
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv

# Explicitly load the .env file
load_dotenv()

# ============================================================================
# MASTER CONFIG & DEBUG TOGGLE
# ============================================================================
DEBUG = os.environ.get("DEBUG_MODE", "0") == "1"

TOP_K = int(os.environ.get("HISTORIAN_AGENT_TOP_K", 100))
MAX_CONTEXT_TOKENS = int(os.environ.get("MAX_CONTEXT_TOKENS", 100000))
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-oss:20b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://admin:secret@mongodb:27017/admin")
DB_NAME = os.environ.get("DB_NAME", "railroad_documents")

TITLE_FIELDS = ["filename", "title", "relative_path", "source_name"]
CONTENT_FIELDS = ["text", "ocr_text", "content", "summary"]

def debug_print(msg: str):
    if DEBUG:
        sys.stderr.write(f"🔍 [DEBUG] {msg}\n")
        sys.stderr.flush()

# ============================================================================
# CORE HANDLER
# ============================================================================

class RAGQueryHandler:
    def __init__(self):
        debug_print("Initializing Database & Vector Store connections...")
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        self.docs_coll = self.db['documents']
        
        from embeddings import EmbeddingService
        from vector_store import get_vector_store
        from retrievers import HybridRetriever, VectorRetriever, KeywordRetriever
        
        self.embedding_service = EmbeddingService(provider="ollama", model="qwen3-embedding:0.6b")
        self.vector_store = get_vector_store(store_type="chroma")
        
        v_ret = VectorRetriever(self.vector_store, self.embedding_service, self.db['document_chunks'], top_k=TOP_K)
        class ConfigShim: context_fields = CONTENT_FIELDS
        k_ret = KeywordRetriever(self.db['document_chunks'], ConfigShim(), top_k=TOP_K)
        
        self.hybrid_retriever = HybridRetriever(v_ret, k_ret, top_k=TOP_K)
        debug_print(f"RAG Handler Ready. Mode: {'DEBUG' if DEBUG else 'PRODUCTION'}")

    def get_best_field(self, data: Dict, field_list: List[str], default: str = "") -> str:
        """Exposed helper for retrieving first available metadata field."""
        for field in field_list:
            if data.get(field): return str(data[field]).strip()
        return default

    def hydrate_parent_metadata(self, doc_ids: List[str]) -> Dict[str, Dict]:
        """Exposed method to bulk fetch parent documents from MongoDB."""
        if not doc_ids: return {}
        query_ids = []
        for d in doc_ids:
            query_ids.append(d)
            try:
                query_ids.append(ObjectId(d))
            except:
                pass
        cursor = self.docs_coll.find({"_id": {"$in": query_ids}})
        return {str(doc["_id"]): doc for doc in cursor}

    def process_query(self, question: str):
        overall_start = time.time()
        
        # 1. Retrieval
        t_start = time.time()
        chunks = self.hybrid_retriever.get_relevant_documents(question)
        debug_print(f"Phase 1: Retrieval | {len(chunks)} chunks | {time.time() - t_start:.2f}s")

        # 2. Hydration
        t_start = time.time()
        parent_ids = list(set([c.metadata.get("document_id") for c in chunks if c.metadata.get("document_id")]))
        meta_map = self.hydrate_parent_metadata(parent_ids)
        debug_print(f"Phase 2: Hydration | {len(meta_map)} files linked | {time.time() - t_start:.2f}s")

        # 3. Assembly
        t_start = time.time()
        context_parts, sources, tokens = [], [], 0
        for chunk in chunks:
            p_id = str(chunk.metadata.get("document_id"))
            parent = meta_map.get(p_id, {})
            fname = self.get_best_field(parent, TITLE_FIELDS, f"Doc-{p_id[:8]}")
            text = chunk.page_content or self.get_best_field(chunk.metadata, CONTENT_FIELDS)
            
            entry = f"--- SOURCE: {fname} ---\nTEXT: {text}\n\n"
            t_count = len(entry)//4
            
            if tokens + t_count > MAX_CONTEXT_TOKENS:
                debug_print("⚠️ CONTEXT FULL: Token limit reached before processing all chunks.")
                break
            
            context_parts.append(entry)
            tokens += t_count
            sources.append({"filename": fname, "id": p_id})
        
        if tokens > (MAX_CONTEXT_TOKENS * 0.9):
            debug_print(f"☢️ HIGH LOAD: Using {tokens} tokens (over 90% of limit).")
            
        debug_print(f"Phase 3: Assembly  | {tokens} tokens packed | {time.time() - t_start:.2f}s")

        # 4. LLM Generation
        t_start = time.time()
        debug_print(f"Phase 4: LLM Start | Model: {LLM_MODEL}")
        
        resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json={
            "model": LLM_MODEL,
            "prompt": f"Context:\n{''.join(context_parts)}\n\nQuestion: {question}\nAnswer:",
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 10000}
        }, timeout=400)
        
        answer = resp.json().get("response", "")
        duration = time.time() - t_start
        debug_print(f"Phase 4: LLM End   | Generated in {duration:.2f}s")
        
        return answer, sources, time.time() - overall_start

    def close(self):
        self.client.close()