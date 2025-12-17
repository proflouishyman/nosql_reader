#!/usr/bin/env python3
"""
RAG Query Handler - Research Format Version
Outputs tab-separated Filenames and IDs for easy cross-script integration.
"""

import sys
import os
import logging
import requests
import time
import psutil
from typing import Dict, List, Any
from pymongo import MongoClient
from bson import ObjectId

# Project-specific imports
try:
    from embeddings import EmbeddingService
    from vector_store import get_vector_store
    from retrievers import VectorRetriever, KeywordRetriever, HybridRetriever
except ImportError as e:
    print(f"Error: Missing required module: {e}")
    sys.exit(1)

# ============================================================================
# GLOBAL CONFIGURATION - MASTER CONTROLS
# ============================================================================

# 1. DATABASE SCHEMA PRIORITIES
CONTENT_FIELDS = ["text", "ocr_text", "content", "summary"]
TITLE_FIELDS = ["filename", "title", "relative_path", "source_name"]
SUMMARY_FIELDS = ["summary", "description", "abstract"]

# 2. RETRIEVAL PARAMETERS
TOP_K = 30
MAX_CONTEXT_TOKENS = 80000 
VECTOR_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3

# 3. LLM OUTPUT PARAMETERS
LLM_MAX_TOKENS = 20000
LLM_TEMPERATURE = 0.2

# 4. CONNECTION SETTINGS
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://admin:secret@mongodb:27017/admin")
DB_NAME = "railroad_documents"
CHUNKS_COLLECTION = "document_chunks"
DOCS_COLLECTION = "documents"

# 5. PROVIDER SETTINGS
LLM_MODEL = "gpt-oss:20b"
OLLAMA_BASE_URL = "http://host.docker.internal:11434"
EMBEDDING_PROVIDER = "ollama"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"

# 6. SYSTEM PROMPT
SYSTEM_PROMPT = """You are a knowledgeable historian assistant specializing in Baltimore & Ohio Railroad history. 
Base your answers on the provided context, which includes both document summaries and specific text fragments. 
Cite filenames precisely. If information is missing, state it clearly."""

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# UTILITIES
# ============================================================================

class MemoryTracker:
    @staticmethod
    def get_usage_mb() -> float:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    @staticmethod
    def log(stage: str):
        print(f"--- [MEMORY] {stage:30} : {MemoryTracker.get_usage_mb():8.2f} MB ---")

# ============================================================================
# CORE HANDLER
# ============================================================================

class RAGQueryHandler:
    def __init__(self):
        MemoryTracker.log("Init: Start")
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        self.chunks_coll = self.db[CHUNKS_COLLECTION]
        self.docs_coll = self.db[DOCS_COLLECTION]
        
        self.embedding_service = EmbeddingService(provider=EMBEDDING_PROVIDER, model=EMBEDDING_MODEL)
        self.vector_store = get_vector_store(store_type="chroma")
        
        self.vector_retriever = VectorRetriever(
            self.vector_store, self.embedding_service, self.chunks_coll, top_k=TOP_K
        )
        
        class ConfigShim: context_fields = CONTENT_FIELDS
        self.keyword_retriever = KeywordRetriever(self.chunks_coll, ConfigShim(), top_k=TOP_K)
        
        self.hybrid_retriever = HybridRetriever(
            self.vector_retriever, self.keyword_retriever,
            vector_weight=VECTOR_WEIGHT, keyword_weight=KEYWORD_WEIGHT, top_k=TOP_K
        )
        MemoryTracker.log("Init: Components Ready")

    def _get_best_field(self, data: Dict, field_list: List[str], default: str = "") -> str:
        for field in field_list:
            value = data.get(field)
            if value and str(value).strip():
                return str(value).strip()
        return default

    def _hydrate_parent_metadata(self, doc_ids: List[str]) -> Dict[str, Dict]:
        """Fetch parent documents by both string and BSON ObjectId."""
        if not doc_ids: return {}
        
        query_ids = []
        for d_id in doc_ids:
            query_ids.append(d_id)
            try:
                query_ids.append(ObjectId(d_id))
            except:
                pass
        
        cursor = self.docs_coll.find({"_id": {"$in": query_ids}})
        meta_map = {}
        for doc in cursor:
            meta_map[str(doc["_id"])] = doc
        
        logger.info(f"Successfully hydrated {len(meta_map)} documents from MongoDB.")
        return meta_map

    def process_query(self, question: str):
        start_time = time.time()
        MemoryTracker.log("Query: Start Retrieval")

        chunks = self.hybrid_retriever.get_relevant_documents(question)
        
        # Hydration Step
        parent_ids = list(set([c.metadata.get("document_id") for c in chunks if c.metadata.get("document_id")]))
        parent_meta_map = self._hydrate_parent_metadata(parent_ids)
        
        context_parts = []
        sources = []
        total_tokens = 0

        for i, chunk in enumerate(chunks):
            p_id = str(chunk.metadata.get("document_id"))
            parent_doc = parent_meta_map.get(p_id, {})
            
            filename = self._get_best_field(parent_doc, TITLE_FIELDS, f"Doc-{p_id[:8]}")
            summary = self._get_best_field(parent_doc, SUMMARY_FIELDS, "No summary available.")
            text = chunk.page_content or self._get_best_field(chunk.metadata, CONTENT_FIELDS)
            
            if not text: continue

            entry = f"--- SOURCE: {filename} ---\nSUMMARY: {summary}\nTEXT: {text}\n\n"
            tokens = len(entry) // 4
            
            if total_tokens + tokens > MAX_CONTEXT_TOKENS:
                break
            
            context_parts.append(entry)
            total_tokens += tokens
            # Store filename and ID for later display
            sources.append({"filename": filename, "id": p_id})

        MemoryTracker.log("Query: Sending to LLM")
        prompt = f"{SYSTEM_PROMPT}\n\nContext Documents:\n{''.join(context_parts)}\n\nQuestion: {question}\n\nAnswer:"
        
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": LLM_TEMPERATURE,
                        "num_predict": LLM_MAX_TOKENS
                    }
                },
                timeout=300
            )
            response.raise_for_status()
            answer = response.json().get("response", "Error: No response from LLM.")
        except Exception as e:
            answer = f"System Error during generation: {str(e)}"
        
        latency = time.time() - start_time
        return answer, sources, latency

    def close(self):
        self.client.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rag_query_handler.py \"[Question]\"")
        sys.exit(1)
    
    query = sys.argv[1]
    handler = RAGQueryHandler()
    
    try:
        answer, sources, latency = handler.process_query(query)
        
        print("\n" + "="*80)
        print(f"HISTORIAN AGENT ANSWER ({latency:.2f}s)")
        print("="*80)
        print(f"\n{answer}\n")
        print("="*80)
        
        print("SOURCES UTILIZED (HYDRATED):")
        # Generate unique pairs of (Filename, ID)
        unique_source_pairs = {}
        for s in sources:
            unique_source_pairs[s['filename']] = s['id']
            
        # Display tab-separated format: FILENAME \t DOCUMENT_ID
        sorted_filenames = sorted(unique_source_pairs.keys())
        for fname in sorted_filenames:
            doc_id = unique_source_pairs[fname]
            print(f"{fname}\t{doc_id}")
            
        print("="*80 + "\n")
        
        MemoryTracker.log("Process Finished")
    finally:
        handler.close()