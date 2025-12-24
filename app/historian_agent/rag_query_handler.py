#!/usr/bin/env python3
import sys, os, requests, time
from typing import Dict, List, Any, Tuple
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
DEBUG = os.environ.get("DEBUG_MODE", "0") == "1"
TOP_K = int(os.environ.get("HISTORIAN_AGENT_TOP_K", 10))
RETRIEVAL_POOL_SIZE = int(os.environ.get("RETRIEVAL_POOL_SIZE", 100)) # Fetch 100, Rerank to 10
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-oss:20b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://admin:secret@mongodb:27017/admin")
DB_NAME = os.environ.get("DB_NAME", "railroad_documents")
SYSTEM_PROMPT = "Avoid repetition: do not restate the same point, do not reuse the same opening phrases, do not repeat yourself. If you have nothing new to add, stop.  You are an expert historian commited to accuracy. False positives are much worse than false negatives."





def count_tokens(text: str) -> int:
    return len(text) // 4

def debug_print(msg: str, detail: str = None, tokens: int = 0, level: str = "INFO"):
    if not DEBUG: return
    timestamp = time.strftime("%H:%M:%S")
    token_str = f" | 🔋 {tokens} tokens" if tokens > 0 else ""
    icon = "🔍" if level == "INFO" else "⚠️" if level == "WARN" else "🚀"
    sys.stderr.write(f"{icon} [{timestamp}] {msg}{token_str}\n")
    if detail:
        sys.stderr.write(f"   | {detail}\n")
    sys.stderr.flush()

class RAGQueryHandler:
    def __init__(self):
        debug_print("Initializing RAGQueryHandler...", f"DB: {DB_NAME}")
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        self.docs_coll = self.db['documents']
        self.chunks_coll = self.db['document_chunks']
        
        # Load AI Components
        from .embeddings import EmbeddingService
        from .vector_store import get_vector_store
        from .retrievers import HybridRetriever, VectorRetriever, KeywordRetriever
        from .reranking import DocumentReranker 
        
        self.embedding_service = EmbeddingService(provider="ollama", model="qwen3-embedding:0.6b")
        self.vector_store = get_vector_store(store_type="chroma")
        self.reranker = DocumentReranker()
        
        # Retrieve larger pool for reranking
        v_ret = VectorRetriever(self.vector_store, self.embedding_service, self.chunks_coll, top_k=RETRIEVAL_POOL_SIZE)
        class ConfigShim: context_fields = ["text", "ocr_text"]
        k_ret = KeywordRetriever(self.chunks_coll, ConfigShim(), top_k=RETRIEVAL_POOL_SIZE)
        self.hybrid_retriever = HybridRetriever(v_ret, k_ret, top_k=RETRIEVAL_POOL_SIZE)

    def hydrate_parent_metadata(self, doc_ids):
        query_ids = [ObjectId(d) if ObjectId.is_valid(d) else d for d in doc_ids]
        return {str(doc["_id"]): doc for doc in self.docs_coll.find({"_id": {"$in": query_ids}})}

    def get_full_document_text(self, doc_ids: List[str]) -> Tuple[str, Dict[str, str], float]:
        t_start = time.time()
        full_text_block = ""
        meta = self.hydrate_parent_metadata(doc_ids)
        mapping = {}
        
        for d_id in doc_ids:
            fname = meta.get(d_id, {}).get("filename", f"Doc-{d_id[:8]}")
            mapping[fname] = d_id
            chunks = list(self.chunks_coll.find({"document_id": d_id}).sort("chunk_index", 1))
            
            full_text_block += f"\n--- FULL DOCUMENT: {fname} (ID: {d_id}) ---\n"
            for c in chunks:
                full_text_block += c.get("text", "") + "\n"
                
        return full_text_block, mapping, time.time() - t_start

    def process_query(self, question: str, context: str = "", label: str = "LLM") -> Tuple[str, Dict[str, Any]]:
        metrics = {"retrieval_time": 0.0, "llm_time": 0.0, "tokens": 0, "doc_count": 0}
        t_start = time.time()
        mapping = {}
        
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
            
            # 4. Cap to TOP_K (5)
            unique_ids = unique_ids[:TOP_K]
            debug_print(f"Retrieval: {len(raw_chunks)} -> Rerank: {len(reranked)} -> Expansion: {len(unique_ids)}")
            
            # 5. Small-to-Big Expansion
            context, mapping, _ = self.get_full_document_text(unique_ids)
            metrics["retrieval_time"] = time.time() - r_start
            metrics["doc_count"] = len(unique_ids)

        prompt = f"TASK: {question}\n\nCONTEXT:\n{context}"
        metrics["tokens"] = count_tokens(prompt)
        
        try:
            l_start = time.time()
            resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json={
                "model": LLM_MODEL, 
                "prompt": prompt,
                "system": SYSTEM_PROMPT, 
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 4000,
                    "num_ctx": 131072,
                    "repeat_penalty": 1.15
                },
            },
            timeout=500)

            resp.raise_for_status()
            answer = resp.json().get("response", "")
            metrics["llm_time"] = time.time() - l_start
        except Exception as e:
            return f"Error: {str(e)}", metrics

        metrics["total_time"] = time.time() - t_start
        metrics["sources"] = mapping
        metrics["context"] = context 
        return answer, metrics

    def close(self):
        self.client.close()