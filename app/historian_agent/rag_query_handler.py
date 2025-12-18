#!/usr/bin/env python3
import sys, os, requests, time
from typing import Dict, List, Any, Tuple
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

# --- Configuration from Environment ---
DEBUG = os.environ.get("DEBUG_MODE", "0") == "1"
TOP_K = int(os.environ.get("HISTORIAN_AGENT_TOP_K", 50))
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-oss:20b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://admin:secret@mongodb:27017/admin")
DB_NAME = os.environ.get("DB_NAME", "railroad_documents")

def count_tokens(text: str) -> int:
    """Approximates token count (1 token ≈ 4 characters)."""
    return len(text) // 4

def debug_print(msg: str, detail: str = None, tokens: int = 0, level: str = "INFO"):
    """Enhanced debug printer with timing and levels."""
    if not DEBUG:
        return
    
    timestamp = time.strftime("%H:%M:%S")
    token_str = f" | 🔋 {tokens} tokens" if tokens > 0 else ""
    icon = "🔍" if level == "INFO" else "⚠️" if level == "WARN" else "🚀"
    
    sys.stderr.write(f"{icon} [{timestamp}] {msg}{token_str}\n")
    if detail:
        # Indent detail for readability
        indented = "\n".join([f"   | {line}" for line in detail.strip().split('\n')[:15]])
        sys.stderr.write(f"{indented}\n")
    sys.stderr.flush()

class RAGQueryHandler:
    def __init__(self):
        debug_print("Initializing RAGQueryHandler...", f"DB: {DB_NAME}\nURI: {MONGO_URI}")
        
        start_init = time.time()
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        self.docs_coll = self.db['documents']
        self.chunks_coll = self.db['document_chunks']
        
        # Delayed imports for performance trace
        debug_print("Loading AI Services & Vector Store components...")
        from embeddings import EmbeddingService
        from vector_store import get_vector_store
        from retrievers import HybridRetriever, VectorRetriever, KeywordRetriever
        
        self.embedding_service = EmbeddingService(provider="ollama", model="qwen3-embedding:0.6b")
        self.vector_store = get_vector_store(store_type="chroma")
        
        v_ret = VectorRetriever(self.vector_store, self.embedding_service, self.chunks_coll, top_k=TOP_K)
        class ConfigShim: context_fields = ["text", "ocr_text"]
        k_ret = KeywordRetriever(self.chunks_coll, ConfigShim(), top_k=TOP_K)
        self.hybrid_retriever = HybridRetriever(v_ret, k_ret, top_k=TOP_K)
        
        debug_print(f"Handler Ready (Init took {time.time() - start_init:.2f}s)")

    def hydrate_parent_metadata(self, doc_ids):
        debug_print(f"Hydrating metadata for {len(doc_ids)} unique documents")
        query_ids = [ObjectId(d) if ObjectId.is_valid(d) else d for d in doc_ids]
        
        t_start = time.time()
        results = {str(doc["_id"]): doc for doc in self.docs_coll.find({"_id": {"$in": query_ids}})}
        
        debug_print(f"Metadata retrieval complete ({time.time() - t_start:.3f}s)", 
                    f"Found matches for: {list(results.keys())}")
        return results

    def get_full_document_text(self, doc_ids: List[str]) -> Tuple[str, Dict[str, str], float]:
        """SMALL-TO-BIG logic: Reconstructs full text from chunks."""
        debug_print("Executing 'Small-to-Big' expansion", f"Target IDs: {doc_ids}")
        t_start = time.time()
        full_text_block = ""
        meta = self.hydrate_parent_metadata(doc_ids)
        mapping = {}
        
        for d_id in doc_ids:
            fname = meta.get(d_id, {}).get("filename", f"Doc-{d_id[:8]}")
            mapping[fname] = d_id
            
            chunks = list(self.chunks_coll.find({"document_id": d_id}).sort("chunk_index", 1))
            debug_print(f"Reassembling {fname}", f"Found {len(chunks)} chunks for expansion.")
            
            full_text_block += f"\n--- FULL DOCUMENT: {fname} (ID: {d_id}) ---\n"
            for c in chunks:
                full_text_block += c.get("text", "") + "\n"
                
        return full_text_block, mapping, time.time() - t_start

    def get_best_field(self, data, field_list, default=""):
        for f in field_list:
            if data.get(f): return str(data[f]).strip()
        return default

    def process_query(self, question: str, context: str = "", label: str = "LLM") -> Tuple[str, Dict[str, Any]]:
        metrics = {"retrieval_time": 0.0, "llm_time": 0.0, "tokens": 0, "doc_count": 0}
        t_total_start = time.time()
        mapping = {}
        
        debug_print(f"Processing Query [{label}]", f"Question: {question}")
        
        if not context:
            debug_print("No context provided. Invoking Hybrid Retriever...")
            r_start = time.time()
            chunks = self.hybrid_retriever.get_relevant_documents(question)
            
            doc_ids = list(set([c.metadata.get("document_id") for c in chunks]))
            debug_print(f"Retriever returned {len(chunks)} chunks across {len(doc_ids)} documents.")
            
            meta = self.hydrate_parent_metadata(doc_ids)
            context_list = []
            for i, c in enumerate(chunks):
                p_id = c.metadata.get("document_id")
                fname = self.get_best_field(meta.get(p_id, {}), ["filename", "title"], f"Doc-{p_id[:8]}")
                mapping[fname] = p_id
                context_list.append(f"--- SOURCE: {fname} ---\n{c.page_content}")
                
            context = "\n\n".join(context_list)
            metrics["retrieval_time"] = time.time() - r_start
            metrics["doc_count"] = len(doc_ids)
            debug_print(f"Context assembly finished ({metrics['retrieval_time']:.2f}s)")

        prompt = f"TASK: {question}\n\nCONTEXT:\n{context}"
        metrics["tokens"] = count_tokens(prompt)
        
        debug_print(f"Dispatching to LLM: {LLM_MODEL}", f"Prompt Snippet: {prompt[:200]}...", tokens=metrics["tokens"])
        
        llm_start = time.time()
        try:
            resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json={
                "model": LLM_MODEL, 
                "prompt": prompt, 
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 4000, "num_ctx": 131072}
            }, timeout=500)
            resp.raise_for_status()
            
            answer = resp.json().get("response", "")
            
            if not answer or len(answer.strip()) == 0:
                raise ValueError(f"LLM {LLM_MODEL} returned empty response. Check model status.")

            metrics["llm_time"] = time.time() - llm_start
            debug_print(f"LLM Response Received ({metrics['llm_time']:.2f}s)", f"Response Length: {len(answer)} chars")
            
        except Exception as e:
            debug_print("LLM Request Failed!", str(e), level="ERROR")
            return f"Error connecting to LLM: {str(e)}", metrics

        metrics["total_time"] = time.time() - t_total_start
        metrics["sources"] = mapping
        return answer, metrics

    def close(self):
        debug_print("Closing MongoDB connections...")
        self.client.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python rag_query_handler.py 'question'")
    
    # Force DEBUG to True for manual execution to see the verbosity
    DEBUG = True 
    
    handler = RAGQueryHandler()
    try:
        query = sys.argv[1]
        ans, met = handler.process_query(query, label="CLI_RUN")
        
        print("\n" + "="*50)
        print(f"FINAL RESULT ({met['total_time']:.1f}s):")
        print("-" * 50)
        print(ans)
        print("="*50)
        
        print("\n📚 SOURCES REFERENCED:")
        for fname, d_id in sorted(met['sources'].items()):
            print(f" • {fname.ljust(30)} (ID: {d_id})")
        print("\n")
        
    finally:
        handler.close()