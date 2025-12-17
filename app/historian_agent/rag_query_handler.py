#!/usr/bin/env python3
"""
RAG Query Handler - Schema-Aware Version
Usage: python rag_query_handler.py "What kinds of injuries did firemen get?"
"""

import sys
import os
import logging
import json
import requests
import time
import uuid
import psutil
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pymongo import MongoClient

# Project-specific imports
try:
    from chunking import DocumentChunker
    from embeddings import EmbeddingService
    from vector_store import get_vector_store
    from retrievers import VectorRetriever, KeywordRetriever, HybridRetriever
except ImportError as e:
    print(f"Error: Missing required module: {e}")
    sys.exit(1)

# ============================================================================
# GLOBAL CONFIGURATION - MASTER CONTROLS
# ============================================================================
TOP_K = 100  # Set to 100 based on your successful run
MAX_CONTEXT_TOKENS = 15000 # Increased slightly to handle more OCR text
VECTOR_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3

# Schema Mapping (Based on your diagnostic)
CONTEXT_FIELDS = ["ocr_text", "summary", "filename"]

# MongoDB Configuration
MONGO_URI = "mongodb://admin:secret@mongodb:27017/admin"
DB_NAME = "railroad_documents"

# LLM Configuration
LLM_MODEL = "gpt-oss:20b"
OLLAMA_BASE_URL = "http://host.docker.internal:11434"
LLM_TEMPERATURE = 0.2 # Lower temperature for historical accuracy
LLM_MAX_TOKENS = 5000

# Embedding Configuration
EMBEDDING_PROVIDER = "ollama"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"

# System Prompt
SYSTEM_PROMPT = """You are a knowledgeable historian assistant specializing in Baltimore & Ohio Railroad history. 
Base your answers on the provided OCR text and summaries. Cite specific filenames when making claims. 
If information is not in the context, say so clearly."""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# UTILITIES
# ============================================================================

class MemoryTracker:
    @staticmethod
    def get_usage_mb() -> float:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    @staticmethod
    def log(stage: str):
        usage = MemoryTracker.get_usage_mb()
        print(f"--- [MEMORY] {stage:30} : {usage:8.2f} MB ---")

@dataclass
class RAGQueryResult:
    answer: str
    sources: List[Dict[str, Any]]
    retrieval_stats: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ============================================================================
# CORE HANDLER
# ============================================================================

class RAGQueryHandler:
    def __init__(self):
        MemoryTracker.log("Init: Start")
        self.mongo_uri = os.environ.get('MONGO_URI') or MONGO_URI
        self.db_name = DB_NAME
        self.top_k = TOP_K
        
        self._init_components()
        MemoryTracker.log("Init: Components Loaded")
    
    def _init_components(self):
        try:
            self.client = MongoClient(self.mongo_uri)
            self.db = self.client[self.db_name]
            self.chunks_collection = self.db['document_chunks']
            
            self.embedding_service = EmbeddingService(
                provider=EMBEDDING_PROVIDER,
                model=EMBEDDING_MODEL
            )
            self.vector_store = get_vector_store(store_type="chroma")
            
            self.vector_retriever = VectorRetriever(
                vector_store=self.vector_store,
                embedding_service=self.embedding_service,
                mongo_collection=self.chunks_collection,
                top_k=self.top_k
            )
            
            # Use the Context Fields from the top of the script
            class ConfigShim: 
                context_fields = CONTEXT_FIELDS
            
            self.keyword_retriever = KeywordRetriever(
                mongo_collection=self.chunks_collection,
                config=ConfigShim(),
                top_k=self.top_k
            )
            
            self.hybrid_retriever = HybridRetriever(
                vector_retriever=self.vector_retriever,
                keyword_retriever=self.keyword_retriever,
                vector_weight=VECTOR_WEIGHT,
                keyword_weight=KEYWORD_WEIGHT,
                top_k=self.top_k
            )
        except Exception as e:
            logger.error(f"Initialization Failed: {e}")
            raise

    def process_query(self, question: str) -> RAGQueryResult:
        MemoryTracker.log("Query: Start Retrieval")
        start_time = time.time()
        
        documents = self.hybrid_retriever.get_relevant_documents(question)
        MemoryTracker.log(f"Query: Retrieved {len(documents)} chunks")
        
        context, sources = self._assemble_context(documents)
        MemoryTracker.log("Query: Context Assembled")
        
        answer = self._generate_answer(question, context)
        MemoryTracker.log("Query: Answer Generated")
        
        elapsed = time.time() - start_time
        
        return RAGQueryResult(
            answer=answer,
            sources=sources,
            retrieval_stats={
                "num_documents": len(documents),
                "latency": f"{elapsed:.2f}s",
                "top_k": self.top_k
            }
        )

    def _assemble_context(self, documents: List[Any]) -> tuple[str, List[Dict[str, Any]]]:
        context_parts = []
        sources = []
        total_tokens = 0
        
        for i, doc in enumerate(documents):
            meta = doc.metadata
            
            # Content Fallback Chain: page_content -> ocr_text -> summary
            content = doc.page_content or meta.get("ocr_text") or meta.get("summary") or ""
            
            if not content.strip() or len(content) < 10:
                continue

            # Heuristic: 1 token ~= 4 characters
            tokens = len(content) // 4 
            if total_tokens + tokens > MAX_CONTEXT_TOKENS:
                break
            
            # Title Fallback Chain: filename -> relative_path -> document_id
            source_title = meta.get("filename") or meta.get("relative_path") or f"Doc-{meta.get('document_id', 'Unknown')}"
            
            context_parts.append(f"[Document {i+1}: {source_title}]\n{content}\n")
            total_tokens += tokens
            
            sources.append({
                "document_id": str(meta.get("document_id", "")),
                "title": source_title,
                "score": meta.get("score", 0.0)
            })
            
        return "\n".join(context_parts), sources

    def _generate_answer(self, question: str, context: str) -> str:
        if not context:
            return "No relevant historical OCR data or summaries were found to answer this query."

        prompt = f"{SYSTEM_PROMPT}\n\nHistorical Context Documents:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        
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
                timeout=240 # Increased timeout for heavy context
            )
            response.raise_for_status()
            return response.json().get("response", "Error: Empty response.")
        except Exception as e:
            return f"LLM Generation Error: {str(e)}"

    def close(self):
        self.client.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python rag_query_handler.py \"[Question]\"")
        sys.exit(1)

    question = sys.argv[1]
    
    MemoryTracker.log("Process Started")
    handler = RAGQueryHandler()
    
    try:
        result = handler.process_query(question)
        
        print("\n" + "="*80)
        print(f"HISTORIAN AGENT: {question}")
        print("="*80)
        print(f"\n{result.answer}\n")
        print("="*80)
        print(f"STATS: {result.retrieval_stats['num_documents']} chunks | Latency: {result.retrieval_stats['latency']}")
        print("SOURCES (Filenames):")
        # List first 20 sources to keep terminal clean
        for i, src in enumerate(result.sources[:20], 1):
            print(f"  [{i}] {src['title']}")
        if len(result.sources) > 20:
            print(f"  ... and {len(result.sources) - 20} more documents.")
        print("="*80 + "\n")
        
        MemoryTracker.log("Process Finished")
        
    finally:
        handler.close()

if __name__ == "__main__":
    main()